#include "bsp_wrapper.h"
#include "BSP-OT_header_only.h"

template<int dim> 
BSPOT::Points<dim> UnLinearize(double* data,int n,int d) {
    return Eigen::Map<Eigen::Matrix<double,dim,-1,Eigen::ColMajor>>(data, d, n).template cast<BSPOT::scalar>();
}

template<int dim> 
std::function<BSPOT::scalar(int,int)> makeCost(const BSPOT::Points<dim>& A,const BSPOT::Points<dim>& B,std::string cost){
    if (cost == "sqnorm") {
        return [&](int i,int j) {
            return (A.col(i) - B.col(j)).squaredNorm();
        };
    }
    if (cost == "norm") {
        return [&](int i,int j) {
            return (A.col(i) - B.col(j)).norm();
        };
    }
    return [&](int i,int j) {
        return (A.col(i) - B.col(j)).squaredNorm();
    };
}


template<int dim>
std::vector<BSPOT::BijectiveMatching> computeBSPMatchings_dim(const BSPOT::Points<dim>& A,const BSPOT::Points<dim>& B,int nb_plans, bool gaussian = true){
    using namespace BSPOT;

    if (A.rows() > 64) 
        gaussian = false;
    
    std::vector<BSPOT::BijectiveMatching> plans(nb_plans);
    BijectiveBSPMatching<dim> BSP(A,B);
#pragma omp parallel for
    for (auto& plan : plans) {
        if (gaussian)
            plan = BSP.computeGaussianMatching();
        else
            plan = BSP.computeMatching(true);
    }
    return plans;
}


BSPOT::BijectiveMatching MergeBijections(const std::vector<BSPOT::BijectiveMatching>& matchings,const std::function<BSPOT::scalar(int,int)>& cost,const BSPOT::BijectiveMatching& T0 = BSPOT::BijectiveMatching()) {
    using namespace BSPOT;
    return MergePlansNoPar(matchings,cost,T0);
}


template<int dim>
double BSPOT_wrap_dim(int n, int d, double *X, double *Y, uint64_t nb_plans,std::vector<BSPOT::BijectiveMatching>& plans, BSPOT::BijectiveMatching& plan,std::string cost_name,const BSPOT::BijectiveMatching& T0) {
    auto A = UnLinearize<dim>(X,n,d);
    auto B = UnLinearize<dim>(Y,n,d);
    plans = computeBSPMatchings_dim<dim>(A,B,nb_plans);
    auto cost_func = makeCost<dim>(A,B,cost_name);
    plan = MergeBijections(plans,cost_func,T0);

    return plan.evalMatching(cost_func);
}

double BSPOT_wrap(int n, int d, double *X, double *Y, uint64_t nb_plans, int *plans_ptr, int *final_plan_ptr,const char* cn,int* initial_plan) {
    using namespace BSPOT;

    std::string cost_name(cn);

    BijectiveMatching T0;

    if (initial_plan){
        ints t0(n);
        std::copy(initial_plan,initial_plan+n,t0.begin());
        T0 = BijectiveMatching(t0);
    }


    std::vector<BijectiveMatching> plans;
    BijectiveMatching plan;
    scalar cost;

    switch (d)
    {
        case 2: {cost = BSPOT_wrap_dim<2>(n,d,X,Y,nb_plans,plans,plan,cost_name,T0);break;}
        case 3: {cost = BSPOT_wrap_dim<3>(n,d,X,Y,nb_plans,plans,plan,cost_name,T0);break;}
        case 4: {cost = BSPOT_wrap_dim<4>(n,d,X,Y,nb_plans,plans,plan,cost_name,T0);break;}
        case 5: {cost = BSPOT_wrap_dim<5>(n,d,X,Y,nb_plans,plans,plan,cost_name,T0);break;}
        case 6: {cost = BSPOT_wrap_dim<6>(n,d,X,Y,nb_plans,plans,plan,cost_name,T0);break;}
        case 7: {cost = BSPOT_wrap_dim<7>(n,d,X,Y,nb_plans,plans,plan,cost_name,T0);break;}
        case 8: {cost = BSPOT_wrap_dim<8>(n,d,X,Y,nb_plans,plans,plan,cost_name,T0);break;}
        case 9: {cost = BSPOT_wrap_dim<9>(n,d,X,Y,nb_plans,plans,plan,cost_name,T0);break;}
        case 10: {cost = BSPOT_wrap_dim<10>(n,d,X,Y,nb_plans,plans,plan,cost_name,T0);break;}
        default: {cost = BSPOT_wrap_dim<-1>(n,d,X,Y,nb_plans,plans,plan,cost_name,T0);break;}
    }

    std::copy(plan.getPlan().begin(), plan.getPlan().end(), final_plan_ptr);

    int* dst = plans_ptr;
    for (const auto& p : plans)
    {
        std::copy(p.getPlan().begin(), p.getPlan().end(), dst);
        dst += p.size();  // == M
    }

    return cost;
}


template<int dim>
double MergeBijections_dim(int n, int d, double *X, double *Y, uint64_t nb_plans,const std::vector<BSPOT::BijectiveMatching>& plans, BSPOT::BijectiveMatching& plan,std::string cost_name) {
    auto A = UnLinearize<dim>(X,n,d);
    auto B = UnLinearize<dim>(Y,n,d);
    auto cost_func = makeCost<dim>(A,B,cost_name);
    plan = MergeBijections(plans,cost_func);

    return plan.evalMatching(cost_func);
}


double MergeBijections(int n, int d, double *X, double *Y, uint64_t nb_plans, int *plans_ptr, int *final_plan_ptr,const char* cn) {
    using namespace BSPOT;

    std::string cost_name(cn);

    std::vector<BijectiveMatching> plans(nb_plans);

    for (std::size_t i = 0; i < nb_plans; ++i)
    {
        std::vector<int> bij(n);
        std::copy(
            plans_ptr + i * n,
            plans_ptr + (i + 1) * n,
            bij.begin()
        );
        plans[i] = BijectiveMatching(bij);
    }

    BijectiveMatching plan;
    scalar cost;

    switch (d)
    {
        case 2: {cost = MergeBijections_dim<2>(n,d,X,Y,nb_plans,plans,plan,cost_name);break;}
        case 3: {cost = MergeBijections_dim<3>(n,d,X,Y,nb_plans,plans,plan,cost_name);break;}
        case 4: {cost = MergeBijections_dim<4>(n,d,X,Y,nb_plans,plans,plan,cost_name);break;}
        case 5: {cost = MergeBijections_dim<5>(n,d,X,Y,nb_plans,plans,plan,cost_name);break;}
        case 6: {cost = MergeBijections_dim<6>(n,d,X,Y,nb_plans,plans,plan,cost_name);break;}
        case 7: {cost = MergeBijections_dim<7>(n,d,X,Y,nb_plans,plans,plan,cost_name);break;}
        case 8: {cost = MergeBijections_dim<8>(n,d,X,Y,nb_plans,plans,plan,cost_name);break;}
        case 9: {cost = MergeBijections_dim<9>(n,d,X,Y,nb_plans,plans,plan,cost_name);break;}
        case 10: {cost = MergeBijections_dim<10>(n,d,X,Y,nb_plans,plans,plan,cost_name);break;}
        default: {cost = MergeBijections_dim<-1>(n,d,X,Y,nb_plans,plans,plan,cost_name);break;}
    }

    std::copy(plan.getPlan().begin(), plan.getPlan().end(), final_plan_ptr);
    return cost;
}
