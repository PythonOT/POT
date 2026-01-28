#include "bsp_wrapper.h"
#include "BSP-OT_header_only.h"

template<int dim> 
BSPOT::Points<dim> UnLinearize(double* data,int n,int d) {
    return BSPOT::Points<dim>(Eigen::Map<Eigen::Matrix<double,dim,-1,Eigen::ColMajor>>(data, d, n));
}

template<int dim> 
std::function<BSPOT::scalar(int,int)> makeCost(const BSPOT::Points<dim>& A,const BSPOT::Points<dim>& B,std::string cost){
    if (cost == "sqnorm") {
        return [&](int i,int j) {
            return (A.col(i) - B.col(j)).squaredNorm();
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
            plan = BSP.computeMatching();
    }
    return plans;
}


BSPOT::BijectiveMatching MergeBijections(const std::vector<BSPOT::BijectiveMatching>& matchings,const std::function<BSPOT::scalar(int,int)>& cost) {
    using namespace BSPOT;
return MergePlansNoPar(matchings,cost);
}

double BSPOT_wrap(int n, int d, double *X, double *Y, uint64_t nb_plans, int *plans_ptr, int *final_plan_ptr,std::string cost_name) {

    using namespace BSPOT;

    // std::cout << "load data" << std::endl;

    auto A = UnLinearize<-1>(X,n,d);
    auto B = UnLinearize<-1>(Y,n,d);

    // std::cout << "make cost" << std::endl;

    auto cost_func = makeCost(A,B,cost_name);

    // std::cout << "compute" << std::endl;
    auto plans = computeBSPMatchings_dim<-1>(A,B,nb_plans);

    // std::cout << "merge" << std::endl;
    auto plan = MergeBijections(plans,cost_func);

    
    // std::cout << "copy" << std::endl;
    std::copy(plan.getPlan().begin(), plan.getPlan().end(), final_plan_ptr);

    // std::cout << "copy all" << std::endl;
    int* dst = plans_ptr;
    for (const auto& p : plans)
    {
        std::copy(p.getPlan().begin(), p.getPlan().end(), dst);
        dst += p.size();  // == M
    }



    // switch (d)
    // {
    //     case 1: {auto A = UnLinearize<1>(X,n,D);auto B = UnLinearize<1>(Y,n,d);plans = computeBSPMatchings_dim<1>(A,B,nb_plans); cost = makeCost<1>(A,B,cost_name);break;}
    //     case 2: {auto A = UnLinearize<2>(X,n,D);auto B = UnLinearize<2>(Y,n,d);plans = computeBSPMatchings_dim<2>(A,B,nb_plans); cost = makeCost<2>(A,B,cost_name);break;}
    //     case 3: {auto A = UnLinearize<3>(X,n,D);auto B = UnLinearize<3>(Y,n,d);plans = computeBSPMatchings_dim<3>(A,B,nb_plans); cost = makeCost<3>(A,B,cost_name);break;}
    //     case 4: {auto A = UnLinearize<4>(X,n,D);auto B = UnLinearize<4>(Y,n,d);plans = computeBSPMatchings_dim<4>(A,B,nb_plans); cost = makeCost<4>(A,B,cost_name);break;}
    //     case 5: {auto A = UnLinearize<5>(X,n,D);auto B = UnLinearize<5>(Y,n,d);plans = computeBSPMatchings_dim<5>(A,B,nb_plans); cost = makeCost<5>(A,B,cost_name);break;}
    //     default: {auto A = UnLinearize<-1>(X,n,D);auto B = UnLinearize<-1>(Y,n,d);plans = computeBSPMatchings_dim<-1>(A,B,nb_plans); cost = makeCost<-1>(A,B,cost_name);break;}
    // }



    return plan.evalMatching(cost_func);
}

