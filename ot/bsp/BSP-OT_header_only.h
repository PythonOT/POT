
// Amalgamation-specific define
#ifndef BSP_OT_HEADER_ONLY
#define BSP_OT_HEADER_ONLY
#endif


#pragma once
#include <Eigen/Dense>
#include <numeric>
#include <vector>
#include <Eigen/Sparse>

#include <filesystem>
#include <ranges>

namespace BSPOT {
namespace fs = std::filesystem;

 using scalar = double;
//using scalar = float;
using scalars = std::vector<scalar>;

using vec = Eigen::Vector3<scalar>;
using vec2 = Eigen::Vector2<scalar>;
using mat2 = Eigen::Matrix2<scalar>;
using mat = Eigen::Matrix3<scalar>;
using mat4 = Eigen::Matrix4<scalar>;
using vec4 = Eigen::Vector4<scalar>;

using Mat = Eigen::Matrix<scalar,-1,-1,Eigen::ColMajor>;
using Diag = Eigen::DiagonalMatrix<scalar,-1>;
using vecs = std::vector<vec>;
using vec2s = std::vector<vec2>;

using triplet = Eigen::Triplet<scalar>;

using ints = std::vector<int>;
using Vec = Eigen::Vector<scalar,-1>;
using Vecs = std::vector<Vec>;

using smat = Eigen::SparseMatrix<scalar>;

using Index = long;
using grid_Index = std::pair<Index,Index>;



inline auto range(int i) {
    return std::views::iota(0,i);
}
inline auto range(int a,int b) {
    return std::views::iota(a,b);
}


inline ints rangeVec(int a,int b) {
    ints rslt(b-a);
    std::iota(rslt.begin(),rslt.end(),a);
    return rslt;
}

inline ints rangeVec(int i) {
    return rangeVec(0,i);
}


template<class T>
using twins = std::pair<T,T>;

inline smat Identity(int V) {
    smat I(V,V);
    I.setIdentity();
    return I;
   }

using Time = std::chrono::high_resolution_clock;
using TimeStamp = std::chrono::time_point<std::chrono::high_resolution_clock>;
using TimeTypeSec = float;
using DurationSec = std::chrono::duration<TimeTypeSec>;

inline TimeTypeSec TimeBetween(const TimeStamp& A,const TimeStamp& B){
    return DurationSec(B-A).count();
}

inline TimeTypeSec TimeFrom(const TimeStamp& A){
    return DurationSec(Time::now()-A).count();
}


template<class T>
bool Smin(T& a,T b) {
    if (b < a){
        a = b;
        return true;
    }
    return false;
}

template<class T>
bool Smax(T& a,T b) {
    if (a < b){
        a = b;
        return true;
    }
    return false;
}

}


// begin --- BSPOT.h --- 

#ifndef BSPOT_H
#define BSPOT_H

namespace BSPOT {

template<int dim>
using Points = Eigen::Matrix<scalar,dim,-1,Eigen::ColMajor>;
template<int dim>
using Vector = Eigen::Vector<scalar,dim>;

using cost_function = std::function<scalar(size_t,size_t)>;
template<class Point>
using geometric_cost = std::function<scalar(const Point&,const Point&)>;

template<int dim>
using CovType = Eigen::Matrix<scalar,dim,dim>;

template<int dim>
struct Moments {
    Vector<dim> mean;
    CovType<dim> Cov;
};


template<int D>
Vector<D> Mean(const Points<D>& X) {
    return X.rowwise().mean();
}

template<int D>
CovType<D> Covariance(const Points<D>& X) {
    Vector<D> mean = X.rowwise().mean();
    Points<D> centered = X.colwise() - mean;
    CovType<D> rslt = centered * centered.adjoint() / double(X.cols());
    return rslt;
}

template<int D>
CovType<D> Covariance(const Points<D>& X,const Points<D>& Y) {
    Vector<D> meanX = Mean(X);
    Points<D> centeredX = X.colwise() - meanX;
    Vector<D> meanY = Mean(Y);
    Points<D> centeredY = Y.colwise() - meanY;
    CovType<D> rslt = centeredX * centeredY.adjoint() / double(X.cols());
    return rslt;
}


template<int dim>
CovType<dim> sqrt(const CovType<dim> &A) {
    Eigen::SelfAdjointEigenSolver<CovType<dim>> root(A);
    return root.operatorSqrt();
}

template<int dim>
CovType<dim> W2GaussianTransportMap(const CovType<dim>& A,const CovType<dim>& B){
    Eigen::SelfAdjointEigenSolver<CovType<dim>> sasA(A);
    CovType<dim> root_A = sasA.operatorSqrt();
    CovType<dim> inv_root_A = sasA.operatorInverseSqrt();
    CovType<dim> C = root_A * B * root_A;
    C = sqrt(C);
    C = inv_root_A*C*inv_root_A;
    return C;
}


}

#endif // BSPOT_H


// end --- BSPOT.h --- 




// end --- types.h --- 


// begin --- sliced.cpp --- 



// begin --- sliced.h --- 

#ifndef SLICED_H
#define SLICED_H



namespace BSPOT {



template<int static_dim>
ints SlicedAssign(const Points<static_dim>& A,const Points<static_dim>& B) {
    int N = A.cols();
    int dim = A.rows();
    std::vector<std::pair<scalar,int>> dot_mu(N),dot_nu(N);
    Vector<static_dim> d = sampleUnitGaussian<static_dim>(dim);
    for (auto j : range(N)) {
        dot_mu[j] = {d.dot(A.col(j)),j};
        dot_nu[j] = {d.dot(B.col(j)),j};
    }
    std::sort(dot_mu.begin(),dot_mu.end());
    std::sort(dot_nu.begin(),dot_nu.end());
    ints plan(N);
    for (auto j : range(N))
        plan[dot_mu[j].second] = dot_nu[j].second;
    return plan;
}





}

#endif // SLICED_H


// end --- sliced.h --- 





// end --- sliced.cpp --- 



// begin --- coupling.cpp --- 



// begin --- coupling.h --- 

#ifndef COUPLING_H
#define COUPLING_H

namespace BSPOT {

using Coupling = Eigen::SparseMatrix<scalar,Eigen::RowMajor>;

scalar EvalCoupling(const Coupling& pi,const cost_function& cost);

template<int D>
Points<D> CouplingToGrad(const Coupling& pi,const Points<D>& A,const Points<D>& B) {
    Points<D> Grad = Points<D>::Zero(A.rows(),A.cols());
    for (int k = 0;k<pi.outerSize();k++)
        for (Coupling::InnerIterator it(pi,k);it;++it)
            Grad.col(it.row()) += (B.col(it.col()) - A.col(it.row()))*it.value();
    return Grad;
}

struct Atom {
    scalar mass;
    int id;
    bool operator<(const Atom& other) const {
        return dot < other.dot;
    }
    scalar dot;
};


using Atoms = std::vector<Atom>;

inline Vec Mass(const Atoms& A) {
    Vec M(A.size());
    for (auto i : range(A.size()))
        M[i] = A[i].mass;
    return M;
}

inline Atoms FromMass(const Vec& x) {
    Atoms rslt(x.size());
    for (auto i : range(x.size())) {
        rslt[i].mass = x[i];
        rslt[i].id = i;
    }
    return rslt;
}

inline Atoms UniformMass(int n) {
    Atoms rslt(n);
    for (auto i : range(n)) {
        rslt[i].mass = 1./n;
        rslt[i].id = i;
    }
    return rslt;
}



struct arrow {
    scalar mass;
    scalar cost;
};

using mapping = std::unordered_map<int,arrow>;

struct CouplingMerger {

    cost_function cost;

    CouplingMerger(const cost_function& cost) : cost(cost) {}
    CouplingMerger() {}


    bool rotateIfUpdate(std::vector<mapping>& pi,std::vector<mapping>& piI,int a,int b,int ap,int bp) {
        if (a == ap || b == bp)
            return false;
        // if (!pi[a].contains(b) || !pi[ap].contains(bp) || !piI[b].contains(a) || !piI[bp].contains(ap)){
        //     std::cerr << "wrong vertices" << std::endl;;
        //     return false;
        // }
        const auto& T = pi[a][b];
        const auto& Tp = pi[ap][bp];
        const scalar rho1 = T.mass;
        const scalar rho2 = Tp.mass;
        if (rho1 < 1e-8 || rho2 < 1e-8)
            return false;
        const scalar rho = std::min(rho1,rho2);
        const scalar curr_cost = T.cost*rho1 + Tp.cost*rho2;
        scalar cabp = cost(a,bp);
        scalar capb = cost(ap,b);
        const scalar new_cost = T.cost*(rho1 - rho) + Tp.cost*(rho2-rho) + (cabp + capb)*rho;
        if (new_cost < curr_cost) {

            if (rho1 < rho2) {
                // a-b is deleted
                pi[a].erase(b);
                piI[b].erase(a);

                pi[ap][bp].mass -= rho;
                piI[bp][ap].mass -= rho;
            } else {
                pi[ap].erase(bp);
                piI[bp].erase(ap);

                pi[a][b].mass -= rho;
                piI[b][a].mass -= rho;
            }

            pi[a][bp].mass += rho;
            pi[a][bp].cost = cabp;
            piI[bp][a].mass += rho;
            piI[bp][a].cost = cabp;

            pi[ap][b].mass += rho;
            pi[ap][b].cost = capb;
            piI[b][ap].mass += rho;
            piI[b][ap].cost = capb;
            // spdlog::info("old cost {} new cost {}",curr_cost,new_cost);
            return true;
        }

        return false;
    }

    //connect two portions of the tree by an edge
    void connectTree(std::vector<int>& forest, int tip, int parent, int from) {
        //assumes from is an ancestor of tip
        //flips all edges on the path from tip to from
        //connects tip to its new parent
        //beware that this removes the last edge on the path from tip to from
        int previous = parent ;
        int current = tip ;
        while(current != from) {
            int next = forest[current] ;
            forest[current] = previous ;
            previous = current ;
            current = next ;
        }
    }

    void findLoop(const std::vector<int>& forest, int n1, int n2, std::vector<int>& loop) {
        int size = forest.size() ;
        //static marks buffer to find the forest loop
        //TODO benchmark the utility of the static, not thread safe
        static std::vector<int> marked(size, size) ;
        //mark for this run
        //FIXME this may break if more than 2^32 calls are made
        static int mark = 0 ;
        ++mark ;
        //determine the loop between the source and the target
        //TODO benchmark the utility of the static, not thread safe
        static std::vector<int> loop_buf ;
        loop.resize(0) ;
        loop_buf.resize(0) ;
        loop.push_back(n1) ;
        loop_buf.push_back(n2) ;
        marked[n1] = mark ;
        marked[n2] = mark ;
        while(true) {
            int next = forest[loop.back()] ;
            if(next != size) {
                //this side of the path has not reached the root
                if(marked[next] == mark) {
                    //the loop is found, trim the other portion of the loop
                    while(loop_buf.back() != next) {
                        //safety check, ensure the loop is well formed
                        assert(!loop_buf.empty()) ;
                        loop_buf.pop_back() ;
                    }
                    break ;
                } else {
                    marked[next] = mark ;
                }
            } else {
                if(loop_buf.back() == size) {
                    //the edge creates no loop
                    loop.resize(0) ;
                    return ;
                }
            }
            //no loop found yet, grow the loop and swap the portion to grow
            loop.push_back(next) ;
            if(loop_buf.back() != size) {
                loop.swap(loop_buf) ;
            }
        }
        //finalize the loop in a single vector
        if(loop[0] != n1) loop.swap(loop_buf) ;
        for(int node : loop_buf | std::views::reverse) {
            loop.push_back(node) ;
        }
    }

    //mutate the tree to improve the coupling
    void forestImproveLoop(Coupling& coupling, std::vector<int>& forest, std::vector<int>& loop) {
        //problem dimensions
        int n = coupling.rows() ;
        int m = coupling.cols() ;

        //source and target
        int source = loop[0] ;
        int target = loop.back() - n ;

        //safety check, the loop should alternate source and target in equal numbers
        assert(loop.size() % 2 == 0) ;

        //change in transport cost when rotating mass around the loop
        scalar factor = cost(source, target) ;

        //bottlenecks when rotating mass
        //0 => adding mass transfer between loop extremities (always possible)
        //1 => decreasing mass transfer between loop extremities (only if edge already in the coupling)
        scalar bottleneck[2] = {
                                std::numeric_limits<scalar>::infinity(),
            coupling.coeff(source, target)
        } ;
        int bottleneck_edge[2] = {n+m, n+m} ;
        int bottleneck_start[2] = {n+m, n+m} ;
        //iterate over loop edges
        for(std::size_t i = 0; i < loop.size() - 1; ++i) {
            //extremitiex of the edge
            int v1 = loop[i] ;
            int v2 = loop[i+1] ;
            //alternate adding / removing
            scalar c = 2*(i%2) ;
            c -= 1 ; //beware adding -1 above yields havoc because i is unsigned
            //determine whether extremities are sources or targets
            //get transport cost and currently transiting mass
            scalar m = std::numeric_limits<scalar>::infinity() ;
            if(v2 > v1) {
                c *= cost(v1, v2-n) ;
                m = coupling.coeff(v1, v2-n) ;
            } else {
                c *= cost(v2, v1-n) ;
                m = coupling.coeff(v2, v1-n) ;
            }
            //update bottlenecks
            if(m < bottleneck[i%2]) {
                bottleneck[i%2] = m ;
                if(v2 == forest[v1]) {
                    //the bottleneck is such that there is a path source -> ... -> v1 -> v2
                    bottleneck_edge[i%2] = v2 ;
                    bottleneck_start[i%2] = source ;
                } else {
                    //the bottleneck is such that there is a path target -> ... -> v2 -> v1
                    bottleneck_edge[i%2] = v1 ;
                    bottleneck_start[i%2] = target + n ;
                }
            }
            //contribute to the global cost
            factor += c ;
        }

        //determine how mass should rotate around the loop to yield an improvement
        int index = factor > 0 ;
        int direction = -2*index + 1 ;
        if(bottleneck[index] > 0) {
            //improvement when increasing transfer between loop extremities
            //rotate mass in the coupling
            for(std::size_t i = 0; i < loop.size() - 1; ++i) {
                //extremitiex of the edge
                int v1 = loop[i] ;
                int v2 = loop[i+1] ;
                //alternate adding / removing
                scalar c = 2*(i%2) ;
                c -= 1 ;
                c *= direction ;
                if(v2 > v1) {
                    coupling.coeffRef(v1, v2-n) += c*bottleneck[index] ;
                    assert(coupling.coeffRef(v1, v2-n) >= 0) ;
                } else {
                    coupling.coeffRef(v2, v1-n) += c*bottleneck[index] ;
                    assert(coupling.coeffRef(v2, v1-n) >= 0) ;
                }
            }

            //insert the new edge in the coupling
            coupling.coeffRef(source, target) += direction * bottleneck[index] ;

            //update the forest inserting the edge if it is not the bottleneck
            if(bottleneck_edge[index] != n+m) {
                connectTree(forest,
                        bottleneck_start[index],
                        source + target + n - bottleneck_start[index],
                        bottleneck_edge[index]) ;
            }
            //checkForest(forest, n) ;
        }
    }

    void forestTryEdge(Coupling& coupling, std::vector<int>& forest, int source, int target) {
        //problem dimensions
        int n = coupling.rows() ;
        int m = coupling.cols() ;

        //check whether the edge creates a loop
        //TODO benchmark the utility of the static
        static std::vector<int> loop ;
        findLoop(forest, source, target + n, loop) ;

        if(loop.empty()) {
            //no loop created, add the edge
            connectTree(forest, source, target + n, n+m) ;
            //checkForest(forest, n) ;
            return ;
        }

        if(loop.size() == 2) {
            //the edge is already present in the forest
            return ;
        }

        //a loop is created, try improving it
        forestImproveLoop(coupling, forest, loop) ;
    }

    //build a tree from a coupling
    void buildForest(Coupling& coupling, std::vector<int>& forest) {
        //problem dimensions
        int n = coupling.rows() ;
        int m = coupling.cols() ;
        //the forest stores the parents
        //clear provided vector
        forest.resize(0) ;
        //when no parent use n+m
        forest.resize(n+m, n+m) ;

        //list edges to avoid iterator invalidation
        std::vector<std::tuple<int, int, scalar>> edges ;
        std::vector<scalar> max_edge(n, 0) ;
        edges.reserve(coupling.nonZeros()) ;
        for(int source = 0; source < coupling.outerSize(); ++source) {
            for(Coupling::InnerIterator it(coupling, source); it; ++it) {
                edges.emplace_back(source, it.col(), it.value()) ;
                max_edge[source] = std::max(max_edge[source], it.value()) ;
            }
        }

        //sorting directly by decreasing edge values yields better results
        //but it becomes much slower probably because sorting edges by source
        //has a much better memory access pattern
        std::sort(edges.begin(), edges.end(), 
            [&] (auto const& e1, auto const& e2) { 
              auto [s1, t1, v1] = e1 ;
              auto [s2, t2, v2] = e2 ;
              if(s1 == s2) return v1 > v2 ;
              return max_edge[s1] > max_edge[s2] ; 
            }
            ) ;

        for(auto [source, target, value] : edges) {
          //spdlog::info("trying edge {} -> {} with value {}", source, target, -value) ;
          //edge vertices belong to trees
          //if its the same tree, adding the edge may create a loop
          //if a loop exists, it is deleted, improving transport cost
          forestTryEdge(coupling, forest, source, target) ;
        }
    }

    void improveQuads(Coupling& coupling, std::vector<int>& forest) {
      //store neighborhoods
      std::vector<int> source_neighbors ;
      std::vector<int> source_offsets ;
      std::vector<int> target_neighbors ;
      std::vector<int> target_offsets ;

      source_neighbors.reserve(coupling.nonZeros()) ;
      source_offsets.reserve(coupling.outerSize() + 1) ;
      source_offsets.push_back(0) ;
      target_neighbors.resize(coupling.nonZeros()) ;
      target_offsets.resize(coupling.innerSize() + 1, 0) ;

      //source -> target
      for(int source = 0; source < coupling.outerSize(); ++source) {
        for(Coupling::InnerIterator it(coupling, source); it; ++it) {
          int target = it.col() ;
          source_neighbors.push_back(target) ;
          ++target_offsets[target] ;
        }
        source_offsets.push_back(source_neighbors.size()) ;
      }

      //target->source
      for(int target = 1; target < target_offsets.size(); ++target) {
        target_offsets[target] += target_offsets[target-1] ;
      }
      for(int source = 0; source < coupling.outerSize(); ++source) {
        for(Coupling::InnerIterator it(coupling, source); it; ++it) {
          int target = it.col() ;
          --target_offsets[target] ;
          target_neighbors[target_offsets[target]] = source ;
        }
      }

      //list quad edges
      for(int source = 0; source < coupling.outerSize(); ++source) {
        for(Coupling::InnerIterator it(coupling, source); it; ++it) {
          int target = it.col() ;
          //we have a source->target edge
          //try every edge between their respective neighbors
          for(int i = target_offsets[target]; i < target_offsets[target+1]; ++i) {
            for(int j = source_offsets[source]; j < source_offsets[source+1]; ++j) {
              forestTryEdge(coupling, forest, target_neighbors[i], source_neighbors[j]) ;
            }
          }
        }
      }

      //cleanup zeros in the sparse matrix
      coupling = coupling.pruned() ;
    }

    //safety check the tree
    void checkForest(const std::vector<int>& forest, int target_start) {
        int size = forest.size() ;
        //ensure no loop happens
        std::vector<int> marked(size, size) ;
        for(int i = 0; i < size; ++i) {
            int current = i ;
            marked[i] = i ;
            while(forest[current] < size) {
                current = forest[current] ;
                //assert the graph has no loop
                assert(marked[current] != i) ;
                marked[current] = i ;
            }
        }
        //ensure all edges are from source to target
        for(int i = 0; i < size; ++i) {
            int parent = forest[i] ;
            if(parent < size) {
                if(i < target_start) {
                    assert(parent >= target_start) ;
                } else {
                    assert(parent < target_start) ;
                }
            }
        }
    }

    Coupling forestMerge(const std::vector<Coupling>& couplings) {
        Coupling result = couplings[0] ;
        // spdlog::info("initial coupling cost is {}",eval(A,B,result));

        //source size
        int n = result.rows() ;

        //build initial tree
        std::vector<int> forest ;
        buildForest(result, forest) ;
        //checkForest(forest, n) ;

        //merge the other couplings
        for(std::size_t i = 1; i < couplings.size(); ++i) {
            const Coupling& coupling = couplings[i] ;
            //spdlog::info("merging cost {}",eval(A,B,coupling));
            for(int source = 0; source < coupling.outerSize(); ++source) {
                for (Coupling::InnerIterator it(coupling, source); it; ++it) {
                    int target = it.col() ;
                    forestTryEdge(result, forest, source, target) ;
                }
            }
            //spdlog::info("coupling cost is now {}",eval(A,B,result));
            //checkForest(forest, n) ;
        }

        return result.pruned() ;
    }

    Coupling CycleMerge(const std::vector<Coupling>& couplings) {
        std::vector<bool> visited;

        auto pi1 = couplings[0];

        int n = pi1.rows();
        int m = pi1.cols();

        std::vector<std::unordered_map<int,arrow>> edges(n);
        std::vector<std::unordered_map<int,arrow>> edgesI(m);
        for (auto i = 0;i<pi1.outerSize();i++){
            for (Coupling::InnerIterator it(pi1,i);it;++it) {
                int j = it.col();
                scalar c = cost(i,j);
                edges[i][j] = {it.value(),c};
                edgesI[j][i] = {it.value(),c};
            }
        }
        for (auto i : range(1,couplings.size())){
            const auto& pip = couplings[i];
            //spdlog::info("merging cost {}",eval(A,B,pip));
            for (auto a = 0;a<n;a++){
                for (Coupling::InnerIterator it(pip,a);it;++it) {
                    int bp = it.col();
                    bool ok;
                    do {
                        ok = true;
                        for (auto b : edges[a]) {
                            for (auto ap : edgesI[bp]) {
                                if (rotateIfUpdate(edges,edgesI,a,b.first,ap.first,bp)) {
                                    ok = false;
                                    break;
                                }
                            }
                            if (!ok)
                                break;
                        }
                    } while (!ok);
                }
            }
        }

        std::vector<triplet> triplets;
        for (auto i = 0;i<edges.size();i++){
            for (auto j : edges[i]){
                if (j.second.mass > 1e-8) {
                    triplet t(i,j.first,j.second.mass);
                    triplets.push_back(t);
                }
            }
        }
        Coupling pi(n,m);
        pi.setFromTriplets(triplets.begin(),triplets.end());
        return pi;
    }
};


}

#endif // COUPLING_H


// end --- coupling.h --- 




BSPOT::scalar BSPOT::EvalCoupling(const Coupling &pi, const cost_function &cost) {
    scalar W = 0;
    for (int k = 0;k<pi.outerSize();k++)
        for (Coupling::InnerIterator it(pi,k);it;++it)
            W += cost(it.row(),it.col())*it.value();
    return W;
}


// end --- coupling.cpp --- 



// begin --- cloudutils.cpp --- 



// begin --- cloudutils.h --- 

#ifndef CLOUDUTILS_H
#define CLOUDUTILS_H
#include <random>

namespace BSPOT {

inline void NormalizeDyn(Points<-1> &X, scalar dilat = 1)
{
    Vector<-1> min = X.rowwise().minCoeff();
    Vector<-1> max = X.rowwise().maxCoeff();
    Vector<-1> scale = max - min;
    double f = dilat/scale.maxCoeff();
    Vector<-1> c = (min+max)*0.5;
    X.colwise() -= c;
    X *= f;
}


template<int dim>
void Normalize(Points<dim> &X, Vector<dim> offset = Vector<dim>::Zero(dim), scalar dilat = 1)
{
    if (dim == -1) {
        offset = Vector<dim>::Zero(X.rows());
    }
    Vector<dim> min = X.rowwise().minCoeff();
    Vector<dim> max = X.rowwise().maxCoeff();
    Vector<dim> scale = max - min;
    double f = dilat/scale.maxCoeff();
    Vector<dim> c = (min+max)*0.5;
    X.colwise() -= c;
    X *= f;
    X.colwise() += offset;
}


template<int dim>
Points<dim> concat(const Points<dim>& X,const Points<dim>& Y) {
    Points<dim> rslt(X.rows(),X.cols() + Y.cols());
    rslt << X,Y;
    return rslt;
}

template<int dim>
Points<dim> pad(const Points<dim>& X,int target) {
    int n = X.cols();
    Points<dim> rslt(dim,target);
    for (auto i : range(target))
        rslt.col(i) = X.col(rand()%n);
    return rslt;
}


template<int dim>
Points<dim> trunc(const Points<dim>& X,int target) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 g(rd());
    ints I = rangeVec(X.cols());
    ::std::shuffle(I.begin(),I.end(),g);
    Points<dim> rslt(X.rows(),target);
    for (auto i : range(target))
        rslt.col(i) = X.col(I[i]);
    return rslt;
}

template<int dim>
inline Points<dim> ForceToSize(const Points<dim>& X,int target) {
    if (X.size() == target)
        return X;
    if (X.size() < target)
        return pad(X,target);
    return trunc(X,target);
}

}


#endif // CLOUDUTILS_H


// end --- cloudutils.h --- 


#include <random>

/*
namespace BSPOT {

void normalize(Vecs &X, Vec offset, scalar dilat){
    int dim = X[0].size();
    Vec min = Vec::Ones(dim)*1e9;
    Vec max = Vec::Ones(dim)*(-1e9);
    for (const auto& x : X){
        min = min.cwiseMin(x);
        max = max.cwiseMax(x);
    }
    Vec scale = max - min;
    double f = dilat/scale.maxCoeff();
    if (!offset.size())
        offset = Vec::Zero(dim);
    Vec c = (min+max)*0.5;
    for (auto& x : X){
        x = (x-c)*f + offset;
    }
}

Vecs concat(const Vecs &X, const Vecs &Y)
{
    Vecs rslt(X.begin(),X.end());
    rslt.insert(rslt.end(),Y.begin(),Y.end());
    return rslt;
}

Vecs pad(const Vecs &X, int target) {
    int n = X.size();
    Vecs rslt = X;
    while (rslt.size() != target)
        rslt.push_back(X[rand()%X.size()]);
    return rslt;
}

Vecs trunc(const Vecs &X, int target){
    Vecs rslt = X;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(rslt.begin(),rslt.end(),g);
    rslt.resize(target);
    return rslt;
}

void translate(Vecs &X, Vec offset)
{
    for (auto& x : X)
        x += offset;
}

void normalize(Mat &X, Vec offset, scalar dilat)
{
    Vec min = X.colwise().minCoeff();
    Vec max = X.colwise().maxCoeff();
    Vec scale = max - min;
    double f = dilat/scale.maxCoeff();
    if (!offset.size())
        offset = Vec::Zero(X.cols());
    Vec c = (min+max)*0.5;
    X.rowwise() -= c.transpose();
    X *= f;
    X.rowwise() += offset.transpose();

    // for (auto i : range(X.rows()))
        // X.row(i) = (X.row(i)-c).array()*f + offset.array();
}

}

*/


// end --- cloudutils.cpp --- 



// begin --- BijectiveMatching.cpp --- 



// begin --- BijectiveMatching.h --- 

#ifndef BIJECTIVEMATCHING_H
#define BIJECTIVEMATCHING_H

// begin --- data_structures.h --- 

#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H
#include <queue>
#include <vector>
#include <set>
#include <unordered_map>
#include <iostream>

namespace BSPOT {

class UnionFind {
private:
    std::vector<int> parent, rank,componentSize;
public:
    UnionFind(int n);

    int find(int u);

    void unite(int u, int v);

    std::vector<std::vector<int>> getConnectedComponents(int n);
} ;


class StampedPriorityQueue {
private:

    struct stamped_element {
        scalar priority;
        int id;
        int timestamp;
        bool operator<(const stamped_element& other) const {
            return priority < other.priority;
        }
    };
    std::priority_queue<stamped_element> queue;
    std::map<int,int> timestamp;

public:
    void insert(int key, scalar priority);

    std::pair<int, scalar> pop();

    bool empty() const;
};


class StopWatch {
     std::map<std::string,scalar> profiler;
     TimeStamp clock;

public:
     void start() {
        clock = Time::now();
    }

     void reset() {
        profiler.clear();
    }

     void tick(std::string label) {
        if (profiler.find(label) == profiler.end())
            profiler[label] = 0;
        profiler[label] += TimeFrom(clock);
        clock = Time::now();
    }

     void profile(bool relative = true) {
        std::cout << "         STOPWATCH REPORT            " << std::endl;
        scalar s = 0;
        std::vector<std::pair<std::string,scalar>> stamps;
        for (const auto& [key,value] : profiler){
            s += value;
            stamps.push_back({key,value});
        }
        if (!relative)
            s = 1;
        std::sort(stamps.begin(),stamps.end(),[](std::pair<std::string,scalar> a,std::pair<std::string,scalar> b) {
            return a.second > b.second;
        });
        for (auto x : stamps){
            std::cout << x.first << " : " << x.second/s << "\n";
        }
        std::cout << "         END     REPORT            " << std::endl << std::endl;
    }

};

struct Edge {
    int i, j;
    scalar w;
};


class TreeGraph {
public:
    std::vector<std::unordered_map<int, scalar>> adj; // Adjacency list with unordered maps

    TreeGraph(int n) : adj(n) {} // Constructor initializes adjacency list with 'n' vertices

    void addEdge(int u, int v, scalar w) {
        adj[u][v] = w;
        adj[v][u] = w;
    }

    void changeWeight(int u, int v, scalar w) {
        if (u >= adj.size() || v >= adj.size()) return; // Out of bounds check

        auto it = adj[u].find(v);
        if (it != adj[u].end()) {
            it->second = w;
            adj[v][u] = w; // Update the reverse edge as well
        }
    }

    void removeEdge(int u, int v) {
        if (u >= adj.size() || v >= adj.size()) return;

        adj[u].erase(v);
        adj[v].erase(u);
    }

    std::vector<Edge> findPath(int start, int end) {
        if (start >= adj.size() || end >= adj.size()) return {}; // Out of bounds check

        std::unordered_map<int, Edge> parent; // Maps node -> (parent edge)
        std::queue<int> q;
        q.push(start);
        parent[start] = {-1, -1, 0}; // Root has no parent edge

        bool found = false;

        // BFS traversal
        while (!q.empty()) {
            int node = q.front();
            q.pop();

            if (node == end) {
                found = true;
                break; // Stop early when we reach the target
            }

            for (const auto& [neighbor, weight] : adj[node]) {
                if (parent.find(neighbor) == parent.end()) { // Not visited
                    parent[neighbor] = {node, neighbor, weight};
                    q.push(neighbor);
                }
            }
        }

        if (!found) return {}; // No path found

        // Reconstruct the path from end to start
        std::vector<Edge> path;
        int current = end;
        while (parent[current].i != -1) { // -1 means root node
            path.push_back(parent[current]);
            current = parent[current].i;
        }

        std::reverse(path.begin(), path.end()); // Reverse to get correct order
        return path;
    }
};

}


#endif // DATA_STRUCTURES_H


// end --- data_structures.h --- 



// begin --- sampling.h --- 

#pragma once

// begin --- types.h --- 

#include <random>

namespace BSPOT {

struct PCG32
{
    PCG32( ) : x(), key() { seed(0x853c49e6748fea9b, c); }
    PCG32( const uint64_t s, const uint64_t ss= c ) : x(), key() { seed(s, ss); }

    void seed( const uint64_t s, const uint64_t ss= c )
    {
        key= (ss << 1) | 1;

        x= key + s;
        sample();
    }

    unsigned sample( )
    {
        // f(x), fonction de transition
        uint64_t xx= x;
        x= a*x + key;

        // g(x), fonction résultat
        uint32_t tmp= ((xx >> 18u) ^ xx) >> 27u;
        uint32_t r= xx >> 59u;
        return (tmp >> r) | (tmp << ((~r + 1u) & 31));
    }

    // c++ interface
    unsigned operator() ( ) { return sample(); }
    static constexpr unsigned min( ) { return 0; }
    static constexpr unsigned max( ) { return ~unsigned(0); }
    typedef unsigned result_type;

    static constexpr uint64_t a= 0x5851f42d4c957f2d;
    static constexpr uint64_t c= 0xda3e39cb94b95bdb;

    uint64_t x;
    uint64_t key;
};



inline Vecs sampleUnitGaussian(int N,int dim) {
    //static std::mt19937 gen;

    static std::random_device hwseed;
    static PCG32 gen( hwseed(), hwseed() );
    static std::normal_distribution<scalar> dist{0.0,1.0};
    Vecs X(N,Vec(dim));
    for (auto& x : X){
        for (int i = 0;i<dim;i++)
            x(i) = dist(gen);
    }
    return X;
}


inline Vec sampleUnitGaussian(int dim) {
    /*
    static thread_local std::random_device hwseed;
    static thread_local PCG32 rng( hwseed(), hwseed() );
*/
    std::normal_distribution<scalar> dist{0.0,1.0};
    //static thread_local std::mt19937 gen;
    static thread_local std::random_device rd;
    static thread_local std::mt19937 rng(rd());
    Vec X(dim);
    for (int i = 0;i<dim;i++)
        X(i) = dist(rng);
    return X;
}

inline Mat sampleUnitGaussianMat(int n,int dim) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::normal_distribution<scalar> dist{0.0,1.0};
    Mat X(dim,n);
    for (auto i : range(dim))
        for (auto j : range(n))
            X(i,j) = dist(gen);
    return X;
}

inline Mat sampleUnitSphere(int n,int dim) {
    static std::mt19937 gen;
    static std::normal_distribution<scalar> dist{0.0,1.0};
    Mat X(dim,n);
    for (auto i : range(n)){
        for (auto j : range(dim))
            X(j,i) = dist(gen);
        X.col(i).normalize();
    }
    return X;
}

inline Mat sampleUnitSquare(int n,int dim) {
    static std::mt19937 gen;
    static std::normal_distribution<scalar> dist{0.0,1.0};
    Mat X(dim,n);
    for (auto i : range(n)){
        for (auto j : range(dim))
            X(j,i) = dist(gen);
        X.col(i) /= X.col(i).lpNorm<Eigen::Infinity>();
    }
    return X;
}


template<class T>
size_t WeightedRandomChoice(const T& weights) {
    // Random number generator
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // Create a discrete distribution based on the weights
    std::discrete_distribution<> dist(weights.begin(), weights.end());

    // Draw an index based on weights
    return dist(gen);
}

inline Vecs fibonacci_sphere(int n)
{
    static double goldenRatio = (1 + std::sqrt(5.))/2.;
    Vecs FS(n);
    for (int i = 0;i<n;i++){
        double theta = 2 * M_PI * i / goldenRatio;
        double phi = std::acos(1 - 2*(i+0.5)/n);
        FS[i] = vec(cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi));
    }
    return FS;
}

inline Vecs Uniform(int n,int d)
{
    Vecs U(n);
    for (int i = 0;i<n;i++)
        U[i] = Vec::Random(d);
    return U;
}


// Fonction pour générer un point uniforme dans la boule unité en dimension d
inline Vec sample_point_in_unit_ball(int d) {
    static std::mt19937 gen;
    static std::normal_distribution<double> gaussian_dist;
    static std::uniform_real_distribution<double> uniform_dist;
    // Génère un point gaussien aléatoire
    Vec point(d);
    for (int i = 0; i < d; ++i) {
        point[i] = gaussian_dist(gen);
    }

    // Normalisation pour obtenir un point sur la sphère
    point.normalize();

    // Distance aléatoire à l'intérieur de la boule avec distribution uniforme
    double radius = std::pow(uniform_dist(gen), 1.0 / d);

    return point * radius;
}

// Fonction principale pour échantillonner N points dans la boule unité de dimension d
inline Vecs sample_unit_ball(int N, int d,double r = 1,Vec offset = Vec()) {
    Vecs samples(N);
    if (!offset.size())
        offset = Vec::Zero(d);

    for (int i = 0; i < N; ++i)
        samples[i] = sample_point_in_unit_ball(d)*r + offset;

    return samples;
}

inline Vecs sampleGaussian(int dim,int N,const Vec& mean,const Mat& Cov) {
    Vecs X = sampleUnitGaussian(N,dim);
    for (auto& x : X)
        x = Cov*x + mean;
    return X;
}

inline Mat sampleUnitBall(int N,int d) {
    static std::mt19937 gen;
    static std::normal_distribution<double> gaussian_dist;
    static std::uniform_real_distribution<double> uniform_dist;

    Mat X(d,N);
    for (auto i : range(N)){
        Vec point(d);
        for (int j = 0; j < d; ++j) {
            point[j] = gaussian_dist(gen);
        }

        // Normalisation pour obtenir un point sur la sphère
        point.normalize();

        // Distance aléatoire à l'intérieur de la boule avec distribution uniforme
        double radius = std::pow(uniform_dist(gen), 1.0 / d);

        X.col(i) = point * radius;
    }
    return X;
}


template<int D>
inline Points<D> sampleUnitBall(int N,int dim = D) {
    static std::mt19937 gen;
    static std::normal_distribution<double> gaussian_dist;
    static std::uniform_real_distribution<double> uniform_dist;

    Points<D> X(dim, N);
    for (auto i : range(N)){
        Vector<D> point(dim);
        for (int j = 0; j < dim; ++j)
            point[j] = gaussian_dist(gen);

        // Normalisation pour obtenir un point sur la sphère
        point.normalize();

        // Distance aléatoire à l'intérieur de la boule avec distribution uniforme
        double radius = std::pow(uniform_dist(gen), 1.0 / dim);

        X.col(i) = point * radius;
    }
    return X;
}

template<int D>
inline Vector<D> sampleUnitGaussian(int dim = D) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::normal_distribution<double> gaussian_dist(0,1);
    Vector<D> point(dim);
    for (int j = 0; j < dim; ++j)
        point[j] = gaussian_dist(gen);
    return point;
}

inline int randint(int a,int b) {
    static thread_local std::random_device rd;
    static thread_local std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(a,b);
    return dist(gen);
}


}


// end --- sampling.h --- 


#include <fstream>

namespace BSPOT {


class BijectiveMatching
{
public:
    using TransportPlan = ints;

    BijectiveMatching();
    BijectiveMatching(const TransportPlan& T) : plan(T),inverse_plan(getInverse(T)) {}
    BijectiveMatching(const Eigen::Vector<int,-1>& T);

    scalar evalMatching(const cost_function& cost) const;

    template<int D>
    scalar evalMatchingL2(const Points<D>& A,const Points<D>& B) const {
        return (A - B(Eigen::all,plan)).squaredNorm()/A.cols();
    }

    const TransportPlan& getPlan() const;

    size_t operator[](size_t i) const;
    size_t operator()(size_t i) const;
    size_t size() const;
    operator TransportPlan() const;

    BijectiveMatching inverseMatching();

    BijectiveMatching inverseMatching() const;
    bool checkBijectivity() const;

    BijectiveMatching operator()(const BijectiveMatching& other) const;

    template<class T>
    std::vector<T> operator()(const std::vector<T>& X);

    const TransportPlan& getInversePlan();

    bool operator==(const BijectiveMatching& other) const {
        return plan == other.plan;
    }


    static inline bool swapIfUpgrade(ints &T, ints &TI, const ints &TP, int a, const cost_function &cost) {
        int b = T[a];
        int bp = TP[a];
        int ap  = TI[bp];
        if (a == ap || b == bp)
            return false;
        scalar old_cost = cost(a,b) + cost(ap,bp);
        scalar new_cost = cost(a,bp) + cost(ap,b);
        if (new_cost < old_cost) {
            T[a] = bp;
            T[ap] = b;
            TI[bp] = a;
            TI[b] = ap;
            return true;
        }
        return false;
    }

protected:

    BijectiveMatching(const TransportPlan& T,const TransportPlan& TI);
    TransportPlan plan,inverse_plan;

    static TransportPlan getInverse(const TransportPlan& T);
};

BijectiveMatching Merge(const BijectiveMatching &T, const BijectiveMatching &TP, const cost_function &cost,bool verbose = false);

BijectiveMatching MergePlans(const std::vector<BijectiveMatching> &plans,const cost_function &cost,BijectiveMatching T = BijectiveMatching(),bool cycle = true);
BijectiveMatching MergePlansNoPar(const std::vector<BijectiveMatching> &plans,const cost_function &cost,BijectiveMatching T = BijectiveMatching(),bool cycle = true);

bool swapIfUpgradeK(ints &T, ints &TI, const ints &TP, int a,int k, const cost_function &cost);

inline ints rankPlans(const std::vector<BijectiveMatching>& plans,const cost_function& cost) {
    auto start = Time::now();
    std::vector<std::pair<scalar,int>> scores(plans.size());
    for (auto i : range(plans.size())) {
        scores[i].first = plans[i].evalMatching(cost);
        scores[i].second = i;
    }
    std::sort(scores.begin(),scores.end(),[](const auto& a,const auto& b) {
        return a.first < b.first;
    });
    // spdlog::info("sort timing {}",TimeFrom(start));
    ints rslt(scores.size());
    for (auto i : range(scores.size()))
        rslt[i] = scores[i].second;
    return rslt;
}


inline bool checkBijection(const ints& T,const ints& TI) {
    ints I(T.size(),-1);
    for (auto i : range(T.size()))
        I[T[i]] = i;
    bool ok = true;
    for (auto i : range(T.size()))
        if (I[i] == -1){
            std::cerr << "not bijection" << std::endl;;
            ok = false;
        }
    for (auto i : range(T.size()))
        if (TI[T[i]] != i){
            ok = false;
        }
    return ok;
}

inline void checkBijection(const ints& T) {
    ints I(T.size(),-1);
    for (auto i : range(T.size()))
        I[T[i]] = i;
    for (auto i : range(T.size()))
        if (I[i] == -1)
            std::cerr << "not bijection" << std::endl;;
}

BijectiveMatching load_plan(std::string path);

inline void out_plan(std::string out,const BijectiveMatching& T) {
    std::ofstream file(out);
    for (auto i : range(T.size()))
        file << T[i] << "\n";
    file.close();
}


}

#endif // BIJECTIVEMATCHING_H


// end --- BijectiveMatching.h --- 



namespace BSPOT {

BijectiveMatching::BijectiveMatching(){}

BijectiveMatching::BijectiveMatching(const Eigen::Vector<int, -1> &T) {
    plan.resize(T.size());
    for (auto i : range(T.size()))
        plan[i] = T[i];
    inverse_plan = getInverse(plan);
}

scalar BijectiveMatching::evalMatching(const cost_function &cost) const {
    scalar c = 0;
    if (plan.empty()) {
        std::cerr << "tried to eval cost on empty plan!" << std::endl;;
        return 0;
    }

    for (auto i : range(plan.size()))
        c += cost(i,plan.at(i));
    return c/plan.size();
}

const BijectiveMatching::TransportPlan &BijectiveMatching::getPlan() const {return plan;}

size_t BijectiveMatching::operator[](size_t i) const {return plan.at(i);}

size_t BijectiveMatching::operator()(size_t i) const {return plan.at(i);}

size_t BijectiveMatching::size() const {return plan.size();}

BijectiveMatching::operator TransportPlan() const {
    return plan;
}

BijectiveMatching BijectiveMatching::inverseMatching() {
    if (inverse_plan.empty())
        inverse_plan = getInversePlan();
    return BijectiveMatching(inverse_plan,plan);
}

BijectiveMatching BijectiveMatching::inverseMatching() const {
    if (inverse_plan.empty())
        return BijectiveMatching(getInverse(plan),plan);
    return BijectiveMatching(inverse_plan,plan);
}

bool BijectiveMatching::checkBijectivity() const
{
    auto I = getInverse(plan);
    for (auto i : I)
        if (i == -1)
            return false;
    return true;
}

BijectiveMatching BijectiveMatching::operator()(const BijectiveMatching &other) const {
    TransportPlan rslt(other.size());
    for (auto i : range(other.size()))
        rslt[i] = plan[other[i]];
    return rslt;
}

BijectiveMatching::BijectiveMatching(const TransportPlan &T, const TransportPlan &TI) : plan(T),inverse_plan(TI) {}

const BijectiveMatching::TransportPlan &BijectiveMatching::getInversePlan() {
    if (inverse_plan.empty())
        inverse_plan = getInverse(plan);
    return inverse_plan;
}

BijectiveMatching::TransportPlan BijectiveMatching::getInverse(const TransportPlan &T) {
    TransportPlan TI(T.size(),-1);
    for (auto i : range(T.size())){
        TI[T[i]] = i;
    }
    return TI;
}

template<class T>
std::vector<T> BijectiveMatching::operator()(const std::vector<T> &X) {
    std::vector<T> rslt(X.size());
    for (auto i : range(X.size()))
        rslt[plan[i]] = X[i];
    return rslt;
}


BijectiveMatching Merge(const BijectiveMatching &T, const BijectiveMatching &TP, const cost_function &cost, bool verbose) {
    if (T.size() == 0)
        return TP;
    int N = T.size();

    UnionFind UF(N*2);
    for (auto i : range(N)) {
        UF.unite(i,T[i]+N);
        UF.unite(i,TP[i]+N);
    }

    std::unordered_map<int,ints> components;
    for (auto i  = 0;i<N;i++) {
        auto p = UF.find(i);
        components[p].push_back(i);
    }

    BijectiveMatching::TransportPlan rslt = T;
    BijectiveMatching::TransportPlan rsltI = T.inverseMatching();
    BijectiveMatching::TransportPlan Tp = TP;

    std::vector<ints> connected_components(components.size());
    int i = 0;
    for (auto& [p,cc] : components)
        connected_components[i++] = cc;


    for (int k = 0;k<connected_components.size();k++) {
        const auto& c = connected_components[k];

        if (c.size() == 1)
            continue;
        scalar costT = 0,costTP = 0;
        for (auto i : c) {
            costT  += cost(i,T[i]);
            costTP += cost(i,TP[i]);
        }
        if (costTP < costT){
            for (auto i : c)
                std::swap(Tp[i],rslt[i]);
            for (auto i : c)
                rsltI[rslt[i]] = i;
        }
        for (auto i : c)
            BijectiveMatching::swapIfUpgrade(rslt,rsltI,Tp,i,cost);
    }
    return rslt;
}

Vec evalMappings(const BijectiveMatching& T,const cost_function& cost) {
//    return (A - B(Eigen::all,T)).colwise().squaredNorm();
    Vec costs(T.size());
//#pragma omp parallel for
    for (int i = 0;i<T.size();i++) {
        costs[i] = cost(i,T[i]);
    }
    return costs;
}

BijectiveMatching MergePlans(const std::vector<BijectiveMatching> &plans, const cost_function &cost, BijectiveMatching T,bool cycle) {
    int s = 0;
    auto I = true ? rankPlans(plans,cost) : rangeVec(plans.size());
    if (T.size() == 0) {
        T = plans[I[0]];
        s = 1;
    }
    int N = plans[0].size();

    auto C = evalMappings(T,cost);

    ints rslt = T;
    ints rsltI = T.inverseMatching();

    ints sig(N);

    scalar avg_cc_size = 0;

    StopWatch profiler;
    for (auto k : range(s,plans.size())) {
        ints Tp = plans[I[k]];
        ints Tpi = plans[I[k]].inverseMatching();
        auto Cp = evalMappings(Tp,cost);

        for (auto i : range(N))
            sig[i] = Tpi[rslt[i]];

        // profiler.start();

        std::vector<ints> CCs;

        if (cycle) {
            ints visited(N,-1);
            int c = 0;
            for (auto i : range(N)) {
                if (visited[i] != -1)
                    continue;
                int j = i;
                int i0 = i;
                if (sig[j] == i)
                    continue;

                ints CC;
                scalar costT = 0;
                scalar costTP = 0;

                while (visited[j] == -1) {
                    CC.push_back(j);
                    costT  += C[j];
                    costTP += Cp[j];
                    visited[j] = c;
                    j = sig[j];
                }

                if (costTP < costT) {
                    j = i0;
                    do {
                        std::swap(Tp[j],rslt[j]);
                        std::swap(C[j],Cp[j]);
                        j = sig[j];
                    } while (j != i0);
                    j = i0;
                    do {
                        rsltI[rslt[j]] = j;
                        j = sig[j];
                    } while (j != i0);
                }

                c++;
                CCs.push_back(CC);
                avg_cc_size += CC.size();
            }
        } else {
            CCs.push_back(rangeVec(N));
        }
        // profiler.tick("cycle");
        // for (auto a : range(N))
//        spdlog::info("nb cycles {} avg size {}",CCs.size(),avg_cc_size / CCs.size() );
// #pragma omp parallel for
#pragma omp parallel
        {
#pragma omp single
            {
                for (int i = 0; i < CCs.size(); ++i) {
#pragma omp task firstprivate(i)
                    {
                        for (auto a : CCs[i]){
                            // swapIfUpgradeK(rslt,rsltI,Tp,a,3,cost);
                            int b = rslt[a];
                            int bp = Tp[a];
                            int ap  = rsltI[bp];
                            if (a == ap || b == bp)
                                continue;
                            scalar old_cost = C[a] + C[ap];
                            scalar cabp = Cp[a];
                            if (cabp > old_cost)
                                continue;
                            scalar capb = cost(ap,b);
                            if (cabp + capb < old_cost) {
                                rslt[a] = bp;
                                rslt[ap] = b;
                                rsltI[bp] = a;
                                rsltI[b] = ap;
                                C[a] = cabp;
                                C[ap] = capb;
                            }
                        }
                    }
                }
            }
        }
        // for (const auto& cc : CCs)
        // {
        //     std::cout << "cc size " << cc.size() << std::endl;
        // }
        // profiler.tick("greedy");
    }
    // profiler.profile(false);
    return rslt;
}

BijectiveMatching MergePlansNoPar(const std::vector<BijectiveMatching> &plans, const cost_function &cost, BijectiveMatching T,bool cycle) {
    int s = 0;
    auto I = true ? rankPlans(plans,cost) : rangeVec(plans.size());
    if (T.size() == 0) {
        T = plans[I[0]];
        s = 1;
    }
    int N = plans[0].size();

    auto C = evalMappings(T,cost);

    ints rslt = T;
    ints rsltI = T.inverseMatching();

    ints sig(N);

    StopWatch profiler;
    for (auto k : range(s,plans.size())) {
        ints Tp = plans[I[k]];
        ints Tpi = plans[I[k]].inverseMatching();
        auto Cp = evalMappings(Tp,cost);

        for (auto i : range(N))
            sig[i] = Tpi[rslt[i]];

        // profiler.start();

        std::vector<ints> CCs;

        if (cycle) {
            ints visited(N,-1);
            int c = 0;
            for (auto i : range(N)) {
                if (visited[i] != -1)
                    continue;
                int j = i;
                int i0 = i;
                if (sig[j] == i)
                    continue;

                ints CC;
                scalar costT = 0;
                scalar costTP = 0;

                while (visited[j] == -1) {
                    CC.push_back(j);
                    costT  += C[j];
                    costTP += Cp[j];
                    visited[j] = c;
                    j = sig[j];
                }

                if (costTP < costT) {
                    j = i0;
                    do {
                        std::swap(Tp[j],rslt[j]);
                        std::swap(C[j],Cp[j]);
                        j = sig[j];
                    } while (j != i0);
                    j = i0;
                    do {
                        rsltI[rslt[j]] = j;
                        j = sig[j];
                    } while (j != i0);
                }

                c++;
                CCs.push_back(CC);
            }
        } else {
            CCs.push_back(rangeVec(N));
        }
        for (int i = 0; i < CCs.size(); ++i) {
            {
                for (auto a : CCs[i]){
                    // swapIfUpgradeK(rslt,rsltI,Tp,a,3,cost);
                    int b = rslt[a];
                    int bp = Tp[a];
                    int ap  = rsltI[bp];
                    if (a == ap || b == bp)
                        continue;
                    scalar old_cost = C[a] + C[ap];
                    scalar cabp = Cp[a];
                    if (cabp > old_cost)
                        continue;
                    scalar capb = cost(ap,b);
                    if (cabp + capb < old_cost) {
                        rslt[a] = bp;
                        rslt[ap] = b;
                        rsltI[bp] = a;
                        rsltI[b] = ap;
                        C[a] = cabp;
                        C[ap] = capb;
                    }
                }
            }
        }
    }
    return rslt;
}

BijectiveMatching load_plan(std::string path) {
    std::ifstream file(path);
    ints plan;
    while (file) {
        int i;
        file >> i;
        plan.push_back(i);
    }
    //remove last element
    plan.pop_back();
    return plan;
}


template<class T>
inline std::vector<std::vector<T>> getPermutations(std::vector<T> C) {
    std::vector<std::vector<T>> rslt;
    do
    {
        rslt.push_back(C);
    }
    while (std::next_permutation(C.begin(), C.end()));
    return rslt;
}


bool swapIfUpgradeK(ints &plan, ints &inverse_plan, const ints &T, int a, int k, const cost_function &cost)
{
    if (k == 2) {
        return BijectiveMatching::swapIfUpgrade(plan,inverse_plan,T,a,cost);
    }
    scalar s = 0;
    std::set<int> A,TA;
    A.insert(a);
    TA.insert(plan[a]);
    auto i = a;
    for (auto k : range(k-1)) {
        auto j = T[i];
        i = inverse_plan[j];
        A.insert(i);
        TA.insert(j);
    }
    if (TA.size() != A.size() || TA.size() == 1)
        return BijectiveMatching::swapIfUpgrade(plan,inverse_plan,T,a,cost);
    ints TAvec(TA.begin(),TA.end());
    ints Avec(A.begin(),A.end());
    auto Sig = getPermutations(TAvec);
    ints best;
    scalar score = 1e8;

    scalar curr = 0;
    for (auto i : range(A.size()))
        curr += cost(Avec[i],plan[Avec[i]]);

    for (const auto& sig : Sig) {
        scalar c = 0;
        for (auto i : range(sig.size()))
            c += cost(Avec[i],sig[i]);
        if (Smin(score,c))
            best = sig;
    }
    if (score > curr)
        return false;
    for (auto i : range(best.size())){
        plan[Avec[i]] = best[i];
        inverse_plan[best[i]] = Avec[i];
    }
    return true;
}

}


// end --- BijectiveMatching.cpp --- 



// begin --- data_structures.cpp --- 





BSPOT::UnionFind::UnionFind(int n) {
    parent.resize(n);
    rank.resize(n, 0);
    componentSize.resize(n, 1); // Initialize each component size to 1
    for (int i = 0; i < n; ++i) parent[i] = i;
}

int BSPOT::UnionFind::find(int u) {
    if (parent[u] != u) {
        parent[u] = find(parent[u]); // Path compression
    }
    return parent[u];
}

//void UnionFind::unite(int u, int v) {
//    int rootU = find(u);
//    int rootV = find(v);
//    if (rootU != rootV) {
//        if (rank[rootU] > rank[rootV]) {
//            parent[rootV] = rootU;
//        } else if (rank[rootU] < rank[rootV]) {
//            parent[rootU] = rootV;
//        } else {
//            parent[rootV] = rootU;
//            rank[rootU]++;
//        }
//    }
//}


void BSPOT::UnionFind::unite(int x, int y) {
    int rootX = find(x), rootY = find(y);
    if (rootX != rootY) {
        if (rank[rootX] > rank[rootY]) {
            parent[rootY] = rootX;
            componentSize[rootX] += componentSize[rootY];
        } else if (rank[rootX] < rank[rootY]) {
            parent[rootX] = rootY;
            componentSize[rootY] += componentSize[rootX];
        } else {
            parent[rootY] = rootX;
            componentSize[rootX] += componentSize[rootY];
            rank[rootX]++;
        }
    }
}

std::vector<std::vector<int>> BSPOT::UnionFind::getConnectedComponents(int n) {
    std::unordered_map<int, int> rootIndex;  // Maps root -> index in components
    std::vector<std::vector<int>> components;

    // **Step 1: Determine component sizes and allocate memory**
    for (int i = 0; i < n; i++) {
        int root = find(i);
        if (rootIndex.find(root) == rootIndex.end()) {
            rootIndex[root] = components.size();
            components.emplace_back();
            components.back().reserve(componentSize[root]); // Preallocate!
        }
    }

    // **Step 2: Populate components without push_back overhead**
    for (int i = 0; i < n; i++) {
        int root = find(i);
        components[rootIndex[root]].push_back(i);
    }

    return components;
}

void BSPOT::StampedPriorityQueue::insert(int key, scalar priority) {
    int ts = 0;
    if (timestamp.contains(key))
        ts = timestamp[key]+1;
    timestamp[key] = ts;
    queue.push(stamped_element{priority, key, ts});
}

std::pair<int, BSPOT::scalar> BSPOT::StampedPriorityQueue::pop() {
    if (queue.empty())
        return {-1, 0};
    stamped_element e = queue.top();
    queue.pop();
    while (timestamp[e.id] != e.timestamp) {
        if (queue.empty())
            return {-1, 0};
        e = queue.top();
        queue.pop();
    }
    return {e.id, e.priority};
}

bool BSPOT::StampedPriorityQueue::empty() const {
    return queue.empty();
}


// end --- data_structures.cpp --- 



// begin --- InjectiveMatching.cpp --- 



// begin --- InjectiveMatching.h --- 

#ifndef INJECTIVEMATCHING_H
#define INJECTIVEMATCHING_H

namespace BSPOT {

class InjectiveMatching
{
public:
    using TransportPlan = ints;
    using InverseTransportPlan = ints;

    int image_domain_size  = -1;

    InjectiveMatching(int m);
    InjectiveMatching();
    InjectiveMatching(const TransportPlan& T,int m);

    scalar evalMatching(const cost_function& cost) const;

    const TransportPlan& getPlan() const;

    size_t operator[](size_t i) const;
    size_t operator()(size_t i) const;
    size_t size() const;
    operator TransportPlan() const;

    InverseTransportPlan inversePlan();
    InverseTransportPlan inversePlan() const;

    static bool swapIfUpgrade(ints& T,ints& TI,const ints& TP,int a,const cost_function& cost);

    static InjectiveMatching Merge(const InjectiveMatching& T1,const InjectiveMatching& T2,const cost_function& cost);

    InverseTransportPlan getInverse() const;


protected:
    InjectiveMatching(const TransportPlan& T,const TransportPlan& TI);
    TransportPlan plan;
    InverseTransportPlan inverse_plan;

    const TransportPlan& getInversePlan();

};


Vec evalMappings(const InjectiveMatching& T,const cost_function& cost);

InjectiveMatching MergePlans(const std::vector<InjectiveMatching>& plans,const cost_function& cost,InjectiveMatching T = InjectiveMatching());


}
#endif // INJECTIVEMATCHING_H


// end --- InjectiveMatching.h --- 




BSPOT::InjectiveMatching::InjectiveMatching(int m) : image_domain_size(m) {}

BSPOT::InjectiveMatching::InjectiveMatching() {}

BSPOT::InjectiveMatching::InjectiveMatching(const TransportPlan &T, int m) : image_domain_size(m),plan(T) {

}

BSPOT::scalar BSPOT::InjectiveMatching::evalMatching(const cost_function &cost) const {
    scalar c = 0;
    for (auto i : range(plan.size()))
        c += cost(i,plan[i])/plan.size();
    return c;
}

const BSPOT::InjectiveMatching::TransportPlan &BSPOT::InjectiveMatching::getPlan() const {return plan;}

size_t BSPOT::InjectiveMatching::operator[](size_t i) const {return plan[i];}

size_t BSPOT::InjectiveMatching::operator()(size_t i) const {return plan[i];}

size_t BSPOT::InjectiveMatching::size() const {return plan.size();}

BSPOT::InjectiveMatching::operator TransportPlan() const {return plan;}

bool BSPOT::InjectiveMatching::swapIfUpgrade(ints &T, ints &TI, const ints &TP, int a, const cost_function &cost) {
    int b = T[a];
    int bp = TP[a];
    int ap  = TI[bp];
    if (a == ap || b == bp)
        return false;
    if (a == ap || b == bp)
        return false;
    if (ap != -1) {
        if (cost(ap,b) + cost(a,bp) < cost(a,b) + cost(ap,bp) ){
            T[a] = bp;
            T[ap] = b;
            TI[bp] = a;
            TI[b] = ap;
            return true;
        }
    }
    else {
        if (cost(a,bp) < cost(a,b)) {
            T[a] = bp;
            TI[b] = -1;
            TI[bp] = a;
            return true;
        }
    }
    return false;
}

BSPOT::InjectiveMatching::InverseTransportPlan BSPOT::InjectiveMatching::inversePlan() {
    if (inverse_plan.empty())
        inverse_plan = getInverse();
    return inverse_plan;
}

BSPOT::InjectiveMatching::InverseTransportPlan BSPOT::InjectiveMatching::inversePlan() const {
    if (inverse_plan.empty())
        std::cerr << "inverse plan not computed" << std::endl;;
    return inverse_plan;
}

const BSPOT::InjectiveMatching::TransportPlan &BSPOT::InjectiveMatching::getInversePlan() {
    inverse_plan = getInverse();
    return inverse_plan;
}



BSPOT::InjectiveMatching::InverseTransportPlan BSPOT::InjectiveMatching::getInverse() const {
    if (image_domain_size == -1) {
        return {};
    }
    InverseTransportPlan rslt(image_domain_size,-1);
    for (auto i : range(plan.size()))
        rslt[plan[i]] = i;
    return rslt;
}

bool checkValid(const BSPOT::ints &T,const BSPOT::ints& TI) {
    int M = TI.size();
    std::set<int> image;
    for (auto i : BSPOT::range(T.size())) {
        if (T[i] == -1)
            return false;
        image.insert(T[i]);
    }
    if (image.size() != T.size()){
        std::cerr << "not injective" << std::endl;;
        return false;
    }
    for (auto i : BSPOT::range(T.size()))
        if (TI[T[i]] != i){
            std::cerr << "wrong inverse" << std::endl;;
            return false;
        }
    for (auto i : BSPOT::range(M)){
        if (TI[i] != -1 && !image.contains(i)){
            std::cerr << "wrong inverse" << std::endl;;
            return false;
        }
    }
    return true;
}


BSPOT::InjectiveMatching BSPOT::InjectiveMatching::Merge(const InjectiveMatching &T, const InjectiveMatching &TP, const cost_function &cost)
{
    if (T.size() == 0)
        return TP;
    int N = T.size();
    int M = T.image_domain_size;

    UnionFind UF(N + M);
    for (auto i : range(N)) {
        UF.unite(i,T[i]+N);
        UF.unite(i,TP[i]+N);
    }

    std::map<int,ints> components;
    for (auto i  = 0;i<N;i++) {
        auto p = UF.find(i);
        components[p].push_back(i);
    }

    ints rslt = T;
    ints rsltI = T.getInverse();
    ints Tp = TP;

    std::vector<ints> connected_components(components.size());
    int i = 0;
    for (auto& [p,cc] : components)
        connected_components[i++] = cc;


#pragma omp parallel for
    for (int k = 0;k<connected_components.size();k++) {
        const auto& c = connected_components[k];

        if (c.size() == 1)
            continue;
        scalar costT = 0,costTP = 0;
        for (auto i : c) {
            costT  += cost(i,T[i]);
            costTP += cost(i,TP[i]);
        }
        if (costTP < costT){
            for (auto i : c)
                rsltI[rslt[i]] = -1;
            for (auto i : c)
                std::swap(Tp[i],rslt[i]);
            for (auto i : c)
                rsltI[rslt[i]] = i;
        }
        for (auto i : c)
            InjectiveMatching::swapIfUpgrade(rslt,rsltI,Tp,i,cost);
    }
    //checkValid(rslt,rsltI);
    return InjectiveMatching(rslt,M);
}

BSPOT::Vec BSPOT::evalMappings(const BSPOT::InjectiveMatching& T,const BSPOT::cost_function& cost) {
    BSPOT::Vec costs(T.size());
    for (int i = 0;i<T.size();i++)
        costs[i] = cost(i,T[i]);

    return costs;
}


BSPOT::InjectiveMatching BSPOT::MergePlans(const std::vector<InjectiveMatching> &plans, const cost_function &cost, BSPOT::InjectiveMatching T) {
    int s = 0;
    if (T.size() == 0) {
        T = plans[0];
        s = 1;
    }
    int N = plans[0].size();

    auto C = evalMappings(T,cost);

    ints rslt = T;
    ints rsltI = T.getInverse();

    for (auto k : range(s,plans.size())) {

        auto Cp = evalMappings(plans[k],cost);

        const auto& Tp = plans[k];
        for (auto a : range(N))
        {
            int b = rslt[a];
            int bp = Tp[a];
            int ap  = rsltI[bp];
            if (a == ap || b == bp)
                continue;
            if (ap != -1) {
                scalar old_cost = C[a] + C[ap];
                scalar cabp = Cp[a];
                if (cabp > old_cost)
                    continue;
                scalar capb = cost(ap,b);
                if (cabp + capb < old_cost) {
                    rslt[a] = bp;
                    rslt[ap] = b;
                    rsltI[bp] = a;
                    rsltI[b] = ap;
                    C[a] = cabp;
                    C[ap] = capb;
                }
            } else {
                scalar old_cost = C[a];
                scalar cabp = cost(a,bp);
                if (cabp < old_cost) {
                    rslt[a] = bp;
                    rsltI[b] = -1;
                    rsltI[bp] = a;
                }
            }
        }
    }
    return InjectiveMatching(rslt,plans[0].image_domain_size);
}


// end --- InjectiveMatching.cpp --- 



// begin --- PartialBSPMatching.h --- 

#ifndef PARTIALBSPMATCHING_H
#define PARTIALBSPMATCHING_H

namespace BSPOT {

template<int D>
class PartialBSPMatching {
public:
    using TransportPlan = ints;

    using Pts = Points<D>;
    const Pts& A;
    const Pts& B;

protected:
    int dim;
    cost_function cost;

    struct dot_id {
        scalar dot;
        int id;
        bool operator<(const dot_id& other) const {
            return dot < other.dot;
        }
    };

    using ids = std::vector<dot_id>;


    int partition(ids &atoms, int beg, int end, int idx) {
        scalar d = atoms[idx].dot;
        int idmin = beg;
        int idmax = end-1;
        while (idmin < idmax) {
            while (idmin < end && atoms[idmin].dot < d){
                idmin++;
            }
            while (idmax >= beg && atoms[idmax].dot > d)
                idmax--;
            if (idmin >= idmax)
                break;
            if (idmin < idmax)
                std::swap(atoms[idmin],atoms[idmax]);
        }
        return idmin;
    }


    Vector<D> getSlice(ids &idA,ids &idB, int b, int e) {
        return sampleUnitGaussian<D>(dim);
    }

    void computeDots(ids& idA,ids& idB,int begA,int endA,int begB,int endB,const Vector<D>& d) {
        for (auto i : range(begA,endA))
            idA[i].dot = d.dot(A.col(idA[i].id));
        for (auto i : range(begB,endB))
            idB[i].dot = d.dot(B.col(idB[i].id));
    }

    bool random_pivot = true;
    Mat sliceBasis;
    bool hasSliceBasis = false;

    int best_choice(int a,ids& idB,int b,int e) {
        if (e - b == 0) {
            std::cerr << "error gap null" << std::endl;;
        }
        int best = 0;
        scalar score = 1e8;
        for (auto i : range(b,e)) {
            scalar s = cost(a,idB[i].id);
            if (s < score) {
                best = i;
                score = s;
            }
        }
        return best;
    }

    void partialBSPOT(ints& plan,ids &idA, ids &idB, int begA, int endA,int begB,int endB,int height = 0) {
        auto gap = (endA-begA);
        if (gap == 1){
            int a = idA[begA].id;
            plan[a] = idB[best_choice(a,idB,begB,endB)].id;
            return;
        }
        const Vector<D> d = hasSliceBasis ? sliceBasis.col(height % dim) : sampleUnitGaussian<D>(dim);

        computeDots(idA,idB,begA,endA,begB,endB,d);

        int pivotA = random_pivot ? randint(begA+1,endA-1) : begA + (endA-begA)/2;
        std::nth_element(idA.begin()+begA,idA.begin() + pivotA,idA.begin() + endA);

        if (endB - begB == gap) {
            int pivotB = begB + pivotA - begA;
            std::nth_element(idB.begin()+begB,idB.begin() + pivotB,idB.begin() + endB);
            partialBSPOT(plan,idA,idB,begA,pivotA,begB,pivotB,height + 1);
            partialBSPOT(plan,idA,idB,pivotA,endA,pivotB,endB,height + 1);
            return;
        }


        int nb_left = pivotA - begA;
        int nb_right = endA - pivotA;

        std::nth_element(idB.begin()+ begB,idB.begin() + begB + nb_left,idB.begin() + endB);
        std::nth_element(idB.begin() + begB + nb_left,idB.begin() + endB - nb_right,idB.begin() + endB);
  //      std::sort(idB.begin() + begB,idB.begin() + endB);

        int pivotB = best_choice(idA[pivotA].id,idB,begB + nb_left,endB - nb_right);
        pivotB = partition(idB,begB + nb_left,endB - nb_right,pivotB);

        partialBSPOT(plan,idA,idB,begA,pivotA,begB,pivotB,height+1);
        partialBSPOT(plan,idA,idB,pivotA,endA,pivotB,endB,height+1);
    }

public:

    PartialBSPMatching(const Pts& A_,const Pts& B_,const cost_function& c) : A(A_),B(B_),cost(c) {
        if (A.cols() > B.cols()) {
            std::cerr << "B must be the larger cloud" << std::endl;;
        }
        dim = A.rows();
        if (D != -1 && dim != D) {
            std::cerr << "dynamic dimension is different from static one !" << std::endl;;
        }
    }

    InjectiveMatching computePartialMatching(const Eigen::Matrix<scalar,D,D>& M,bool rp = false){
        sliceBasis = M;
        hasSliceBasis = true;
        return computePartialMatching(rp);
    }


    InjectiveMatching computePartialMatching(bool random_pivot = true){
        ids idA(A.cols()),idB(B.cols());
        for (auto i : range(A.cols()))
            idA[i].id = i;
        for (auto i : range(B.cols()))
            idB[i].id = i;

        this->random_pivot = random_pivot;
        ints plan = TransportPlan(A.cols(),-1);
        partialBSPOT(plan,idA,idB,0,A.cols(),0,B.cols());
        std::set<int> image;
        for (auto i : range(A.cols())) {
            if (plan[i] == -1){
				std::cout << "unassigned" << i << std::endl;
		}
            else
                image.insert(plan[i]);
        }
        if (image.size() != A.cols())
            std::cerr << "not injective" << std::endl;;
        return InjectiveMatching(plan,B.cols());
    }

};
}

#endif // PARTIALBSPMATCHING_H


// end --- PartialBSPMatching.h --- 






// begin --- PointCloudIO.h --- 

#ifndef POINTCLOUDIO_H
#define POINTCLOUDIO_H
#include <fstream>
#include <iostream>

namespace BSPOT {

template<int D>
inline Points<D> ReadPointCloud(std::filesystem::path path) {
    std::ifstream infile(path);

    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << path.filename() << std::endl;
        throw std::runtime_error("File not found");
    }

    std::vector<double> data; // Store all values in a single contiguous array
    int rows = 0;
    std::string line;
    int dim = D;

    // First pass: Read the file and store numbers in a vector
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double num;
        int current_cols = 0;

        while (iss >> num) {
            data.push_back(num);
            ++current_cols;
        }


        if (dim == -1)
            dim = current_cols;
        if (current_cols != dim) {
            throw std::runtime_error("Inconsistent dimensions in point cloud file or static dim != point cloud dim");
        }
        ++rows;
    }

    // Second pass: Copy the data into an Eigen matrix
    // where each col is a point
    Points<D> pointCloud(dim, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < dim; ++j) {
            pointCloud(j, i) = data[i * dim + j];
        }
    }
    return pointCloud;
}

template<int D>
void WritePointCloud(std::filesystem::path path,const Points<D>& pts) {
    // each row is a point
    std::ofstream outfile(path);
    if (!outfile.is_open()) {
        std::cerr << "Error opening file: " << path.filename() << std::endl;
        return;
    }

    for (int i = 0; i < pts.cols(); ++i) {
        for (int j = 0; j < pts.rows(); ++j) {
            outfile << pts(j, i);
            if (j < pts.rows() - 1) {
                outfile << " ";
            }
        }
        outfile << "\n";
    }
}

}

#endif // POINTCLOUDIO_H


// end --- PointCloudIO.h --- 



// begin --- BijectiveBSPMatching.h --- 

#ifndef BIJECTIVEBSPMATCHING_H
#define BIJECTIVEBSPMATCHING_H

namespace BSPOT {

template<int D>
class BijectiveBSPMatching {
public:
    using TransportPlan = ints;

    using Pts = BSPOT::Points<D>;
    const Pts& A;
    const Pts& B;
    int dim;

protected:

    struct dot_id {
        scalar dot;
        int id;
        bool operator<(const dot_id& other) const {
            return dot < other.dot;
        }
    };

    using ids = std::vector<dot_id>;
    struct SliceView {
        const ids& id;
        int b,e;

        int operator[](int i) const {return id[b + i].id;}

        int size() const {return e - b;}
    };


    static Moments<D> computeMoments(const Pts& mat,const ids& I,int b,int e) {
        SliceView view(I,b,e);
        thread_local static Pts sub;
        sub = mat(Eigen::all,view);
        Vector<D> mean = sub.rowwise().mean();

        CovType<D> rslt = CovType<D>::Zero(mat.rows(),mat.rows());
        for (auto i : range(sub.cols())){
            Vector<D> c = sub.col(i) - mean;
            rslt += c*c.transpose()/scalar(e-b);
        }
        // Pts centered = sub.colwise() - mean;
        // CovType<D> rslt = centered * centered.adjoint() / double(e-b);
        return {mean,rslt};
    }



    Vector<D> getSlice(ids &idA,ids &idB, int b, int e) {
        return sampleUnitGaussian<D>(dim);
    }

    void BSP(ids& idA,ids& idB,int beg,int end,int pivot,const Vector<D>& d) {

        for (auto i : range(beg,end)) {
            idA[i].dot = d.dot(A.col(idA[i].id));// + sampleUnitGaussian<1>()(0)*0e-3;
            idB[i].dot = d.dot(B.col(idB[i].id));// + sampleUnitGaussian<1>()(0)*0e-3;
        }
        std::nth_element(idA.begin() + beg,idA.begin() + pivot,idA.begin() + end);
        std::nth_element(idB.begin() + beg,idB.begin() + pivot,idB.begin() + end);
    }


    bool random_pivot = true;

    std::pair<Moments<D>,Moments<D>> decomposeMoments(const Pts& X,const Moments<D>& M, const ids& id, int beg, int end,int pivot) {
        scalar alpha = scalar(pivot - beg)/scalar(end - beg);
        scalar beta = 1 - alpha;

        auto [ML,CL] = computeMoments(X,id,beg,pivot);

        Vector<D> MR = (M.mean - alpha*ML)/beta;
        CovType<D> DL = (M.mean - ML)*(M.mean - ML).transpose();
        CovType<D> DR = (M.mean - MR)*(M.mean - MR).transpose();
        CovType<D> CR = CovType<D>(M.Cov - alpha*(CL + DL))/beta - DR;

        return {{ML,CL},{MR,CR}};
    }

    bool init_mode = false;

    Vector<D> DrawEigenVector(const CovType<D> &GT) {
        Eigen::SelfAdjointEigenSolver<CovType<D>> solver(GT);
        return solver.eigenvectors().col(randint(0,dim-1));
    }


    Vector<D> gaussianSlice(const Moments<D>& MA,const Moments<D>& MB) {
        CovType<D> GT = W2GaussianTransportMap(MA.Cov,MB.Cov);
        return DrawEigenVector(GT);
    }


    void gaussianPartialBSPOT(ids &idA, ids &idB, int beg, int end, const Moments<D>& MA,const Moments<D>& MB) {
        auto gap = (end-beg);
        if (gap == 0){
            std::cerr << "end - beg == 0" << std::endl;;
            return;
        }
        if (gap == 1)
            return;
        if (gap < 50) {
            // random_pivot = true;
            // partialBSPOT(idA,idB,beg,end);
            partialOrthogonalBSPOT(idA,idB,beg,end,sampleUnitGaussian<D>(dim));
            // random_pivot = false;
            return;
        }

        const Vector<D> d = gaussianSlice(MA,MB);


        // int pivot = randint(beg + gap/4,beg + gap*3/4);
        int pivot = random_pivot ? randint(beg+1,end-1) : beg + (end-beg)/2;

        // for (auto i : range(beg,end)) {
        //     idA[i].dot = d.dot(A.col(idA[i].id));
        //     idB[i].dot = d.dot(B.col(idB[i].id));
        // }
        // std::nth_element(idA.begin() + beg,idA.begin() + pivot,idA.begin() + end);
        // std::nth_element(idB.begin() + beg,idB.begin() + pivot,idB.begin() + end);
        BSP(idA,idB,beg,end,pivot,d);

        auto SMA = decomposeMoments(A,MA,idA,beg,end,pivot);
        auto SMB = decomposeMoments(B,MB,idB,beg,end,pivot);

        gaussianPartialBSPOT(idA,idB,beg,pivot,SMA.first,SMB.first);
        gaussianPartialBSPOT(idA,idB,pivot,end,SMA.second,SMB.second);
    }

    Mat sliceBasis;
    bool hasSliceBasis = false;

    void partialBSPOT(ids &idA, ids &idB, int beg, int end,int height = 0) {
        auto gap = (end-beg);
        if (gap == 0){
            std::cerr << "end - beg == 0" << std::endl;;
        }
        if (gap == 1){
            return;
        }
        int pivot = random_pivot ? randint(beg+1,end-1) : beg + (end-beg)/2;
        const Vector<D> d = hasSliceBasis ? sliceBasis.col(height % dim) : getSlice(idA,idB,beg,end);
        BSP(idA,idB,beg,end,pivot,d);
        partialBSPOT(idA,idB,beg,pivot,height+1);
        partialBSPOT(idA,idB,pivot,end,height+1);
    }

    void selectBSPOT(std::map<int,int>& T,ids &idA, ids &idB, int beg, int end,std::set<int> targets,int height = 0) {
        auto gap = (end-beg);
        if (gap == 0){
            std::cerr << "end - beg == 0" << std::endl;;
        }
        if (gap == 1){
            if (!targets.contains(idA[beg].id))
                std::cerr << "target not found" << std::endl;;
            T[idA[beg].id] = idB[beg].id;
            return;
        }
        int pivot = random_pivot ? randint(beg+1,end-1) : beg + (end-beg)/2;
        const Vector<D> d = hasSliceBasis ? sliceBasis.col(height % dim) : getSlice(idA,idB,beg,end);
        BSP(idA,idB,beg,end,pivot,d);
        std::set<int> L,R;
        for (auto i : range(beg,pivot))
            if (targets.contains(idA[i].id))
                L.insert(idA[i].id);
        for (auto i : range(pivot,end))
            if (targets.contains(idA[i].id))
                R.insert(idA[i].id);
        if (L.size())
            selectBSPOT(T,idA,idB,beg,pivot,L,height+1);
        if (R.size())
            selectBSPOT(T,idA,idB,pivot,end,R,height+1);
    }



    void partialOrthogonalBSPOT(ids &idA, ids &idB, int beg, int end,Vector<D> prev_slice) {
        auto gap = (end-beg);
        if (gap == 0){
            std::cerr << "end - beg == 0" << std::endl;;
            //return;
        }
        if (gap == 1){
            return;
        }
        int pivot = random_pivot ? randint(beg+1,end-1) : beg + (end-beg)/2;
        Vector<D> d = getSlice(idA,idB,beg,end);
        d -= d.dot(prev_slice)*prev_slice/prev_slice.squaredNorm();
        d.normalized();
        BSP(idA,idB,beg,end,pivot,d);
        partialOrthogonalBSPOT(idA,idB,beg,pivot,d);
        partialOrthogonalBSPOT(idA,idB,pivot,end,d);
    }



public:

    BijectiveBSPMatching(const Pts& A_,const Pts& B_) : A(A_),B(B_) {
        dim = A.rows();
        if (D != -1 && dim != D) {
            std::cerr << "dynamic dimension is different from static one !" << std::endl;;
        }
    }

    std::map<int,int> quickselectTransport(const std::set<int>& targets,const Mat& _sliceBasis) {
        sliceBasis = _sliceBasis;
        hasSliceBasis = true;
        return quickselectTransport(targets);
    }

    std::map<int,int> quickselectTransport(const std::set<int>& targets) {
        ids idA(A.cols()),idB(B.cols());
        for (auto i : range(A.cols())) {
            idA[i].id = i;
            idB[i].id = i;
        }
        std::map<int,int> T;
        selectBSPOT(T,idA,idB,0,A.cols(),targets);
        return T;
    }


    BijectiveMatching computeMatching(bool random_pivot = true){
        ids idA(A.cols()),idB(B.cols());
        for (auto i : range(A.cols())) {
            idA[i].id = i;
            idB[i].id = i;
        }

        this->random_pivot = random_pivot;
        partialBSPOT(idA,idB,0,A.cols());

        ints plan = TransportPlan(A.cols());
        for (int i = 0;i<A.cols();i++)
            plan[idA[i].id] = idB[i].id;
        return BijectiveMatching(plan);
    }

    BijectiveMatching computeOrthogonalMatching(const Mat& _sliceBasis,bool random_pivot_ = true){
        ids idA(A.cols()),idB(B.cols());
        for (auto i : range(A.cols())) {
            idA[i].id = i;
            idB[i].id = i;
        }

        hasSliceBasis = true;
        sliceBasis = _sliceBasis;

        this->random_pivot = random_pivot_;
        partialBSPOT(idA,idB,0,A.cols());

        ints plan = TransportPlan(A.cols());
        for (int i = 0;i<A.cols();i++)
            plan[idA[i].id] = idB[i].id;
        return BijectiveMatching(plan);
    }


    BijectiveMatching computeGaussianMatching(){
        ids idA(A.cols()),idB(B.cols());
        for (auto i : range(A.cols())) {
            idA[i].id = i;
            idB[i].id = i;
        }

        random_pivot = false;

        Vector<D> meanA = A.rowwise().mean();
        Vector<D> meanB = B.rowwise().mean();
        Moments<D> MA = {meanA,Covariance(A)};
        Moments<D> MB = {meanB,Covariance(B)};

        gaussianPartialBSPOT(idA,idB,0,A.cols(),MA,MB);


        ints plan = TransportPlan(A.cols());
        for (int i = 0;i<A.cols();i++)
            plan[idA[i].id] = idB[i].id;
        return BijectiveMatching(plan);
    }

    std::pair<ints,ints> computeGaussianMatchingOrders(){
        ids idA(A.cols()),idB(B.cols());
        for (auto i : range(A.cols())) {
            idA[i].id = i;
            idB[i].id = i;
        }

        random_pivot = false;

        Vector<D> meanA = A.rowwise().mean();
        Vector<D> meanB = B.rowwise().mean();
        Moments<D> MA = {meanA,Covariance(A)};
        Moments<D> MB = {meanB,Covariance(B)};



        // partialBSPOT(idA,idB,0,A.cols());
        // partialOrthogonalBSPOT(idA,idB,0,A.cols(),sampleUnitGaussian<D>(dim));
        gaussianPartialBSPOT(idA,idB,0,A.cols(),MA,MB);
        ints OA(A.cols()),OB(A.cols());
        for (int i = 0;i<A.cols();i++)
            OA[i] = idA[i].id;
        for (int i = 0;i<A.cols();i++)
            OB[i] = idB[i].id;
        return {OA,OB};
    }


};

}

#endif // BIJECTIVEBSPMATCHING_H


// end --- BijectiveBSPMatching.h --- 



// begin --- GeneralBSPMatching.h --- 

#ifndef GENERALBSPMATCHING_H
#define GENERALBSPMATCHING_H
#include <random>

namespace BSPOT {

template<int D>
class GeneralBSPMatching {
public:
protected:
    using Pts = Points<D>;

    int dim;

    const Pts& A;
    const Pts& B;

    Atoms mu,nu;
    Atoms src_mu;
    Atoms src_nu;

    struct CDFSplit {
        int id;
        scalar rho;
    };

    std::vector<triplet> triplets;

    struct atom_split {
        int id = -1;
        scalar mass_left,mass_right;
    };

    Pts Grad;
    scalar W = 0;
    bool random_pivot = true;
    Coupling coupling;

    struct SliceView {
        const Atoms& id;
        int b,e;

        int operator[](int i) const {return id[b + i].id;}

        int size() const {return e - b;}
    };


public:

    GeneralBSPMatching(const Pts& A_,const Atoms& MU,const Pts& B_,const Atoms& NU) : src_mu(MU),src_nu(NU),A(A_),B(B_) {
        dim = A.rows();
        if (D != -1 && dim != D) {
            std::cerr << "dynamic dimension is different from static one !" << std::endl;;
        }
        mu.resize(MU.size());
        nu.resize(NU.size());
        Grad = Pts::Zero(dim,MU.size());
        coupling = Coupling(mu.size(),nu.size());
    }

    GeneralBSPMatching(const Pts& A_,const Pts& B_,bool random_pivot = true) : A(A_),B(B_) {
        dim = A.rows();
        if (D != -1 && dim != D) {
            std::cerr << "dynamic dimension is different from static one !" << std::endl;;
        }
        Grad = Pts::Zero(dim,A.cols());
        coupling = Coupling(A.cols(),B.cols());
    }

protected:


    CDFSplit partition(Atoms &atoms, int beg, int end, int idx) {
        scalar d = atoms[idx].dot;
        int idmin = beg;
        int idmax = end-1;
        scalar sum_min = 0;
        while (idmin < idmax) {
            while (idmin < end && atoms[idmin].dot < d){
                sum_min += atoms[idmin].mass;
                idmin++;
            }
            while (idmax >= beg && atoms[idmax].dot > d)
                idmax--;
            if (idmin >= idmax)
                break;
            if (idmin < idmax)
                std::swap(atoms[idmin],atoms[idmax]);
        }
        return {idmin,sum_min};
    }

    CDFSplit quickCDF(Atoms &atoms, int beg, int end, scalar rho, scalar sum) {
        if (end - beg == 1)
            return {beg,sum};
        int idx = getRandomPivot(beg,end-1);
        auto [p,sum_min] = partition(atoms,beg,end,idx);
        if (sum_min >= rho){
            return quickCDF(atoms,beg,p,rho,sum);
        }
        else
            return quickCDF(atoms,p,end,rho - sum_min,sum + sum_min);
    }

    CDFSplit quickCDF(Atoms &atoms, int beg, int end, scalar rho) {
        return quickCDF(atoms,beg,end,rho,0);
    }

    int dotMedian(const Atoms &atoms, int a, int b, int c) {
        const auto& da = atoms[a].dot;
        const auto& db = atoms[b].dot;
        const auto& dc = atoms[c].dot;
        if ((da >= db && da <= dc) || (da >= dc && da <= db)) return a;
        if ((db >= da && db <= dc) || (db >= dc && db <= da)) return b;
        return c;
    }

    CDFSplit partitionCDF(Atoms &atoms, int beg, int end) {
        if (end - beg == 2) {
            if (atoms[beg].dot > atoms[beg+1].dot)
                std::swap(atoms[beg],atoms[beg+1]);
            return {beg+1,atoms[beg].mass};
        }
        int rand_piv = getRandomPivot(beg+1,end-2);
        int piv = dotMedian(atoms,rand_piv,beg,end-1);
        //spdlog::info("start partition b{} p{} e{}",beg,piv,end);
        return partition(atoms,beg,end,piv);
    }

    atom_split splitCDF(Atoms &atoms, int beg, int end, scalar rho) {
        auto selected = quickCDF(atoms,beg,end,rho);
        scalar mass_left = rho - selected.rho;
        scalar mass_right = atoms[selected.id].mass - mass_left;

        return {selected.id,mass_left,mass_right};
    }

    void computeDots(Atoms &atoms, const Pts &X, int beg, int end, const Vector<D> &d) {
        for (auto i : range(beg,end))
            atoms[i].dot = X.col(atoms[i].id).dot(d) + i*1e-8;
    }

    CovType<D> slice_basis;
    bool slice_basis_computed = false;

    Vector<D> getSlice(const Atoms &m, int begA, int endA, const Atoms &n, int begB, int endB,int h) const
    {
        if (slice_basis_computed)
            return slice_basis.col(h % dim);
        if (endA - begA < 50 || endB - begB < 50)
            return sampleUnitGaussian<D>(dim);
        return sampleUnitGaussian<D>(dim);
        CovType<D> CovA = Cov(A,m,begA,endA);
        CovType<D> CovB = Cov(B,n,begB,endB);
        CovType<D> T = W2GaussianTransportMap(CovA,CovB);
        Eigen::SelfAdjointEigenSolver<Mat> solver(T);
        return solver.eigenvectors().col(getRandomPivot(0,T.cols()-1));
    }

    int getRandomPivot(int beg, int end) const {
        if (beg == end)
            return beg;
        if (end < beg)
            std::cerr << "invalid pivot range" << std::endl;;
        static thread_local std::random_device rd;
        static thread_local std::mt19937 rng(rd());
        std::uniform_int_distribution<int> gen(beg, end); // uniform, unbiased
        return gen(rng);
    }

    bool checkMassLeak(int begA, int endA, int begB, int endB) const {
        scalar sumA = 0,sumB = 0;
        for (auto i : range(begA,endA))
            sumA += mu[i].mass;
        for (auto i : range(begB,endB))
            sumB += nu[i].mass;
        if (std::abs(sumA - sumB) > 1e-8){
            return true;
        }
        return false;
    }

    void partialBSPOT(int begA, int endA, int begB, int endB,int height = 0) {
        int gapA = endA - begA;
        int gapB = endB - begB;

        if (gapA == 0 || gapB == 0){
            std::cerr << "null gap" << std::endl;;
            return;
        }

        //        checkMassLeak(begA,endA,begB,endB);


        if (gapA == 1) {
            for (auto i : range(begB,endB)) {
                if (nu[i].mass < 1e-12)
                    continue;
                Grad.col(mu[begA].id) += (B.col(nu[i].id) - A.col(mu[begA].id))*nu[i].mass;
                triplet t = {mu[begA].id,nu[i].id,nu[i].mass};
                triplets.push_back(t);
            }
            return;
        }
        if (gapB == 1) {
            for (auto i : range(begA,endA)) {
                if (mu[i].mass < 1e-12)
                    continue;
                Grad.col(mu[i].id) += (B.col(nu[begB].id) - A.col(mu[i].id))*mu[i].mass;
                triplet t = {mu[i].id,nu[begB].id,mu[i].mass};
                triplets.push_back(t);
            }
            return;
        }
        const Vector<D> d = getSlice(mu,begA,endA,nu,begB,endB,height);

        computeDots(mu,A,begA,endA,d);
        computeDots(nu,B,begB,endB,d);

        CDFSplit CDFS;
        if (random_pivot) {
            CDFS = partitionCDF(mu,begA,endA);
        }
        else {
            scalar sumA = 0;
            for (auto i : range(begA,endA))
                sumA += mu[i].mass;
            CDFS = quickCDF(mu,begA,endA,0.5*sumA);
            if (CDFS.id == begA) {
                CDFS.rho = mu[CDFS.id].mass;
                CDFS.id++;
            }
        }
        int p = CDFS.id;
        scalar rho = CDFS.rho;
        auto split = splitCDF(nu,begB,endB,rho);
        int splitted_atom = nu[split.id].id;

        nu[split.id].mass = split.mass_left;
        partialBSPOT(begA,p,begB,split.id+1,height + 1);

        nu[split.id].id = splitted_atom;
        nu[split.id].mass = split.mass_right;
        partialBSPOT(p,endA,split.id,endB,height + 1);
    }

    void init() {
        for (auto i : range(src_mu.size()))
            mu[i] = src_mu[i];
        for (auto i : range(src_nu.size()))
            nu[i] = src_nu[i];
        Grad = Pts::Zero(dim,A.cols());
        triplets.clear();
        coupling.setZero();
    }

    void setMeasures(const Atoms &mu_, const Atoms &nu_)
    {
        src_mu = mu_;
        src_nu = nu_;
        mu.resize(mu_.size());
        nu.resize(nu_.size());
    }

    Moments<D> computeMoments(const Pts& X,const Atoms& id,int b,int e) const {
        Vec masses(e-b);
        scalar S = 0;
        for (auto i : range(b,e)){
            masses(i) = id[i].mass;
            S += id[i].mass;
        }
        Eigen::DiagonalMatrix<scalar,-1> M = (masses/S).asDiagonal();
        SliceView view(id,b,e);
        Pts sub = X(Eigen::all,view);
        Pts wsub = sub*M;
        Vector<D> mean = wsub.rowwise().sum();
        Pts centered = sub.colwise() - mean;
        CovType<D> rslt = (centered*M) * centered.adjoint() / double(e-b);
        return {mean,rslt};

    }

    Vector<D> getMean(const Pts &X, const Atoms &id, int b, int e) const
    {
        Vector<D> m = Vector<D>::Zero(dim);
        scalar s = 0;
        for (auto i : range(b,e)) {
            m += X.col(id[i].id)*id[i].mass;
            s += id[i].mass;
        }
        return m/s;
    }

    CovType<D> Cov(const Pts &X, const Atoms &atoms, int b, int e) const
    {
        Vector<D> m = getMean(X,atoms,b,e);
        CovType<D> Cov = CovType<D>::Zero(dim,dim);
        scalar s = 0;
        for (auto i : range(b,e)) {
            Vector<D> x = X.col(atoms[i].id) - m;
            Cov.noalias() += x*x.transpose()*atoms[i].mass;
            s += atoms[i].mass;
        }
        return Cov/s;
    }

public:

    const Coupling &computeCoupling(bool rp = true){
        init();
        random_pivot = rp;
        if (checkMassLeak(0,src_mu.size(),0,src_nu.size())) {
            std::cerr << "cannot compute plan to unbalanced marginals" << std::endl;;
        }
        partialBSPOT(0,src_mu.size(),0,src_nu.size());
        coupling.setFromTriplets(triplets.begin(),triplets.end());
        //coupling.makeCompressed();
        return coupling;
    }

    const Coupling &computeOrthogonalCoupling(const CovType<D>& slice_basis = CovType<D>::Identity(D,D)){
        this->slice_basis = slice_basis;
        slice_basis_computed = true;
        return computeCoupling(false);
    }


    const Pts &computeTransportGradient(bool random_pivot = true){
        init();
        this->random_pivot = random_pivot;
        partialBSPOT(0,src_mu.size(),0,src_nu.size());
        for (auto i : range(src_mu.size()))
            Grad.col(i) /= src_mu[i].mass;
        return Grad;
    }

    const Pts &computeOrthogonalTransportGradient(const CovType<D>& slice_basis = CovType<D>::Identity(D,D),bool rp = false){
        this->slice_basis = slice_basis;
        slice_basis_computed = true;
        return computeTransportGradient(rp);
    }
};

}

#endif // GENERALBSPMATCHING_H


// end --- GeneralBSPMatching.h --- 



// begin --- BSPOTWrapper.h --- 

#ifndef BSPOTWRAPPER_H
#define BSPOTWRAPPER_H

namespace BSPOT {

/*
BijectiveMatching MergePlans(const std::vector<BijectiveMatching>& plans,const cost_function& cost,BijectiveMatching T = BijectiveMatching()) {
    std::vector<std::pair<scalar,int>> scores(plans.size());
#pragma omp parallel for
    for (int i = 0;i<plans.size();i++)
        scores[i] = {plans[i].evalMatching(cost),i};
    //std::sort(scores.begin(),scores.end());
    for (auto i : range(scores.size()))
        T = BSPOT::Merge(T,plans[scores[i].second],cost);
    return T;
}
*/


template<int dim>
BijectiveMatching computeGaussianBSPOT(const Points<dim>& A,const Points<dim>& B,int nb_plans,const cost_function& cost,BijectiveMatching T = BijectiveMatching()) {
    std::vector<BijectiveMatching> plans(nb_plans);
    BijectiveBSPMatching BSP(A,B);
    auto start = Time::now();
#pragma omp parallel for
    for (int i = 0;i<nb_plans;i++)
        plans[i] = BSP.computeGaussianMatching();
    return MergePlans(plans,cost,T,(A.cols() < 500000));
}

template<int dim>
BijectiveMatching computeBijectiveBSPOT(const Points<dim>& A,const Points<dim>& B,int nb_plans,const cost_function& cost,BijectiveMatching T = BijectiveMatching()) {
    std::vector<BijectiveMatching> plans(nb_plans);
    BijectiveBSPMatching BSP(A,B);
    int d = A.rows();
#pragma omp parallel for
    for (int i = 0;i<nb_plans;i++){
        plans[i] = BSP.computeMatching();
    }
    return MergePlans(plans,cost,T);
}

template<int dim>
BijectiveMatching computeBijectiveOrthogonalBSPOT(const Points<dim>& A,const Points<dim>& B,int nb_plans,const cost_function& cost,BijectiveMatching T = BijectiveMatching()) {
    std::vector<BijectiveMatching> plans(nb_plans);
    BijectiveBSPMatching BSP(A,B);
    int d = A.rows();
#pragma omp parallel for
    for (int i = 0;i<nb_plans;i++){
        Mat Q = sampleUnitGaussianMat(d,d);
        Q = Q.fullPivHouseholderQr().matrixQ();
        plans[i] = BSP.computeOrthogonalMatching(Q);
    }
    return MergePlans(plans,cost,T);
}

template<int dim>
Coupling computeBSPOTCoupling(const Points<dim>& A,const Atoms& mu,const Points<dim>& B,const Atoms& nu) {
    GeneralBSPMatching BSP(A,mu,B,nu);
    return BSP.computeCoupling();
}

template<int dim>
Points<dim> computeBSPOTGradient(const Points<dim>& A,const Atoms& mu,const Points<dim>& B,const Atoms& nu,int nb_plans) {
    Points<dim> Grad = Points<dim>::Zero(A.rows(),A.cols());
    int d = A.rows();
#pragma omp parallel for
    for (int i = 0;i<nb_plans;i++) {
        GeneralBSPMatching BSP(A,mu,B,nu);
        Points<dim> Grad_i = BSP.computeTransportGradient();
        #pragma omp critical
        {
            Grad += Grad_i/nb_plans;
        }
    }
    return Grad;
}


template<int dim>
InjectiveMatching computePartialBSPOT(const Points<dim>& A,const Points<dim>& B,int nb_plans,const cost_function& cost,InjectiveMatching T = InjectiveMatching()) {
    std::vector<InjectiveMatching> plans(nb_plans);
    PartialBSPMatching<dim> BSP(A,B,cost);
#pragma omp parallel for
    for (int i = 0;i<nb_plans;i++){
        plans[i] = BSP.computePartialMatching();
    }
    return MergePlans(plans,cost,T);
    // InjectiveMatching plan = T;
    // for (int i = 0;i<nb_plans;i++)
    //     plan = InjectiveMatching::Merge(plan,plans[i],cost);

    // return plan;
}


template<int dim>
InjectiveMatching computePartialOrthogonalBSPOT(const Points<dim>& A,const Points<dim>& B,int nb_plans,const cost_function& cost,InjectiveMatching T = InjectiveMatching()) {
    std::vector<InjectiveMatching> plans(nb_plans);
    PartialBSPMatching<dim> BSP(A,B,cost);
#pragma omp parallel for
    for (int i = 0;i<nb_plans;i++){
        Points<dim> Q = sampleUnitGaussianMat(dim,dim).fullPivHouseholderQr().matrixQ();
        plans[i] = BSP.computePartialMatching(Q,false);
    }
    return MergePlans(plans,cost,T);
    // InjectiveMatching plan = T;
    // for (int i = 0;i<nb_plans;i++)
    //     plan = InjectiveMatching::Merge(plan,plans[i],cost);

    // return plan;
}



}

#endif // BSPOTWRAPPER_H


// end --- BSPOTWrapper.h --- 

