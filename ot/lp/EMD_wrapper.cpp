/* This file is a c++ wrapper function for computing the transportation cost
 * between two vectors given a cost matrix.
 *
 * It was written by Antoine Rolet (2014) and mainly consists of a wrapper
 * of the code written by Nicolas Bonneel available on this page
 *          http://people.seas.harvard.edu/~nbonneel/FastTransport/
 *
 * It was then modified to make it more amenable to python inline calling
 *
 * Please give relevant credit to the original author (Nicolas Bonneel) if
 * you use this code for a publication.
 *
 */


#include "network_simplex_simple.h"
#include "network_simplex_simple_omp.h"
#include "sparse_bipartitegraph.h"
#include "EMD.h"
#include <cstdint>
#include <unordered_map>


int EMD_wrap(int n1, int n2, double *X, double *Y, double *D, double *G,
                double* alpha, double* beta, double *cost, uint64_t maxIter)  {
    // beware M and C are stored in row major C style!!!

    using namespace lemon;
    uint64_t n, m, cur;

    typedef FullBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(Digraph);

    // Get the number of non zero coordinates for r and c
    n=0;
    for (int i=0; i<n1; i++) {
        double val=*(X+i);
        if (val>0) {
            n++;
        }else if(val<0){
			return INFEASIBLE;
		}
    }
    m=0;
    for (int i=0; i<n2; i++) {
        double val=*(Y+i);
        if (val>0) {
            m++;
        }else if(val<0){
			return INFEASIBLE;
		}
    }

    // Define the graph

    std::vector<uint64_t> indI(n), indJ(m);
    std::vector<double> weights1(n), weights2(m);
    Digraph di(n, m);
    NetworkSimplexSimple<Digraph,double,double, node_id_type> net(di, true, (int) (n + m), n * m, maxIter);

    // Set supply and demand, don't account for 0 values (faster)

    cur=0;
    for (uint64_t i=0; i<n1; i++) {
        double val=*(X+i);
        if (val>0) {
            weights1[ cur ] = val;
            indI[cur++]=i;
        }
    }

    // Demand is actually negative supply...

    cur=0;
    for (uint64_t i=0; i<n2; i++) {
        double val=*(Y+i);
        if (val>0) {
            weights2[ cur ] = -val;
            indJ[cur++]=i;
        }
    }


    net.supplyMap(&weights1[0], (int) n, &weights2[0], (int) m);

    // Set the cost of each edge
    int64_t idarc = 0;
    for (uint64_t i=0; i<n; i++) {
        for (uint64_t j=0; j<m; j++) {
            double val=*(D+indI[i]*n2+indJ[j]);
            net.setCost(di.arcFromId(idarc), val);
            ++idarc;
        }
    }


    // Solve the problem with the network simplex algorithm

    int ret=net.run();
    uint64_t i, j;
    if (ret==(int)net.OPTIMAL || ret==(int)net.MAX_ITER_REACHED) {
        *cost = 0;
        Arc a; di.first(a);
        for (; a != INVALID; di.next(a)) {
            i = di.source(a);
            j = di.target(a);
            double flow = net.flow(a);
            *cost += flow * (*(D+indI[i]*n2+indJ[j-n]));
            *(G+indI[i]*n2+indJ[j-n]) = flow;
            *(alpha + indI[i]) = -net.potential(i);
            *(beta + indJ[j-n]) = net.potential(j);
        }

    }


    return ret;
}







int EMD_wrap_omp(int n1, int n2, double *X, double *Y, double *D, double *G,
             double* alpha, double* beta, double *cost, uint64_t maxIter, int numThreads)  {
    // beware M and C are stored in row major C style!!!

    using namespace lemon_omp;
    uint64_t n, m, cur;

    typedef FullBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(Digraph);

    // Get the number of non zero coordinates for r and c
    n=0;
    for (int i=0; i<n1; i++) {
        double val=*(X+i);
        if (val>0) {
            n++;
        }else if(val<0){
            return INFEASIBLE;
        }
    }
    m=0;
    for (int i=0; i<n2; i++) {
        double val=*(Y+i);
        if (val>0) {
            m++;
        }else if(val<0){
            return INFEASIBLE;
        }
    }

    // Define the graph

    std::vector<uint64_t> indI(n), indJ(m);
    std::vector<double> weights1(n), weights2(m);
    Digraph di(n, m);
    NetworkSimplexSimple<Digraph,double,double, node_id_type> net(di, true, (int) (n + m), n * m, maxIter, numThreads);

    // Set supply and demand, don't account for 0 values (faster)

    cur=0;
    for (uint64_t i=0; i<n1; i++) {
        double val=*(X+i);
        if (val>0) {
            weights1[ cur ] = val;
            indI[cur++]=i;
        }
    }

    // Demand is actually negative supply...

    cur=0;
    for (uint64_t i=0; i<n2; i++) {
        double val=*(Y+i);
        if (val>0) {
            weights2[ cur ] = -val;
            indJ[cur++]=i;
        }
    }


    net.supplyMap(&weights1[0], (int) n, &weights2[0], (int) m);

    // Set the cost of each edge
    int64_t idarc = 0;
    for (uint64_t i=0; i<n; i++) {
        for (uint64_t j=0; j<m; j++) {
            double val=*(D+indI[i]*n2+indJ[j]);
            net.setCost(di.arcFromId(idarc), val);
            ++idarc;
        }
    }


    // Solve the problem with the network simplex algorithm

    int ret=net.run();
    uint64_t i, j;
    if (ret==(int)net.OPTIMAL || ret==(int)net.MAX_ITER_REACHED) {
        *cost = 0;
        Arc a; di.first(a);
        for (; a != INVALID; di.next(a)) {
            i = di.source(a);
            j = di.target(a);
            double flow = net.flow(a);
            *cost += flow * (*(D+indI[i]*n2+indJ[j-n]));
            *(G+indI[i]*n2+indJ[j-n]) = flow;
            *(alpha + indI[i]) = -net.potential(i);
            *(beta + indJ[j-n]) = net.potential(j);
        }

    }


    return ret;
}

// ============================================================================
// SPARSE VERSION: Accepts edge list instead of dense cost matrix
// ============================================================================
int EMD_wrap_sparse(
    int n1,
    int n2,
    double *X,
    double *Y,
    uint64_t n_edges,
    uint64_t *edge_sources,
    uint64_t *edge_targets,
    double *edge_costs,
    uint64_t *flow_sources_out,
    uint64_t *flow_targets_out,
    double *flow_values_out,
    uint64_t *n_flows_out,
    double *alpha,
    double *beta,
    double *cost,
    uint64_t maxIter
) {
    using namespace lemon;
    
    uint64_t n = 0;  
    for (int i = 0; i < n1; i++) {
        double val = *(X + i);
        if (val > 0) {
            n++;
        } else if (val < 0) {
            return INFEASIBLE; 
        }
    }
    
    uint64_t m = 0;
    for (int i = 0; i < n2; i++) {
        double val = *(Y + i);
        if (val > 0) {
            m++;
        } else if (val < 0) {
            return INFEASIBLE; 
        }
    }

    std::vector<uint64_t> indI(n);  // indI[graph_idx] = original_source_idx
    std::vector<uint64_t> indJ(m);  // indJ[graph_idx] = original_target_idx
    std::vector<double> weights1(n);  // Source masses (positive only)
    std::vector<double> weights2(m);  // Target masses (negative for demand)
    
    // Create reverse mapping: original_idx → graph_idx
    std::vector<int64_t> source_to_graph(n1, -1);  
    std::vector<int64_t> target_to_graph(n2, -1);
    
    uint64_t cur = 0;
    for (int i = 0; i < n1; i++) {
        double val = *(X + i);
        if (val > 0) {
            weights1[cur] = val;           // Store the mass
            indI[cur] = i;                 // Forward map: graph → original
            source_to_graph[i] = cur;      // Reverse map: original → graph
            cur++;
        }
    }
    
    cur = 0;
    for (int i = 0; i < n2; i++) {
        double val = *(Y + i);
        if (val > 0) {
            weights2[cur] = -val;         
            indJ[cur] = i;                 // Forward map: graph → original
            target_to_graph[i] = cur;      // Reverse map: original → graph
            cur++;
        }
    }
    
    typedef SparseBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(Digraph);

    Digraph di(n, m);  

    std::vector<std::pair<int, int>> edges;  // (source, target) pairs
    std::vector<uint64_t> edge_to_arc;       // edge_to_arc[k] = arc ID for edge k
    std::vector<double> arc_costs;            // arc_costs[arc_id] = cost (for O(1) lookup)
    edges.reserve(n_edges);
    edge_to_arc.reserve(n_edges);

    uint64_t valid_edge_count = 0;
    for (uint64_t k = 0; k < n_edges; k++) {
        int64_t src_orig = edge_sources[k];
        int64_t tgt_orig = edge_targets[k];
        int64_t src = source_to_graph[src_orig];
        int64_t tgt = target_to_graph[tgt_orig];

        if (src >= 0 && tgt >= 0) {
            edges.emplace_back(src, tgt + n);
            edge_to_arc.push_back(valid_edge_count);
            arc_costs.push_back(edge_costs[k]);  // Store cost indexed by arc ID
            valid_edge_count++;
        } else {
            edge_to_arc.push_back(UINT64_MAX);  
        }
    }


    di.buildFromEdges(edges);

    NetworkSimplexSimple<Digraph, double, double, node_id_type> net(
        di, true, (int)(n + m), di.arcNum(), maxIter
    );

    net.supplyMap(&weights1[0], (int)n, &weights2[0], (int)m);

    for (uint64_t k = 0; k < n_edges; k++) {
        if (edge_to_arc[k] != UINT64_MAX) {
            net.setCost(edge_to_arc[k], edge_costs[k]);
        }
    }
    
    int ret = net.run();

    if (ret == (int)net.OPTIMAL || ret == (int)net.MAX_ITER_REACHED) {
        *cost = 0;
        *n_flows_out = 0; 
        
        Arc a;
        di.first(a);
        for (; a != INVALID; di.next(a)) {
            uint64_t i = di.source(a); 
            uint64_t j = di.target(a);  
            double flow = net.flow(a);
            
            uint64_t orig_i = indI[i];
            uint64_t orig_j = indJ[j - n];  


            double arc_cost = arc_costs[a]; 

            *cost += flow * arc_cost;
            

            *(alpha + orig_i) = -net.potential(i);
            *(beta + orig_j) = net.potential(j);
            
            if (flow > 1e-15) {  
                flow_sources_out[*n_flows_out] = orig_i;
                flow_targets_out[*n_flows_out] = orig_j;
                flow_values_out[*n_flows_out] = flow;
                (*n_flows_out)++;
            }
        }
    }
    return ret;
}

int EMD_wrap_lazy(int n1, int n2, double *X, double *Y, double *coords_a, double *coords_b, 
                  int dim, int metric, double *G, double *alpha, double *beta, 
                  double *cost, uint64_t maxIter) {
    using namespace lemon;
    typedef FullBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(Digraph);
    
    // Filter source nodes with non-zero weights
    std::vector<int> idx_a;
    std::vector<double> weights_a_filtered;
    std::vector<double> coords_a_filtered;
    
    // Reserve space to avoid reallocations
    idx_a.reserve(n1);
    weights_a_filtered.reserve(n1);
    coords_a_filtered.reserve(n1 * dim);
    
    for (int i = 0; i < n1; i++) {
        if (X[i] > 0) {
            idx_a.push_back(i);
            weights_a_filtered.push_back(X[i]);
            for (int d = 0; d < dim; d++) {
                coords_a_filtered.push_back(coords_a[i * dim + d]);
            }
        }
    }
    int n = idx_a.size();
    
    // Filter target nodes with non-zero weights
    std::vector<int> idx_b;
    std::vector<double> weights_b_filtered;
    std::vector<double> coords_b_filtered;
    
    // Reserve space to avoid reallocations
    idx_b.reserve(n2);
    weights_b_filtered.reserve(n2);
    coords_b_filtered.reserve(n2 * dim);
    
    for (int j = 0; j < n2; j++) {
        if (Y[j] > 0) {
            idx_b.push_back(j);
            weights_b_filtered.push_back(-Y[j]);  // Demand is negative supply
            for (int d = 0; d < dim; d++) {
                coords_b_filtered.push_back(coords_b[j * dim + d]);
            }
        }
    }
    int m = idx_b.size();
    
    if (n == 0 || m == 0) {
        *cost = 0.0;
        return 0;
    }
    
    // Create full bipartite graph
    Digraph di(n, m);
    
    NetworkSimplexSimple<Digraph, double, double, node_id_type> net(
        di, true, (int)(n + m), (uint64_t)(n) * (uint64_t)(m), maxIter
    );
    
    // Set supplies
    net.supplyMap(&weights_a_filtered[0], n, &weights_b_filtered[0], m);
    
    // Enable lazy cost computation - costs will be computed on-the-fly
    net.setLazyCost(&coords_a_filtered[0], &coords_b_filtered[0], dim, metric, n, m);
    
    // Run solver
    int ret = net.run();
    
    if (ret == (int)net.OPTIMAL || ret == (int)net.MAX_ITER_REACHED) {
        *cost = 0;
        
        // Initialize output arrays
        for (int i = 0; i < n1 * n2; i++) G[i] = 0.0;
        for (int i = 0; i < n1; i++) alpha[i] = 0.0;
        for (int i = 0; i < n2; i++) beta[i] = 0.0;
        
        // Extract solution
        Arc a;
        di.first(a);
        for (; a != INVALID; di.next(a)) {
            int i = di.source(a);
            int j = di.target(a) - n;
            
            int orig_i = idx_a[i];
            int orig_j = idx_b[j];
            
            double flow = net.flow(a);
            G[orig_i * n2 + orig_j] = flow;
            
            alpha[orig_i] = -net.potential(i);
            beta[orig_j] = net.potential(j + n);
            
            if (flow > 0) {
                double c = net.computeLazyCost(i, j);
                *cost += flow * c;
            }
        }
    }
    
    return ret;
}
