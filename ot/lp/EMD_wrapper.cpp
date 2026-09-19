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
#include "sparse_bipartitegraph.h"
#include "sparse_digraph.h"
#include "EMD.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace {

struct SetupPolicy {
    bool full_support;
    bool use_arc_mixing;
    bool use_dense_cost_pointer;
};

inline SetupPolicy make_setup_policy(
    uint64_t n,
    uint64_t m,
    int n1,
    int n2,
    bool dense_cost_pointer_supported
) {
    SetupPolicy policy;
    policy.full_support = (n == static_cast<uint64_t>(n1)) && (m == static_cast<uint64_t>(n2));
    policy.use_arc_mixing = !policy.full_support;
    policy.use_dense_cost_pointer = dense_cost_pointer_supported && policy.full_support;
    return policy;
}

template <typename NetType, typename DigraphType>
inline void setup_explicit_arc_costs(
    NetType& net,
    DigraphType& di,
    const double* D,
    int n2,
    const std::vector<uint64_t>& indI,
    const std::vector<uint64_t>& indJ,
    uint64_t n,
    uint64_t m
) {
    int64_t idarc = 0;
    for (uint64_t i = 0; i < n; ++i) {
        for (uint64_t j = 0; j < m; ++j) {
            net.setCost(di.arcFromId(idarc), D[indI[i] * n2 + indJ[j]]);
            ++idarc;
        }
    }
}

template <typename NetType>
inline void setup_warmstart_potentials(
    NetType& net,
    const double* alpha_init,
    const double* beta_init,
    const std::vector<uint64_t>& indI,
    const std::vector<uint64_t>& indJ,
    uint64_t n,
    uint64_t m
) {
    if (alpha_init == nullptr || beta_init == nullptr) return;
    std::vector<double> alpha_compressed(n);
    std::vector<double> beta_compressed(m);
    for (uint64_t i = 0; i < n; ++i) alpha_compressed[i] = alpha_init[indI[i]];
    for (uint64_t j = 0; j < m; ++j) beta_compressed[j] = beta_init[indJ[j]];
    net.setWarmstartPotentials(&alpha_compressed[0], &beta_compressed[0], (int)n, (int)m);
}

template <typename NetType>
inline void extract_dense_full_support(
    const NetType& net,
    const double* D,
    double* G,
    double* alpha,
    double* beta,
    double* cost,
    uint64_t n,
    uint64_t m
) {
    const int node_total = net.nodeNum();
    const int pi_base = node_total - 1;

    for (uint64_t ii = 0; ii < n; ++ii) {
        alpha[ii] = -net._pi[pi_base - static_cast<int>(ii)];
    }
    for (uint64_t jj = 0; jj < m; ++jj) {
        beta[jj] = net._pi[pi_base - static_cast<int>(n + jj)];
    }

    // Only write non-zero entries. G is already zero-initialized in Python.
    const int64_t arc_total = net.arcNum();
    for (int64_t a = 0; a < arc_total; ++a) {
        const double flow = net._flow[a];
        if (flow == 0.0) continue;
        const int64_t d_idx = arc_total - a - 1;  // row-major index in D/G
        *cost += flow * D[d_idx];
        G[d_idx] = flow;
    }
}

template <typename NetType, typename DigraphType, typename InvalidType>
inline void extract_compressed_support(
    const NetType& net,
    DigraphType& di,
    InvalidType invalid,
    const double* D,
    double* G,
    double* alpha,
    double* beta,
    double* cost,
    const std::vector<uint64_t>& indI,
    const std::vector<uint64_t>& indJ,
    uint64_t n,
    int n2
) {
    for (uint64_t ii = 0; ii < n; ++ii) {
        alpha[indI[ii]] = -net.potential(ii);
    }
    for (uint64_t jj = 0; jj < indJ.size(); ++jj) {
        beta[indJ[jj]] = net.potential(jj + n);
    }

    uint64_t i, j;
    typename DigraphType::Arc a;
    di.first(a);
    for (; a != invalid; di.next(a)) {
        i = di.source(a);
        j = di.target(a);
        const double flow = net.flow(a);
        *cost += flow * D[indI[i] * n2 + indJ[j - n]];
        G[indI[i] * n2 + indJ[j - n]] = flow;
    }
}

template <
    typename NetType,
    typename DigraphType,
    typename InvalidType,
    typename SourceIndexVector,
    typename TargetIndexVector,
    typename CostAccessor
>
inline bool extract_sparse_solution(
    const NetType& net,
    DigraphType& di,
    InvalidType invalid,
    const SourceIndexVector& idx_a,
    const TargetIndexVector& idx_b,
    double* alpha,
    double* beta,
    double* cost,
    uint64_t* flow_sources_out,
    uint64_t* flow_targets_out,
    double* flow_values_out,
    uint64_t* n_flows_out,
    uint64_t max_flows_out,
    CostAccessor cost_accessor,
    double min_output_flow
) {
    const int n = static_cast<int>(idx_a.size());

    for (int i = 0; i < n; i++) {
        alpha[static_cast<uint64_t>(idx_a[i])] = -net.potential(i);
    }
    for (int j = 0; j < static_cast<int>(idx_b.size()); j++) {
        beta[static_cast<uint64_t>(idx_b[j])] = net.potential(j + n);
    }

    typename DigraphType::Arc a;
    di.first(a);
    for (; a != invalid; di.next(a)) {
        const int i = di.source(a);
        const int j = di.target(a) - n;
        const double flow = net.flow(a);
        if (flow != 0) {
            *cost += flow * cost_accessor(a, i, j);
        }
        if (flow > min_output_flow) {
            if (*n_flows_out >= max_flows_out) {
                return false;
            }
            flow_sources_out[*n_flows_out] = static_cast<uint64_t>(idx_a[i]);
            flow_targets_out[*n_flows_out] = static_cast<uint64_t>(idx_b[j]);
            flow_values_out[*n_flows_out] = flow;
            ++(*n_flows_out);
        }
    }
    return true;
}

// An arc carrying positive flow, keyed by its head node. Used to decompose
// grid min-cost-flow arc flows (which move mass between *adjacent* grid
// cells) into a direct (source_bin, target_bin, mass) transport plan.
struct GridFlowEdge {
    int head;
    double flow;
};

// Decomposes multi-hop arc flows into direct (source, target, mass) entries.
//
// A min-cost flow on the grid adjacency graph reports how much mass crosses
// each arc between neighbouring cells, but a transport plan has to say which
// bin each unit of mass came from and where it ended up. This walks from
// each node with leftover supply along arcs that still carry flow until it
// reaches one with a deficit, records that path's bottleneck as one plan
// entry, and subtracts it from every arc on the path, repeating until no
// supply is left. `flow_adj` is consumed in place and `rem_supply` is taken
// by value, both as scratch.
//
// Returns false if the plan would exceed `max_plan_entries` entries.
inline bool decompose_grid_flows(
    std::vector<std::vector<GridFlowEdge>>& flow_adj,
    std::vector<double> rem_supply,
    uint64_t* plan_sources_out,
    uint64_t* plan_targets_out,
    double* plan_values_out,
    uint64_t* n_plan_entries_out,
    uint64_t max_plan_entries
) {
    const double eps = 1e-10;
    const std::size_t n_nodes = flow_adj.size();
    std::vector<std::size_t> ptr(n_nodes, 0);

    for (std::size_t src = 0; src < n_nodes; ++src) {
        while (rem_supply[src] > eps) {
            std::vector<std::pair<std::size_t, std::size_t>> path_edges;
            std::size_t cur = src;

            while (true) {
                if (rem_supply[cur] < -eps && cur != src) {
                    break;
                }
                auto& list = flow_adj[cur];
                std::size_t p = ptr[cur];
                while (p < list.size() && list[p].flow <= eps) {
                    ++p;
                }
                ptr[cur] = p;
                if (p >= list.size()) {
                    break;
                }
                path_edges.emplace_back(cur, p);
                cur = static_cast<std::size_t>(list[p].head);
            }

            if (path_edges.empty()) {
                break;
            }

            const std::size_t target = cur;
            if (rem_supply[target] >= -eps) {
                break;
            }

            double bottleneck = rem_supply[src];
            bottleneck = std::min(bottleneck, -rem_supply[target]);
            for (const auto& edge : path_edges) {
                bottleneck = std::min(bottleneck, flow_adj[edge.first][edge.second].flow);
            }

            if (bottleneck <= eps) {
                break;
            }

            for (const auto& edge : path_edges) {
                flow_adj[edge.first][edge.second].flow -= bottleneck;
            }
            rem_supply[src] -= bottleneck;
            rem_supply[target] += bottleneck;

            if (*n_plan_entries_out >= max_plan_entries) {
                return false;
            }
            plan_sources_out[*n_plan_entries_out] = static_cast<uint64_t>(src);
            plan_targets_out[*n_plan_entries_out] = static_cast<uint64_t>(target);
            plan_values_out[*n_plan_entries_out] = bottleneck;
            ++(*n_plan_entries_out);
        }
    }
    return true;
}

} // namespace


int EMD_wrap(int n1, int n2, double *X, double *Y, double *D, double *G,
                double* alpha, double* beta, double *cost, uint64_t maxIter,
                double* alpha_init, double* beta_init)  {
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

    // Define graph and solver
    std::vector<uint64_t> indI(n), indJ(m);
    std::vector<double> weights1(n), weights2(m);
    Digraph di(n, m);
    const SetupPolicy policy = make_setup_policy(n, m, n1, n2, true);
    typedef NetworkSimplexSimple<Digraph, double, double, node_id_type> Simplex;
    Simplex::SimplexOptions simplex_options(policy.use_arc_mixing);
    Simplex net(di, simplex_options, (int) (n + m), n * m, maxIter);

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

    if (policy.use_dense_cost_pointer) {
        net.setDenseCostMatrix(D, n2);
    } else {
        setup_explicit_arc_costs(net, di, D, n2, indI, indJ, n, m);
    }
    setup_warmstart_potentials(net, alpha_init, beta_init, indI, indJ, n, m);
    // Solve the problem with the network simplex algorithm

    int ret=net.run();

    if (ret==(int)net.OPTIMAL || ret==(int)net.MAX_ITER_REACHED) {
        *cost = 0;
        if (policy.full_support) {
            extract_dense_full_support(net, D, G, alpha, beta, cost, n, m);
        } else {
            extract_compressed_support(
                net, di, INVALID, D, G, alpha, beta, cost, indI, indJ, n, n2
            );
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
    uint64_t max_flows_out,
    double *alpha,
    double *beta,
    double *cost,
    uint64_t maxIter,
    double *alpha_init,
    double *beta_init
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

    typedef NetworkSimplexSimple<Digraph, double, double, node_id_type> Simplex;
    Simplex::SimplexOptions simplex_options(true);
    Simplex net(di, simplex_options, (int)(n + m), di.arcNum(), maxIter);

    net.supplyMap(&weights1[0], (int)n, &weights2[0], (int)m);

    for (uint64_t k = 0; k < n_edges; k++) {
        if (edge_to_arc[k] != UINT64_MAX) {
            net.setCost(edge_to_arc[k], edge_costs[k]);
        }
    }
    
    // Initialize warmstart if provided
    if (alpha_init != nullptr && beta_init != nullptr) {
        // Map original indices to graph indices for warmstart
        std::vector<double> alpha_filtered(n);
        std::vector<double> beta_filtered(m);
        for (uint64_t i = 0; i < n; i++) {
            uint64_t orig_i = indI[i];
            alpha_filtered[i] = alpha_init[orig_i];
        }
        for (uint64_t j = 0; j < m; j++) {
            uint64_t orig_j = indJ[j];
            beta_filtered[j] = beta_init[orig_j];
        }
        net.setWarmstartPotentials(&alpha_filtered[0], &beta_filtered[0], n, m);
    }
    
    int ret = net.run();
    if (ret == (int)net.OPTIMAL || ret == (int)net.MAX_ITER_REACHED) {
        *cost = 0;
        *n_flows_out = 0;
        
        auto sparse_cost = [&arc_costs](Arc a, int, int) {
            return arc_costs[a];
        };
        if (!extract_sparse_solution(
                net, di, INVALID, indI, indJ, alpha, beta, cost,
                flow_sources_out, flow_targets_out, flow_values_out,
                n_flows_out, max_flows_out, sparse_cost, 1e-15)) {
            return (int)net.MAX_ITER_REACHED;
        }
    }
    return ret;
}

int EMD_wrap_grid_l1(
    int ndim,
    int64_t *shape,
    double *X,
    double *Y,
    bool return_plan,
    uint64_t *plan_sources_out,
    uint64_t *plan_targets_out,
    double *plan_values_out,
    uint64_t *n_plan_entries_out,
    uint64_t max_plan_entries,
    double *alpha,
    double *cost,
    uint64_t maxIter
) {
    using namespace lemon;

    int64_t n_nodes = 1;
    for (int d = 0; d < ndim; ++d) {
        if (shape[d] <= 0) {
            return INFEASIBLE;
        }
        n_nodes *= shape[d];
    }

    double total_x = 0.0;
    double total_y = 0.0;
    bool any_diff = false;
    for (int64_t i = 0; i < n_nodes; ++i) {
        if (X[i] < 0 || Y[i] < 0) {
            return INFEASIBLE;
        }
        total_x += X[i];
        total_y += Y[i];
        any_diff = any_diff || (X[i] != Y[i]);
    }
    if (std::abs(total_x - total_y) > 1e-8 * std::max(1.0, total_x)) {
        return INFEASIBLE;
    }

    *cost = 0.0;
    *n_plan_entries_out = 0;

    if (!any_diff) {
        // Histograms are identical: the cost is 0 and constant in a
        // neighbourhood of X == Y, so the zero potential is a valid
        // (sub)gradient here.
        std::fill(alpha, alpha + n_nodes, 0.0);
        // Nothing to transport, but if a plan is requested, the identity
        // coupling is still the (trivially optimal) transportation plan.
        if (return_plan) {
            for (int64_t i = 0; i < n_nodes; ++i) {
                if (X[i] > 1e-10) {
                    if (*n_plan_entries_out >= max_plan_entries) {
                        return (int)MAX_ITER_REACHED;
                    }
                    plan_sources_out[*n_plan_entries_out] = static_cast<uint64_t>(i);
                    plan_targets_out[*n_plan_entries_out] = static_cast<uint64_t>(i);
                    plan_values_out[*n_plan_entries_out] = X[i];
                    ++(*n_plan_entries_out);
                }
            }
        }
        return OPTIMAL;
    }

    // Grid-adjacent arcs: one forward and one backward arc per adjacent cell
    // pair, unit cost each. On a unit-spaced Cartesian grid this reduces the
    // cityblock-EMD problem to a min-cost flow on the grid graph, which is
    // far sparser than the full bipartite graph (Ling & Okada, 2007). Unlike
    // that paper's bespoke tree-based solver, the reduced graph below is
    // handed to the off-the-shelf NetworkSimplexSimple LP solver.
    std::vector<int64_t> stride(ndim);
    stride[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        stride[d] = stride[d + 1] * shape[d + 1];
    }

    std::vector<std::pair<int, int>> edges;
    for (int d = 0; d < ndim; ++d) {
        const int64_t extent = shape[d];
        if (extent < 2) {
            continue;
        }
        const int64_t st = stride[d];
        for (int64_t u = 0; u < n_nodes; ++u) {
            if ((u / st) % extent < extent - 1) {
                edges.emplace_back(static_cast<int>(u), static_cast<int>(u + st));
                edges.emplace_back(static_cast<int>(u + st), static_cast<int>(u));
            }
        }
    }

    typedef SparseDigraph Digraph;
    Digraph di(static_cast<int>(n_nodes));
    di.buildFromEdges(edges);
    const int64_t total_arcs = static_cast<int64_t>(edges.size());

    std::vector<double> supply(n_nodes);
    for (int64_t i = 0; i < n_nodes; ++i) {
        supply[i] = X[i] - Y[i];
    }

    typedef NetworkSimplexSimple<Digraph, double, double, node_id_type> Simplex;
    Simplex::SimplexOptions simplex_options(true);
    Simplex net(di, simplex_options, static_cast<int>(n_nodes), total_arcs, maxIter);
    net.supplyMap(supply);
    for (int64_t k = 0; k < total_arcs; ++k) {
        net.setCost(Digraph::arcFromId(k), 1.0);
    }

    int ret = net.run();
    if (ret != (int)net.OPTIMAL && ret != (int)net.MAX_ITER_REACHED) {
        return ret;
    }

    *cost = net.totalCost();

    // Node potentials (dual variables) are a byproduct of the solve, cheap
    // to extract regardless of whether a plan was requested: dW/dX[i] =
    // alpha[i], dW/dY[i] = -alpha[i] (beta = -alpha, since supply[i] =
    // X[i] - Y[i] uses a single graph, not a bipartite source/target split).
    // Negated to match LEMON's sign convention, same as the bipartite
    // extract_compressed_support above (alpha = -potential).
    for (int64_t i = 0; i < n_nodes; ++i) {
        alpha[i] = -net.potential(Digraph::nodeFromId(static_cast<int>(i)));
    }

    if (!return_plan) {
        // The caller only wants the cost: skip decomposing the Beckmann-style
        // arc flow into a transportation plan (coupling) entirely.
        return ret;
    }

    // A bin's mass that already overlaps between X and Y needs no transport,
    // so the min-cost flow above never routes it and the arc-flow
    // decomposition below never reports it. Emit it directly as a same-bin
    // plan entry so the plan is a genuine coupling (row sums X, column sums
    // Y), not just the net residual.
    for (int64_t i = 0; i < n_nodes; ++i) {
        const double self_mass = std::min(X[i], Y[i]);
        if (self_mass > 1e-10) {
            if (*n_plan_entries_out >= max_plan_entries) {
                return (int)net.MAX_ITER_REACHED;
            }
            plan_sources_out[*n_plan_entries_out] = static_cast<uint64_t>(i);
            plan_targets_out[*n_plan_entries_out] = static_cast<uint64_t>(i);
            plan_values_out[*n_plan_entries_out] = self_mass;
            ++(*n_plan_entries_out);
        }
    }

    // Decompose the arc flow into a direct (source_bin, target_bin, mass)
    // transportation plan.
    std::vector<std::vector<GridFlowEdge>> flow_adj(n_nodes);
    for (int64_t k = 0; k < total_arcs; ++k) {
        const Digraph::Arc a = Digraph::arcFromId(k);
        const double f = net.flow(a);
        if (f > 1e-10) {
            flow_adj[di.source(a)].push_back({di.target(a), f});
        }
    }

    if (!decompose_grid_flows(flow_adj, supply, plan_sources_out, plan_targets_out,
                              plan_values_out, n_plan_entries_out, max_plan_entries)) {
        return (int)net.MAX_ITER_REACHED;
    }

    return ret;
}

int EMD_wrap_lazy(int n1, int n2, double *X, double *Y, double *coords_a, double *coords_b,
                  int dim, int metric, uint64_t *flow_sources_out,
                  uint64_t *flow_targets_out, double *flow_values_out,
                  uint64_t *n_flows_out, uint64_t max_flows_out,
                  double *alpha, double *beta, double *cost, uint64_t maxIter,
                  double *alpha_init, double *beta_init) {
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
    
    typedef NetworkSimplexSimple<Digraph, double, double, node_id_type> Simplex;
    Simplex::SimplexOptions simplex_options(false);
    // Lazy mode does not store costs or endpoints for the real complete
    // bipartite arcs. Artificial root arcs are still explicit because the
    // simplex initialization assigns them costs 0 or ART_COST.
    simplex_options.cost_storage_mode = Simplex::CostStorageMode::ArtificialArcCosts;
    simplex_options.flow_storage_mode = Simplex::FlowStorageMode::SparseArcFlows;
    simplex_options.endpoint_storage_mode =
        Simplex::EndpointStorageMode::ArcEndpoints;
    simplex_options.state_storage_mode = Simplex::StateStorageMode::PackedArcStates;

    Simplex net(
        di, simplex_options, (int)(n + m), (uint64_t)(n) * (uint64_t)(m), maxIter
    );
    
    // Set supplies
    net.supplyMap(&weights_a_filtered[0], n, &weights_b_filtered[0], m);
    
    // Enable lazy cost computation - costs will be computed on-the-fly
    net.setLazyCost(&coords_a_filtered[0], &coords_b_filtered[0], dim, metric, n, m);
    
    // Initialize warmstart if provided
    if (alpha_init != nullptr && beta_init != nullptr) {
        // Map original indices to graph indices for warmstart
        std::vector<double> alpha_filtered(n);
        std::vector<double> beta_filtered(m);
        for (int i = 0; i < n; i++) {
            int orig_i = idx_a[i];
            alpha_filtered[i] = alpha_init[orig_i];
        }
        for (int j = 0; j < m; j++) {
            int orig_j = idx_b[j];
            beta_filtered[j] = beta_init[orig_j];
        }
        net.setWarmstartPotentials(&alpha_filtered[0], &beta_filtered[0], n, m);
    }
    
    // Run solver
    int ret = net.run();
    
    if (ret == (int)net.OPTIMAL || ret == (int)net.MAX_ITER_REACHED) {
        *cost = 0;
        *n_flows_out = 0;
        
        // Initialize output arrays
        for (int i = 0; i < n1; i++) alpha[i] = 0.0;
        for (int i = 0; i < n2; i++) beta[i] = 0.0;

        auto lazy_cost = [&net](Arc, int i, int j) {
            return net.computeLazyCost(i, j);
        };
        if (!extract_sparse_solution(
                net, di, INVALID, idx_a, idx_b, alpha, beta, cost,
                flow_sources_out, flow_targets_out, flow_values_out,
                n_flows_out, max_flows_out, lazy_cost, 0.0)) {
            return (int)net.MAX_ITER_REACHED;
        }
    }

    return ret;
}
