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


#ifndef EMD_H
#define EMD_H

#include <iostream>
#include <vector>
#include <cstdint>

typedef unsigned int node_id_type;

enum ProblemType {
    INFEASIBLE,
    OPTIMAL,
    UNBOUNDED,
    MAX_ITER_REACHED
};

int EMD_wrap(int n1,int n2, double *X, double *Y,double *D, double *G, double* alpha, double* beta, double *cost, uint64_t maxIter, double* alpha_init, double* beta_init);
int EMD_wrap_omp(int n1,int n2, double *X, double *Y,double *D, double *G, double* alpha, double* beta, double *cost, uint64_t maxIter, int numThreads);

int EMD_wrap_sparse(
    int n1,                      
    int n2,                       
    double *X,                   
    double *Y,                   
    uint64_t n_edges,            // Number of edges in sparse graph
    uint64_t *edge_sources,      // Source indices for each edge (n_edges)
    uint64_t *edge_targets,      // Target indices for each edge (n_edges)
    double *edge_costs,          // Cost for each edge (n_edges)
    uint64_t *flow_sources_out,  // Output: source indices of non-zero flows
    uint64_t *flow_targets_out,  // Output: target indices of non-zero flows
    double *flow_values_out,     // Output: flow values
    uint64_t *n_flows_out,       
    double *alpha,               // Output: dual variables for sources (n1)
    double *beta,                // Output: dual variables for targets (n2)
    double *cost,                // Output: total transportation cost
    uint64_t maxIter             // Maximum iterations for solver
);

int EMD_wrap_lazy(
    int n1,                      // Number of source points
    int n2,                      // Number of target points
    double *X,                   // Source weights (n1)
    double *Y,                   // Target weights (n2)
    double *coords_a,            // Source coordinates (n1 x dim)
    double *coords_b,            // Target coordinates (n2 x dim)
    int dim,                     // Dimension of coordinates
    int metric,                  // Distance metric: 0=sqeuclidean, 1=euclidean, 2=cityblock
    double *G,                   // Output: transport plan (n1 x n2)
    double *alpha,               // Output: dual variables for sources (n1)
    double *beta,                // Output: dual variables for targets (n2)
    double *cost,                // Output: total transportation cost
    uint64_t maxIter             // Maximum iterations for solver
);


#endif
