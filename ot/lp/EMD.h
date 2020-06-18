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

#include <vector>
#include "network_simplex_simple.h"

using namespace lemon;

typedef int64_t arc_id_type; // handle (n1*n2+n1+n2) nodes (I64_MAX=3037000500^2)
typedef double supply_type; // handle sum of supplies and demand (should be signed)
typedef double cost_type; // handle number of arcs * maximum cost (should be signed)

enum ProblemType {
    INFEASIBLE,
    OPTIMAL,
    UNBOUNDED,
};

int EMD_wrap(int n1,int n2, double *X, double *Y,double *D, double *G,
             double* alpha, double* beta, double *cost, int maxIter);

#endif // EMD_H
