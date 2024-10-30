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

#include "EMD.h"

int EMD_wrap_all_sparse(int n1, int n2, double *X, double *Y,
                        long *iD, long *jD, double *D, long nD,
                        long *iG, long *jG, double *G, long *nG,
                        double *alpha, double *beta, double *cost, int maxIter)
{
    // beware M and C anre strored in row major C style!!!

    // Get the number of non zero coordinates for r and c and vectors
    int n, m, cur;

    typedef FullBipartiteDigraph Digraph;
    DIGRAPH_TYPEDEFS(FullBipartiteDigraph);

    n = n1;
    m = n2;

    // Define the graph

    std::vector<double> weights2(m);
    Digraph di(n, m);
    NetworkSimplexSimple<Digraph, double, double, node_id_type> net(di, true, n + m, n * m, maxIter);

    // Set supply and demand, don't account for 0 values (faster)

    // Demand is actually negative supply...

    cur = 0;
    for (int i = 0; i < n2; i++)
    {
        double val = *(Y + i);
        if (val > 0)
        {
            weights2[cur] = -val;
        }
    }

    // Define the graph
    net.supplyMap(X, n, &weights2[0], m);

    // Set the cost of each edge
    for (int k = 0; k < nD; k++)
    {
        int i = iD[k];
        int j = jD[k];
        net.setCost(di.arcFromId(i * m + j), D[k]);
    }

    // Solve the problem with the network simplex algorithm

    int ret = net.run();
    if (ret == (int)net.OPTIMAL || ret == (int)net.MAX_ITER_REACHED)
    {
        *cost = net.totalCost();
        Arc a;
        di.first(a);
        cur = 0;
        for (; a != INVALID; di.next(a))
        {
            int i = di.source(a);
            int j = di.target(a);
            double flow = net.flow(a);
            if (flow > 0)
            {

                *(G + cur) = flow;
                *(iG + cur) = i;
                *(jG + cur) = j - n;
                *(alpha + i) = -net.potential(i);
                *(beta + j - n) = net.potential(j);
                cur++;
            }
        }
        *nG = cur; // nb of value +1 for numpy indexing
    }

    return ret;
}