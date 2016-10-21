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


void EMD_wrap(int n1,int n2, double *X, double *Y,double *D, double *G, double *cost)  {
// beware M and C anre strored in row major C style!!!
  int n, m, i,cur;
  double  max,max_iter;


    typedef FullBipartiteDigraph Digraph;
  DIGRAPH_TYPEDEFS(FullBipartiteDigraph);

  // Get the number of non zero coordinates for r and c
    n=0;
    for (node_id_type i=0; i<n1; i++) {
        double val=*(X+i);
        if (val>0) {
            n++;
        }
    }
    m=0;
    for (node_id_type i=0; i<n2; i++) {
        double val=*(Y+i);
        if (val>0) {
            m++;
        }
    }


    // Define the graph

    std::vector<int> indI(n), indJ(m);
    std::vector<double> weights1(n), weights2(m);
    Digraph di(n, m);
    NetworkSimplexSimple<Digraph,double,double, node_id_type> net(di, true, n+m, n*m,max_iter);

    // Set supply and demand, don't account for 0 values (faster)

    max=0;
    cur=0;
    for (node_id_type i=0; i<n1; i++) {
        double val=*(X+i);
        if (val>0) {
            weights1[ di.nodeFromId(cur) ] = val;
            max+=val;
            indI[cur++]=i;
        }
    }

    // Demand is actually negative supply...

    max=0;
    cur=0;
    for (node_id_type i=0; i<n2; i++) {
        double val=*(Y+i);
        if (val>0) {
            weights2[ di.nodeFromId(cur) ] = -val;
            indJ[cur++]=i;

            max-=val;
        }
    }


    net.supplyMap(&weights1[0], n, &weights2[0], m);

    // Set the cost of each edge
    max=0;
    for (node_id_type i=0; i<n; i++) {
        for (node_id_type j=0; j<m; j++) {
            double val=*(D+indI[i]*n2+indJ[j]);
            net.setCost(di.arcFromId(i*m+j), val);
            if (val>max) {
                max=val;
            }
        }
    }


    // Solve the problem with the network simplex algorithm

    int ret=net.run();
    if (ret!=(int)net.OPTIMAL) {
        if (ret==(int)net.INFEASIBLE) {
            std::cout << "Infeasible problem";
        }
        if (ret==(int)net.UNBOUNDED)
        {
            std::cout << "Unbounded problem";
        }
    } else
    {
        for (node_id_type i=0; i<n; i++)
        {
            for (node_id_type j=0; j<m; j++)
            {
                *(G+indI[i]*n2+indJ[j]) = net.flow(di.arcFromId(i*m+j));
            }
        };
        *cost = net.totalCost();

    };



}
