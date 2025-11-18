#include "BSP-OT_header_only.h"


double BSPOT_wrap(int n1, int n2, int d, double *X, double *Y, uint64_t nb_plans, int *plans, int *plan) {
    BSPOT::Points<d> A(d,n1);
    BSPOT::Points<d> B(d,n2);

    double cost = 0.0;

    for (int i=0;i<n1;i++) {
        A(0,i) = X[i];
    }
    for (int j=0;j<n2;j++) {
        B(0,j) = Y[j];
    }

    BSPOT::cost_function cost = [&] (int i,int j) {
	return (A.col(i) - B.col(j)).squaredNorm();
	};

    // Compute BSPOT matching

    // stotr the plans and individual plan

    // compute cost
    

    return cost;
}

