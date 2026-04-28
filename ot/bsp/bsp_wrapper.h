#ifndef BSP_WRAPPER_H
#define BSP_WRAPPER_H

#include <cstdint>
#include <string>



double BSPOT_wrap(int n, int d, double *X, double *Y, uint64_t nb_plans, int *plans, int *plan,int lp_power,int* initial_plan,bool gaussian);
double MergeBijections(int n, int d, double *X, double *Y, uint64_t nb_plans, int *plans, int *plan,int lp_power);

#endif
