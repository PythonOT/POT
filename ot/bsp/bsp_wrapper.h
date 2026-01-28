#ifndef BSP_WRAPPER_H
#define BSP_WRAPPER_H

#include <cstdint>
#include <string>



double BSPOT_wrap(int n, int d, double *X, double *Y, uint64_t nb_plans, int *plans, int *plan,std::string cost = "sqnorm");

#endif
