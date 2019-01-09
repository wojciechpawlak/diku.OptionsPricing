#ifndef CUDA_INTEROP_H
#define CUDA_INTEROP_H

#include "Real.hpp"

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#define DEVICE __device__
#define HOST __host__
#else
#define CONSTANT const
#define DEVICE
#define HOST
#endif

namespace trinom
{
    DEVICE inline int MAX(int x, int y) { return (((x) > (y)) ? (x) : (y)); }
    DEVICE inline real MAX(real x, real y) { return (((x) > (y)) ? (x) : (y)); }
    DEVICE inline int MIN(int x, int y) { return (((x) < (y)) ? (x) : (y)); }
    DEVICE inline real MIN(real x, real y) { return (((x) < (y)) ? (x) : (y)); }
}

#endif
