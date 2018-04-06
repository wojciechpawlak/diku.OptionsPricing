#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include "Real.hpp"
#include "OptionConstants.hpp"
#include <cmath>
#include <algorithm>

using namespace std;

namespace trinom
{

struct Yield
{
    real p;
    int t;
};

#ifdef __CUDA_ARCH__
#define CONSTANT __constant__
#else
#define CONSTANT const
#endif

#ifdef __CUDA_ARCH__
#define DEVICE __device__
#else
#define DEVICE
#endif

// The DM zero coupon yield curve, July 8, 1994.
CONSTANT Yield h_YieldCurve[] =
    {{.p = 0.0501772, .t = 3},    //
     {.p = 0.0498284, .t = 31},   //
     {.p = 0.0497234, .t = 62},   //
     {.p = 0.0496157, .t = 94},   //
     {.p = 0.0499058, .t = 185},  //
     {.p = 0.0509389, .t = 367},  //
     {.p = 0.0579733, .t = 731},  //
     {.p = 0.0630595, .t = 1096}, //
     {.p = 0.0673464, .t = 1461}, //
     {.p = 0.0694816, .t = 1826}, //
     {.p = 0.0708807, .t = 2194}, //
     {.p = 0.0727527, .t = 2558}, //
     {.p = 0.0730852, .t = 2922}, //
     {.p = 0.0739790, .t = 3287}, //
     {.p = 0.0749015, .t = 3653}};

// Yield h_YieldCurve[] =
//     {{.p = 0.03430, .t = (int)(0.5 * year)}, //
//      {.p = 0.03824, .t = (int)(1.0 * year)}, //
//      {.p = 0.04183, .t = (int)(1.5 * year)}, //
//      {.p = 0.04512, .t = (int)(2.0 * year)}, //
//      {.p = 0.04812, .t = (int)(2.5 * year)}, //
//      {.p = 0.05086, .t = (int)(3.0 * year)}};
DEVICE real getYieldAtYear(real t)
{
    t *= year;
    auto first = h_YieldCurve[0];
    auto second = h_YieldCurve[0];

    for (auto yield : h_YieldCurve)
    {
        if (yield.t > t)
        {
            second = yield;
            break;
        }
        first = yield;
    }

    // Prevent division by zero
    if (first.t == second.t)
    {
        return first.p;
    }

    auto coefficient = (t - first.t) / (second.t - first.t);
    return first.p + coefficient * (second.p - first.p);
}

// Probability Equations

// Exhibit 1A (-jmax < j < jmax) (eq. 3A in Hull-White 1996)
DEVICE inline real PU_A(int j, real M)
{
    return one / six + (j * j * M * M + j * M) * half;
}

DEVICE inline real PM_A(int j, real M)
{
    return two / three - j * j * M * M;
}

DEVICE inline real PD_A(int j, real M)
{
    return one / six + (j * j * M * M - j * M) * half;
}

// Exhibit 1B (j == -jmax) (eq. 3C in Hull-White 1996)
DEVICE inline real PU_B(int j, real M)
{
    return one / six + (j * j * M * M - j * M) * half;
}

DEVICE inline real PM_B(int j, real M)
{
    return -one / three - j * j * M * M + two * j * M;
}

DEVICE inline real PD_B(int j, real M)
{
    return seven / six + (j * j * M * M - three * j * M) * half;
}

// Exhibit 1C (j == jmax) (eq. 3B in Hull-White 1996)
DEVICE inline real PU_C(int j, real M)
{
    return seven / six + (j * j * M * M + three * j * M) * half;
}

DEVICE inline real PM_C(int j, real M)
{
    return -one / three - j * j * M * M - two * j * M;
}

DEVICE inline real PD_C(int j, real M)
{
    return one / six + (j * j * M * M + j * M) * half;
}

DEVICE inline real computeAlpha(const real aggregatedQs, const int i, const real dt)
{
    auto ti = (i + 2) * dt;       // next next time step
    auto R = getYieldAtYear(ti);  // discount rate
    auto P = exp(-R * ti);        // discount bond price
    return log(aggregatedQs / P); // new alpha
}

DEVICE real computeJValue(const int i, const real dr, const real M, const int width, const int jmax, const int expout)
{
    // this i is only local and has nothing to do with the height of the tree.
    if (i == 0)
    {
        switch (expout)
        {
        case 1:
            return PU_B((-jmax), M); // up
        case 2:
            return PM_B((-jmax), M); // mid
        case 3:
            return PD_B((-jmax), M); // down
        case 0:
        default:
            return (-jmax) * dr; // rate
        }
    }
    else if (i == width - 1)
    {
        switch (expout)
        {
        case 1:
            return PU_C(jmax, M); // up
        case 2:
            return PM_C(jmax, M); // mid
        case 3:
            return PD_C(jmax, M); // down
        case 0:
        default:
            return jmax * dr; // rate
        }
    }
    else
    {
        auto j = i + (-jmax);
        switch (expout)
        {
        case 1:
            return PU_A(j, M); // up
        case 2:
            return PM_A(j, M); // mid
        case 3:
            return PD_A(j, M); // down
        case 0:
        default:
            return j * dr; // rate
        }
    }
}

DEVICE inline real computeCallValue(bool isMaturity, const OptionConstants &c, const real res)
{
    if (isMaturity)
    {
        switch (c.type)
        {
        case OptionType::PUT:
            return max(c.X - res, zero);
        case OptionType::CALL:
            return max(res - c.X, zero);
        }
    }
    return isMaturity ? max(c.X - res, zero) : res;
}

// forward propagation helper
DEVICE real fwdHelper(real M, real dr, real dt, real alphai, volatile real *QCopy, int beg_ind, int m, int i, int imax, int jmax, int j)
{
    auto eRdt_u1 = exp(-(((real)(j + 1)) * dr + alphai) * dt);
    auto eRdt = exp(-(((real)j) * dr + alphai) * dt);
    auto eRdt_d1 = exp(-(((real)(j - 1)) * dr + alphai) * dt);
    if (i < jmax)
    {
        auto pu = PU_A(j - 1, M);
        auto pm = PM_A(j, M);
        auto pd = PD_A(j + 1, M);
        if (i == 0 && j == 0)
            return pm * QCopy[beg_ind + j + m] * eRdt;
        else if (j == -imax + 1)
            return pd * QCopy[beg_ind + j + m + 1] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt;
        else if (j == imax - 1)
            return pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j + m - 1] * eRdt_d1;
        else if (j == 0 - imax)
            return pd * QCopy[beg_ind + j + m + 1] * eRdt_u1;
        else if (j == imax)
            return pu * QCopy[beg_ind + j + m - 1] * eRdt_d1;
        else
            return pd * QCopy[beg_ind + j + m + 1] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j + m - 1] * eRdt_d1;
    }
    else if (j == jmax)
    {
        auto pm = PU_C(j, M);
        auto pu = PU_A(j - 1, M);
        return pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j - 1 + m] * eRdt_d1;
    }
    else if (j == jmax - 1)
    {
        auto pd = PM_C(j + 1, M);
        auto pm = PM_A(j, M);
        auto pu = PU_A(j - 1, M);
        return pd * QCopy[beg_ind + j + 1 + m] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j - 1 + m] * eRdt_d1;
    }
    else if (j == jmax - 2)
    {
        auto eRdt_u2 = exp(-(((real)(j + 2)) * dr + alphai) * dt);
        auto pd_c = PD_C(j + 2, M);
        auto pd = PD_A(j + 1, M);
        auto pm = PM_A(j, M);
        auto pu = PU_A(j - 1, M);
        return pd_c * QCopy[beg_ind + j + 2 + m] * eRdt_u2 + pd * QCopy[beg_ind + j + 1 + m] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j - 1 + m] * eRdt_d1;
    }
    else if (j == -jmax + 2)
    {
        auto eRdt_d2 = exp(-(((real)(j - 2)) * dr + alphai) * dt);
        auto pd = PD_A(j + 1, M);
        auto pm = PM_A(j, M);
        auto pu = PU_A(j - 1, M);
        auto pu_b = PU_B(j - 2, M);
        return pd * QCopy[beg_ind + j + 1 + m] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j - 1 + m] * eRdt_d1 + pu_b * QCopy[beg_ind + j - 2 + m] * eRdt_d2;
    }
    else if (j == -jmax + 1)
    {
        auto pd = PD_A(j + 1, M);
        auto pm = PM_A(j, M);
        auto pu = PM_B(j - 1, M);
        return pd * QCopy[beg_ind + j + 1 + m] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j - 1 + m] * eRdt_d1;
    }
    else if (j == -jmax)
    {
        auto pd = PD_A(j + 1, M);
        auto pm = PD_B(j, M);
        return pd * QCopy[beg_ind + j + 1 + m] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt;
    }
    else
    {
        auto pd = PD_A(j + 1, M);
        auto pm = PM_A(j, M);
        auto pu = PU_A(j - 1, M);
        return pd * QCopy[beg_ind + j + 1 + m] * eRdt_u1 + pm * QCopy[beg_ind + j + m] * eRdt + pu * QCopy[beg_ind + j - 1 + m] * eRdt_d1;
    }
}

// backward propagation helper
DEVICE real bkwdHelper(real X, real M, real dr, real dt, real alphai, volatile real *CallCopy, int beg_ind, int m, int i, int jmax, int j)
{
    auto eRdt = exp(-(((real)j) * dr + alphai) * dt);
    real res;
    if (i < jmax)
    {
        // -- central node
        auto pu = PU_A(j, M);
        auto pm = PM_A(j, M);
        auto pd = PD_A(j, M);
        res = (pu * CallCopy[beg_ind + j + m + 1] + pm * CallCopy[beg_ind + j + m] + pd * CallCopy[beg_ind + j + m - 1]) * eRdt;
    }
    else if (j == jmax)
    {
        // top node
        auto pu = PU_C(j, M);
        auto pm = PM_C(j, M);
        auto pd = PD_C(j, M);
        res = (pu * CallCopy[beg_ind + j + m] + pm * CallCopy[beg_ind + j + m - 1] + pd * CallCopy[beg_ind + j + m - 2]) * eRdt;
    }
    else if (j == -jmax)
    {
        // bottom node
        auto pu = PU_B(j, M);
        auto pm = PM_B(j, M);
        auto pd = PD_B(j, M);
        res = (pu * CallCopy[beg_ind + j + m + 2] + pm * CallCopy[beg_ind + j + m + 1] + pd * CallCopy[beg_ind + j + m]) * eRdt;
    }
    else
    {
        // central node
        auto pu = PU_A(j, M);
        auto pm = PM_A(j, M);
        auto pd = PD_A(j, M);
        res = (pu * CallCopy[beg_ind + j + m + 1] + pm * CallCopy[beg_ind + j + m] + pd * CallCopy[beg_ind + j + m - 1]) * eRdt;
    }

    // TODO (WMP) This should be parametrized; length of contract, here 3 years
    if (i == ((int)(three / dt)))
        return max(X - res, zero);
    else
        return res;
}
}

#endif
