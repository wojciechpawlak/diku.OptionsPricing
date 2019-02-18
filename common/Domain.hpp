#ifndef DOMAIN_HPP
#define DOMAIN_HPP

#include "CudaInterop.h"

//#define DEV_LIMIT
//#define DEV
//#define DEV1
//#define DEV2
//#define DEV_EXTRA
#define PRINT_IDX 0
#define PRINT_FIRST_ITER -1
#define PRINT_LAST_ITER -1

namespace trinom
{

DEVICE inline real interpolateRateAtTimeStep(const real t, const int termUnit, const real *rates, const uint16_t *termSteps, const uint16_t size, volatile uint16_t *lastUsedTermIdx)
{
    const int tDays = (int)ROUND(t * termUnit);
    auto first = 0;
    auto second = 0;

    // extrapolate
    if (tDays <= termSteps[0])
    {
        return rates[0];
    }

    // extrapolate
    if (tDays > termSteps[size - 1])
    {
        return rates[size - 1];
    }

    // interpolate
    for (auto i = *lastUsedTermIdx; i < size; ++i)
    {
        if (termSteps[i] >= tDays)
        {
            second = i;
            *lastUsedTermIdx = i;
            first = i - 1;
            break;
        }
    }

    auto t1 = termSteps[first];
    auto t2 = termSteps[second];
    auto p1 = rates[first];
    auto p2 = rates[second];
    auto coefficient = (tDays - t1) / (real)(t2 - t1);
    return p1 + coefficient * (p2 - p1);
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

DEVICE inline real computeAlpha(const real aggregatedQs, const int i, const real dt, const int termUnit, const real *prices, const uint16_t *timeSteps, const int size, volatile uint16_t *lastIdx)
{
    auto ti = (i + 2) * dt;
    auto R = interpolateRateAtTimeStep(ti, termUnit, prices, timeSteps, size, lastIdx); // discount rate
    auto P = exp(-R * ti);                                          // discount bond price
    return log(aggregatedQs / P) / dt;                              // new alpha
}

DEVICE inline real computeJValue(const int j, const int jmax, const real M, const int expout)
{
    if (j == -jmax)
    {
        switch (expout)
        {
        case 1:
            return PU_B(j, M); // up
        case 2:
            return PM_B(j, M); // mid
        case 3:
            return PD_B(j, M); // down
        }
    }
    else if (j == jmax)
    {
        switch (expout)
        {
        case 1:
            return PU_C(j, M); // up
        case 2:
            return PM_C(j, M); // mid
        case 3:
            return PD_C(j, M); // down
        }
    }
    else
    {
        switch (expout)
        {
        case 1:
            return PU_A(j, M); // up
        case 2:
            return PM_A(j, M); // mid
        case 3:
            return PD_A(j, M); // down
        }
    }
    return 0;
}

DEVICE inline real computeAccruedInterest(const int i, const int prevCouponIdx, const int nextCouponIdx, const real nextCoupon)
{
    real couponsTimeDiff = nextCouponIdx - prevCouponIdx;
    real eventsTimeDiff = nextCouponIdx - i;
    eventsTimeDiff = eventsTimeDiff <= 0.0 ? 0.0 : eventsTimeDiff;
    return (couponsTimeDiff - (real)eventsTimeDiff) / couponsTimeDiff * nextCoupon;
}

DEVICE inline real getOptionPayoff(bool isExerciseStep, const real strike, const OptionTypeE type, const real bondPrice, const real ai)
{
    real ret = bondPrice;
    if (isExerciseStep)
    {
        const real exercisePrice = strike + ai;
        switch (type)
        {
        case OptionTypeE::CALL_VANILLA:
            //ret = MAX(bondPrice - X, zero); // Call Option
            ret = bondPrice > exercisePrice ? exercisePrice : bondPrice; // Call on a bond (embedded)
            break;
        case OptionTypeE::PUT_VANILLA:
            //ret = MAX(X - bondPrice, zero);
            ret = exercisePrice > bondPrice ? exercisePrice : bondPrice; // Put on a bond (embedded)
            break;
        }
    }

    if (ret == infinity)
        return REAL_MAX;
    else if (ret == -infinity)
        return REAL_MIN;
    else
        return ret;
}

} // namespace trinom

#endif
