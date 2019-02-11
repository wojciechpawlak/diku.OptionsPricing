#ifndef SEQ_HPP
#define SEQ_HPP

#include "../common/ValuationConstants.hpp"
#include "../common/Domain.hpp"

using namespace trinom;

namespace seq
{

struct jvalue
{
    real rate;
    real pu;
    real pm;
    real pd;
};

/**
 *  Sequential version that computes the bond tree until bond maturity
 *  and prices the option on maturity during backward propagation.
 **/
real computeSingleOption(const ValuationConstants &c, const Valuations &valuations, const int idx)
{
    // Precompute probabilities and rates for all js.
    auto jvalues = new jvalue[c.width];
    auto jmin = -c.jmax;

    jvalue &valmin = jvalues[0];
    valmin.rate = jmin * c.dr;
    valmin.pu = PU_B(jmin, c.M);
    valmin.pm = PM_B(jmin, c.M);
    valmin.pd = PD_B(jmin, c.M);

    jvalue &valmax = jvalues[c.width - 1];
    valmax.rate = c.jmax * c.dr;
    valmax.pu = PU_C(c.jmax, c.M);
    valmax.pm = PM_C(c.jmax, c.M);
    valmax.pd = PD_C(c.jmax, c.M);

    for (auto i = 1; i < c.width - 1; ++i)
    {
        jvalue &val = jvalues[i];
        auto j = i + jmin;
        val.rate = j * c.dr;
        val.pu = PU_A(j, c.M);
        val.pm = PM_A(j, c.M);
        val.pd = PD_A(j, c.M);
    }

    // Forward induction to calculate Qs and alphas
    auto Qs = new real[c.width]();     // Qs[j]: j in jmin..jmax
    auto QsCopy = new real[c.width](); // QsCopy[j]
    Qs[c.jmax] = one;                  // Qs[0] = 1$

    auto alphas = new real[c.n + 1](); // alphas[i]
    volatile uint16_t lastUsedYCTermIdx = 0;
    alphas[0] = interpolateRateAtTimeStep(c.dt, c.termUnit, c.firstYieldCurveRate, c.firstYieldCurveTimeStep, c.yieldCurveTermCount, &lastUsedYCTermIdx); // initial dt-period interest rate
#ifdef DEV
    printf("0 %d alpha %f \n", 0, alphas[0]);
#endif

    for (auto i = 0; i < c.n; ++i)
    {
        auto jhigh = MIN(i, c.jmax);
        auto alpha = alphas[i];
#ifdef DEV
        printf("1 %d alpha %f \n", i, alpha);
#endif

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - jmin;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            auto qexp = Qs[jind] * exp(-(alpha + jval.rate) * c.dt);

            if (j == jmin)
            {
                // Bottom edge branching
                QsCopy[jind + 2] += jval.pu * qexp; // up two
                QsCopy[jind + 1] += jval.pm * qexp; // up one
                QsCopy[jind] += jval.pd * qexp;     // middle
            }
            else if (j == c.jmax)
            {
                // Top edge branching
                QsCopy[jind] += jval.pu * qexp;     // middle
                QsCopy[jind - 1] += jval.pm * qexp; // down one
                QsCopy[jind - 2] += jval.pd * qexp; // down two
            }
            else
            {
                // Standard branching
                QsCopy[jind + 1] += jval.pu * qexp; // up
                QsCopy[jind] += jval.pm * qexp;     // middle
                QsCopy[jind - 1] += jval.pd * qexp; // down
            }
        }

        // Determine the new alpha using equation 30.22
        // by summing up Qs from the next time step
        auto jhigh1 = MIN(i + 1, c.jmax);
        auto aggregatedQs = zero;
        for (auto j = -jhigh1; j <= jhigh1; ++j)
        {
            auto jind = j - jmin;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            aggregatedQs += QsCopy[jind] * exp(-jval.rate * c.dt);
        }

        alphas[i + 1] = computeAlpha(aggregatedQs, i, c.dt, c.termUnit, c.firstYieldCurveRate, c.firstYieldCurveTimeStep, c.yieldCurveTermCount, &lastUsedYCTermIdx);
#ifdef DEV
        printf("2 %d alpha %f \n", i+1, alphas[i + 1]);
#endif
        // Switch Qs
        auto QsT = Qs;
        Qs = QsCopy;
        QsCopy = QsT;
        std::fill_n(QsCopy, c.width, 0);
    }

    // Backward propagation
    auto price = Qs; // call[j]
    auto priceCopy = QsCopy;

    auto lastUsedCIdx = valuations.CashflowIndices[idx] + valuations.Cashflows[idx] - 1;
#ifdef DEV
    printf("%d: %d %f %f %d\n", idx, lastUsedCIdx, valuations.Repayments[lastUsedCIdx], valuations.Coupons[lastUsedCIdx], valuations.CashflowSteps[lastUsedCIdx]);
#endif
    std::fill_n(price, c.width, valuations.Repayments[lastUsedCIdx] + valuations.Coupons[lastUsedCIdx]); // initialize to par/face value: last repayment + last coupon
    auto lastUsedCStep = valuations.CashflowSteps[--lastUsedCIdx];

    for (auto i = c.n - 1; i >= 0; --i)
    {
        auto jhigh = MIN(i, c.jmax);
        auto alpha = alphas[i];
#ifdef DEV
        printf("3 %d alpha %f \n", i, alpha);
#endif
        const auto isExerciseStep = i <= c.LastExerciseStep && i >= c.FirstExerciseStep && (lastUsedCStep - i) % c.ExerciseStepFrequency == 0;

        // add coupon and repayments
        if (i == lastUsedCStep - 1)
        {
            for (auto j = -jhigh; j <= jhigh; ++j)
            {
                const auto jind = j + c.jmax;      // array index for j
                price[jind] += valuations.Repayments[lastUsedCIdx] + valuations.Coupons[lastUsedCIdx];
            }
            lastUsedCStep = (--lastUsedCIdx >= 0) ? valuations.CashflowSteps[lastUsedCIdx] : 0;
        }

        // calculate accrued interest from cashflow
        const auto ai = isExerciseStep && lastUsedCStep != 0 ? computeAccruedInterest(c.termStepCount, i, lastUsedCStep, valuations.CashflowSteps[lastUsedCIdx + 1], valuations.Coupons[lastUsedCIdx]) : zero;

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j - jmin;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            auto discFactor = exp(-(alpha + jval.rate) * c.dt) * c.expmOasdt;

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (jval.pu * price[jind] +
                       jval.pm * price[jind - 1] +
                       jval.pd * price[jind - 2]) *
                      discFactor;
            }
            else if (j == jmin)
            {
                // Bottom edge branching
                res = (jval.pu * price[jind + 2] +
                       jval.pm * price[jind + 1] +
                       jval.pd * price[jind]) *
                      discFactor;
            }
            else
            {
                // Standard branching
                res = (jval.pu * price[jind + 1] +
                       jval.pm * price[jind] +
                       jval.pd * price[jind - 1]) *
                      discFactor;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            priceCopy[jind] = getOptionPayoff(isExerciseStep, c.X, c.type, res, ai);
        }

        // Switch price arrays
        auto priceT = price;
        price = priceCopy;
        priceCopy = priceT;

        std::fill_n(priceCopy, c.width, 0);
    }

    auto result = price[c.jmax];
#ifdef DEV
    printf("res: %f\n", result);
#endif
    delete[] jvalues;
    delete[] alphas;
    delete[] Qs;
    delete[] QsCopy;

    return result;
}

void computeOptions(const Valuations &valuations, std::vector<real> &results)
{
#pragma omp parallel for
    for (auto i = 0; i < valuations.ValuationCount; ++i)
    {
        ValuationConstants c(valuations, i);
        auto result = computeSingleOption(c, valuations, i);
        results[i] = result;
    }
}

} // namespace seq

#endif
