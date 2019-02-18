#ifndef SEQ_HPP
#define SEQ_HPP

#include "../common/ValuationConstants.hpp"
#include "../common/Domain.hpp"

#define FORWARD_GATHER

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
    auto Qs = new real[c.width]();     // Qs[j]: j in -jmax..jmax
    auto QsCopy = new real[c.width](); // QsCopy[j]
    auto alphas = new real[c.n + 1](); // alphas[i]
    auto alpha = 0.0;
    volatile uint16_t lastUsedYCTermIdx = 0;
#ifdef DEV_EXTRA
    auto a = valuations.MeanReversionRates.at(idx);
    auto sigma = valuations.Volatilities.at(idx);
    const auto tmp = -two * a * c.dt;
    if (idx == PRINT_IDX) printf("%d: %d %.18f %.18f %.18f %.18f %.18f %.18f %d %d %d %d\n",
        idx, c.n, a, sigma, tmp, exp(tmp), sigma * sigma * (one - exp(-two * a * c.dt)) / (two * a), c.dt, c.firstYCTermIdx, c.lastExerciseStep, c.firstExerciseStep, c.exerciseStepFrequency);
    if (idx == PRINT_IDX) printf("%d: %.18f %d %d\n", idx, valuations.YieldCurveRates[c.firstYCTermIdx], valuations.YieldCurveTimeSteps[c.firstYCTermIdx], valuations.YieldCurveTerms[valuations.YieldCurveIndices[idx]]);
#endif

    // Precompute exponent of rates for each node on the width on the tree (constant over forward propagation)
    auto jvalues = new jvalue[c.width];

    jvalue &valmin = jvalues[0];
    valmin.rate = exp(c.mdrdt*-c.jmax);
    valmin.pu = PU_B(-c.jmax, c.M);
    valmin.pm = PM_B(-c.jmax, c.M);
    valmin.pd = PD_B(-c.jmax, c.M);
#ifdef DEV
    if (idx == PRINT_IDX) printf("%d: %d: %.18f %.18f %.18f %.18f\n", idx, 0, valmin.rate, valmin.pu, valmin.pm, valmin.pd);
#endif

    for (auto j = -c.jmax + 1; j < c.jmax; ++j)
    {
        auto jind = j + c.jmax;
        jvalue &val = jvalues[jind];
        val.rate = exp(c.mdrdt*j);
        val.pu = PU_A(j, c.M);
        val.pm = PM_A(j, c.M);
        val.pd = PD_A(j, c.M);
#ifdef DEV
        if (idx == PRINT_IDX) printf("%d: %d: %.18f %.18f %.18f %.18f\n", idx, jind, val.rate, val.pu, val.pm, val.pd);
#endif
    }
    jvalue &valmax = jvalues[c.width - 1];
    valmax.rate = exp(c.mdrdt*c.jmax);
    valmax.pu = PU_C(c.jmax, c.M);
    valmax.pm = PM_C(c.jmax, c.M);
    valmax.pd = PD_C(c.jmax, c.M);
#ifdef DEV
    if (idx == PRINT_IDX) printf("%d: %d: %.18f %.18f %.18f %.18f\n", idx, c.width - 1, valmax.rate, valmax.pu, valmax.pm, valmax.pd);
#endif

    // Forward propagation
    Qs[c.jmax] = one;                  // Qs[0] = 1$
    alpha = interpolateRateAtTimeStep(c.dt, c.termUnit, c.firstYieldCurveRate, c.firstYieldCurveTimeStep, c.yieldCurveTermCount, &lastUsedYCTermIdx); // initial dt-period interest rate
    alphas[0] = exp(-alpha * c.dt);

#ifndef FORWARD_GATHER
    // Scatter
    for (auto i = 0; i < c.n; ++i)
    {
        auto jhigh = MIN(i, c.jmax);
        const auto expmAlphadt = alphas[i];
#ifdef DEV1
        if (idx == PRINT_IDX) printf("%d %d: %.18f %.18f %d\n", idx, i, alpha, expmAlphadt, lastUsedYCTermIdx);
#endif

        // Forward iteration step, compute Qs in the next time step
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            auto qexp = Qs[jind] * expmAlphadt * jval.rate;

            if (j == -c.jmax)
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
            auto jind = j + c.jmax;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            aggregatedQs += QsCopy[jind] * jval.rate;
        }

        alpha = computeAlpha(aggregatedQs, i, c.dt, c.termUnit, c.firstYieldCurveRate, c.firstYieldCurveTimeStep, c.yieldCurveTermCount, &lastUsedYCTermIdx);
        alphas[i + 1] = exp(-alpha * c.dt);

        // Switch Qs
        auto QsT = Qs;
        Qs = QsCopy;
        QsCopy = QsT;
        std::fill_n(QsCopy, c.width, 0);

#ifdef DEV1
        if (idx == PRINT_IDX && i > PRINT_FIRST_ITER && i < PRINT_LAST_ITER)
        {
            printf("%d %d: ", idx, i);
            for (auto k = 0; k < c.width; ++k)
            {
                printf("%d: %.18f ", k, Qs[k]);
            }
            printf("\n");
        }
#endif
    }
#else
#ifdef DEV1
    if (idx == PRINT_IDX)
        printf("%d %d: %.18f %.18f %.18f %d\n", idx, 0, 1.0, alpha, alphas[0], lastUsedYCTermIdx);
#endif
#ifdef DEV1
    if (idx == PRINT_IDX && 0 >= PRINT_FIRST_ITER)
    {
        printf("%d %d: ", idx, 0);
        for (auto k = 0; k < c.width; ++k)
        {
            printf("%d: %.18f ", k, Qs[k]);
        }
        printf("\n");
    }
#endif
    // Gather
    for (auto i = 1; i <= c.n; ++i)
    {
        const auto jhigh = MIN(i, c.jmax);

        // Precompute Qexp
        const auto expmAlphadt = alphas[i - 1];
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            const auto jind = j + c.jmax;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            real Qexp = Qs[jind] * expmAlphadt * jval.rate;
            Qs[jind] = Qexp;
        }
        // Forward iteration step, compute Qs in the next time step
        real aggregatedQs = zero;
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            const auto jind = j + c.jmax;      // array index for j            
            auto jvalp1 = jvalues[jind + 1]; // precomputed probabilities and rates
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            auto jvalm1 = jvalues[jind - 1]; // precomputed probabilities and rates
            const auto expu = j == jhigh ? zero : Qs[jind + 1];
            const auto expm = Qs[jind];
            const auto expd = j == -jhigh ? zero : Qs[jind - 1];

            real Q;
            if (i == 1)
            {
                if (j == -jhigh) {
                    Q = jvalp1.pd * expu;
                }
                else if (j == jhigh) {
                    Q = jvalm1.pu * expd;
                }
                else {
                    Q = jval.pm * expm;
                }
            }
            else if (i <= c.jmax)
            {
                if (j == -jhigh) {
                    Q = jvalp1.pd * expu;
                }
                else if (j == -jhigh + 1) {
                    Q = jval.pm * expm +
                        jvalp1.pd * expu;
                }
                else if (j == jhigh) {
                    Q = jvalm1.pu * expd;
                }
                else if (j == jhigh - 1) {
                    Q = jvalm1.pu * expd +
                        jval.pm * expm;
                }
                else {
                    Q = jvalm1.pu * expd +
                        jval.pm * expm +
                        jvalp1.pd * expu;
                }
            }
            else
            {
                if (j == -jhigh) {
                    Q = jval.pd * expm +
                        jvalp1.pd * expu;
                }
                else if (j == -jhigh + 1) {
                    Q = jvalm1.pm * expd +
                        jval.pm * expm +
                        jvalp1.pd * expu;

                }
                else if (j == jhigh) {
                    Q = jvalm1.pu * expd +
                        jval.pu * expm;
                }
                else if (j == jhigh - 1) {
                    Q = jvalm1.pu * expd +
                        jval.pm * expm +
                        jvalp1.pm * expu;
                }
                else {
                    Q = ((j == -jhigh + 2) ? jvalues[jind - 2].pu * Qs[jind - 2] : zero) +
                        jvalm1.pu * expd +
                        jval.pm * expm +
                        jvalp1.pd * expu +
                        ((j == jhigh - 2) ? jvalues[jind + 2].pd * Qs[jind + 2] : zero);
                }
            }
            // Determine the new alpha using equation 30.22
            // by summing up Qs from the next time step
            QsCopy[jind] = Q;
            aggregatedQs += Q * jval.rate;
#ifdef DEV1
            if (idx == PRINT_IDX && i == 1) printf("%d %d: %.18f %.18f %.18f\n", idx, jind, aggregatedQs, Q, jval.rate);
#endif
        }

        alpha = computeAlpha(aggregatedQs, i - 1, c.dt, c.termUnit, c.firstYieldCurveRate, c.firstYieldCurveTimeStep, c.yieldCurveTermCount, &lastUsedYCTermIdx);
        alphas[i] = exp(-alpha * c.dt);
#ifdef DEV1
        if (idx == PRINT_IDX) printf("%d %d: %.18f %.18f %.18f %d\n", idx, i, aggregatedQs, alpha, alphas[i], lastUsedYCTermIdx);
#endif

        // Switch Qs
        auto QsT = Qs;
        Qs = QsCopy;
        QsCopy = QsT;
        std::fill_n(QsCopy, c.width, 0);

#ifdef DEV1
        if (idx == PRINT_IDX && i > PRINT_FIRST_ITER && i < PRINT_LAST_ITER)
        {
            printf("%d %d: ", idx, i);
            for (auto k = 0; k < c.width; ++k)
            {
                printf("%d: %.18f ", k, Qs[k]);
            }
            printf("\n");
        }
#endif
    }
#endif

    // Backward propagation
    auto price = Qs;
    auto priceCopy = QsCopy;

    auto lastUsedCIdx = valuations.CashflowIndices[idx] + valuations.Cashflows[idx] - 1;
    auto cashflowsRemaining = valuations.Cashflows[idx];
#ifdef DEV2
    if (idx == PRINT_IDX) printf("%d %d: %d %d %f %f %d\n", idx, c.n, lastUsedCIdx, cashflowsRemaining, valuations.Repayments[lastUsedCIdx], valuations.Coupons[lastUsedCIdx], valuations.CashflowSteps[lastUsedCIdx]);
#endif
    std::fill_n(price, c.width, valuations.Repayments[lastUsedCIdx] + valuations.Coupons[lastUsedCIdx]); // initialize to par/face value: last repayment + last coupon
    cashflowsRemaining--;
    auto lastCStep = valuations.CashflowSteps[lastUsedCIdx] <= c.n && cashflowsRemaining > 0 ? valuations.CashflowSteps[--lastUsedCIdx] : valuations.CashflowSteps[lastUsedCIdx];

    for (auto i = c.n - 1; i >= 0; --i)
    {
        auto jhigh = MIN(i, c.jmax);
        auto expmAlphadt = alphas[i];

        const auto isExerciseStep = i <= c.lastExerciseStep && i >= c.firstExerciseStep && (lastCStep - i) % c.exerciseStepFrequency == 0;
#ifdef DEV2
        if (idx == PRINT_IDX)
            printf("%d %d: %.18f %d %d %d %d\n", idx, i, expmAlphadt, isExerciseStep, lastCStep, (lastCStep - i) % c.exerciseStepFrequency, (lastCStep - i) % c.exerciseStepFrequency == 0);
#endif
        // add coupon and repayments  if crossed a time step with a cashflow
        if (i == lastCStep - 1 && cashflowsRemaining > 0)
        {
#ifdef DEV2
            if (idx == PRINT_IDX) printf("%d %d: %d %d coupon: %.18f\n", idx, i, lastUsedCIdx, cashflowsRemaining, price[c.jmax]);
#endif
            for (auto j = -jhigh; j <= jhigh; ++j)
            {
                const auto jind = j + c.jmax;
                price[jind] += valuations.Repayments[lastUsedCIdx] + valuations.Coupons[lastUsedCIdx];
            }
#ifdef DEV2
            if (idx == PRINT_IDX) printf("%d %d: %d %d coupon: %.18f\n", idx, i, lastUsedCIdx, cashflowsRemaining, price[c.jmax]);
#endif
            lastUsedCIdx--;
            lastCStep = valuations.CashflowSteps[lastUsedCIdx];
            cashflowsRemaining--;
        }

        // calculate accrued interest from last cashflow
        const auto ai = isExerciseStep && lastCStep != 0 && cashflowsRemaining > 0 ? computeAccruedInterest(i, lastCStep, valuations.CashflowSteps[lastUsedCIdx + 1], valuations.Coupons[lastUsedCIdx]) : zero;
#ifdef DEV2
        if (idx == PRINT_IDX && i == lastCStep - 1)
            printf("%d %d: ai %f %d %d %f %d %d %f\n", idx, i, ai, lastCStep, valuations.CashflowSteps[lastUsedCIdx + 1], valuations.Coupons[lastUsedCIdx],
                valuations.CashflowSteps[lastUsedCIdx + 1] - lastCStep, valuations.CashflowSteps[lastUsedCIdx + 1] - i,
                (real)(valuations.CashflowSteps[lastUsedCIdx + 1] - lastCStep - valuations.CashflowSteps[lastUsedCIdx + 1] - i) / (valuations.CashflowSteps[lastUsedCIdx + 1] - lastCStep));
#endif

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            auto jind = j + c.jmax;      // array index for j
            auto jval = jvalues[jind]; // precomputed probabilities and rates
            auto discFactor = expmAlphadt * jval.rate * c.expmOasdt;

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (jval.pu * price[jind] +
                    jval.pm * price[jind - 1] +
                    jval.pd * price[jind - 2]) *
                    discFactor;
            }
            else if (j == -c.jmax)
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
#ifdef DEV2
        if (idx == PRINT_IDX) printf("%d %d: %.18f\n", idx, i, price[c.jmax]);
#endif
    }

    auto result = price[c.jmax];
#ifdef DEV
    if (idx == PRINT_IDX) printf("%d: res %.18f\n", idx, result);
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
