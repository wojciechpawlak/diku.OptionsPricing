#ifndef OPTION_CONSTANTS_HPP
#define OPTION_CONSTANTS_HPP

#include <vector>
#include <cmath>

#include "CudaInterop.h"
#include "Valuations.hpp"

namespace trinom
{

struct ValuationConstants
{
    real dt; // [years]
    real mdrdt; // -dr * dt
    real expmOasdt; // exponent of option adjusted spread - exp(-oas * dt)
    real X;
    real M;
    int32_t jmax;
    int32_t n;
    int32_t width;
    uint16_t termUnit;
    OptionTypeE type; // char

    uint16_t lastExerciseStep;
    uint16_t firstExerciseStep;
    uint16_t exerciseStepFrequency;

    uint16_t firstYCTermIdx;
    const real *firstYieldCurveRate;
    const uint16_t *firstYieldCurveTimeStep;
    uint16_t yieldCurveTermCount;

    DEVICE HOST ValuationConstants() {}

    ValuationConstants(const Valuations &valuations, const int idx)
    {
        termUnit = valuations.TermUnits.at(idx);
        const auto T = valuations.Maturities.at(idx);
        const auto termUnitsInYearCount = (int)ceil((real)year / termUnit);
        const auto termStepCount = valuations.TermSteps.at(idx);
        n = (int)ceil((real)termStepCount * termUnitsInYearCount * T);
        dt = termUnitsInYearCount / (real)termStepCount; // [years]
        type = valuations.OptionTypes.at(idx);
        X = valuations.StrikePrices.at(idx);

        auto a = valuations.MeanReversionRates.at(idx);
        auto sigma = valuations.Volatilities.at(idx);
        auto V = sigma * sigma * (one - exp(-two * a * dt)) / (two * a);
        auto dr = sqrt(three * V);
        M = exp(-a * dt) - one;

        // simplified computations
        // dr = sigma * sqrt(three * dt);
        // M = -a * dt;

        mdrdt = -dr*dt;
        jmax = (int)(minus184 / M) + 1;
        expmOasdt = exp(-(valuations.Spreads[idx] / hundred)*dt);
        width = 2 * jmax + 1;

        //assert(valuations.YieldCurveIndices != NULL);
        //assert(valuations.CashflowIndices != NULL);

        lastExerciseStep = valuations.LastExerciseSteps[idx];
        firstExerciseStep = valuations.FirstExerciseSteps[idx];
        exerciseStepFrequency = valuations.ExerciseStepFrequencies[idx];

        auto firstYCTermIdx = valuations.YieldCurveTermIndices[valuations.YieldCurveIndices[idx]];
        firstYieldCurveRate = &valuations.YieldCurveRates[firstYCTermIdx];
        firstYieldCurveTimeStep = &valuations.YieldCurveTimeSteps[firstYCTermIdx];
        yieldCurveTermCount = valuations.YieldCurveTerms[valuations.YieldCurveIndices[idx]];
    }
};
} // namespace trinom

#endif
