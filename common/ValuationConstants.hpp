#ifndef OPTION_CONSTANTS_HPP
#define OPTION_CONSTANTS_HPP

#include <vector>
#include <cmath>
#include <cassert>

#include "CudaInterop.h"
#include "Valuations.hpp"

namespace trinom
{

struct ValuationConstants
{
    real dt; // [years]
    real dr;
    real X;
    real M;
    int32_t jmax;
    int32_t n;
    int32_t width;
    uint16_t termUnit;
    uint16_t termStepCount;
    OptionType type; // char

    uint16_t firstYCTermIdx;

    uint16_t LastExerciseStep;
    uint16_t FirstExerciseStep;
    uint16_t ExerciseStepFrequency;

    const real *firstYieldCurveRate;
    const uint16_t *firstYieldCurveTimeStep;
    uint16_t yieldCurveTermCount;

    DEVICE HOST ValuationConstants() {}

    ValuationConstants(const Valuations &valuations, const int idx)
    {
        termUnit = valuations.TermUnits.at(idx);
        const auto T = valuations.Maturities.at(idx);
        const auto termUnitsInYearCount = lround((real)year / termUnit);
        termStepCount = valuations.TermSteps.at(idx);
        n = (int)lround((real)termStepCount * termUnitsInYearCount * T);
        dt = termUnitsInYearCount / (real)termStepCount; // [years]
        type = valuations.OptionTypes.at(idx);

        auto a = valuations.MeanReversionRates.at(idx);
        X = valuations.StrikePrices.at(idx);
        auto sigma = valuations.Volatilities.at(idx);
        auto V = sigma * sigma * (one - exp(-two * a * dt)) / (two * a);
        dr = sqrt(three * V);
        M = exp(-a * dt) - one;

        // simplified computations
        // dr = sigma * sqrt(three * dt);
        // M = -a * dt;

        jmax = (int)(minus184 / M) + 1;
        width = 2 * jmax + 1;
        //assert(valuations.YieldCurveIndices != NULL);
        //assert(valuations.CashflowIndices != NULL);
        firstYCTermIdx = valuations.YieldCurveTermIndices[valuations.YieldCurveIndices[idx]];

        firstYieldCurveRate = &valuations.YieldCurveRates[firstYCTermIdx];
        firstYieldCurveTimeStep = &valuations.YieldCurveTimeSteps[firstYCTermIdx];
        yieldCurveTermCount = valuations.YieldCurveTerms[valuations.YieldCurveIndices[idx]];
    }
};
} // namespace trinom

#endif
