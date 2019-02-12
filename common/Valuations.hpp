#ifndef OPTIONS_HPP
#define OPTIONS_HPP

#include <cinttypes>
#include <fstream>
#include <vector>

#include "Arrays.hpp"
#include "Real.hpp"

namespace trinom
{

enum class SortType : char
{
    WIDTH_DESC = 'W',
    WIDTH_ASC = 'w',
    HEIGHT_DESC = 'H',
    HEIGHT_ASC = 'h',
    NONE = '-'
};

enum class OptionTypeE : int8_t
{
    CALL_VANILLA = 0,
    PUT_VANILLA = 1
};

inline std::ostream &operator<<(std::ostream &os, const OptionTypeE t)
{
    os << static_cast<int>(t);
    return os;
}

inline std::istream &operator>>(std::istream &is, OptionTypeE &t)
{
    int c;
    is >> c;
    t = static_cast<OptionTypeE>(c);
    if (OptionTypeE::CALL_VANILLA != t && OptionTypeE::PUT_VANILLA != t)
    {
        throw std::out_of_range("Invalid OptionType read from stream.");
    }
    return is;
}

struct Valuations
{
    size_t ValuationCount;
    size_t YieldCurveCount;
    // Model parameters
    std::vector<uint16_t> TermUnits;
    std::vector<uint16_t> TermSteps;
    std::vector<real> MeanReversionRates;
    std::vector<real> Volatilities;
    // Bond parameters
    std::vector<real> Maturities;
    std::vector<uint16_t> Cashflows;
    std::vector<uint16_t> CashflowSteps;
    std::vector<real> Repayments;
    std::vector<real> Coupons;
    std::vector<real> Spreads;
    // Option parameters
    std::vector<OptionTypeE> OptionTypes;
    std::vector<real> StrikePrices;
    std::vector<uint16_t> FirstExerciseSteps;
    std::vector<uint16_t> LastExerciseSteps;
    std::vector<uint16_t> ExerciseStepFrequencies;
    // Yield Curve parameters
    std::vector<uint16_t> YieldCurveIndices;

    std::vector<uint16_t> YieldCurveTerms;
    std::vector<real> YieldCurveRates;
    std::vector<uint16_t> YieldCurveTimeSteps;

    std::vector<int32_t> CashflowIndices; // TODO: initialize
    std::vector<int32_t> YieldCurveTermIndices;

    Valuations(const int count)
    {
        ValuationCount = count;

        TermUnits.reserve(ValuationCount);
        TermSteps.reserve(ValuationCount);
        MeanReversionRates.reserve(ValuationCount);
        Volatilities.reserve(ValuationCount);

        Maturities.reserve(ValuationCount);
        Cashflows.reserve(ValuationCount);
        //CashflowSteps.reserve(ValuationCount); // TODO: not N
        //Repayments.reserve(ValuationCount); // TODO: not N
        //Coupons.reserve(ValuationCount); // TODO: not N
        Spreads.reserve(ValuationCount);

        OptionTypes.reserve(ValuationCount);
        StrikePrices.reserve(ValuationCount);
        FirstExerciseSteps.reserve(ValuationCount);
        LastExerciseSteps.reserve(ValuationCount);
        ExerciseStepFrequencies.reserve(ValuationCount);
        YieldCurveIndices.reserve(ValuationCount);

        //YieldCurveTerms.reserve(ValuationCount);  // TODO: not N
        //YieldCurveRates.reserve(ValuationCount);  // TODO: not N
        //YieldCurveTimeSteps.reserve(ValuationCount);  // TODO: not N
    }

    Valuations(const std::string &filename)
    {
        if (filename.empty())
        {
            throw std::invalid_argument("File not specified.");
        }

        std::ifstream in(filename);

        if (!in)
        {
            throw std::invalid_argument("File '" + filename + "' does not exist.");
        }

        Arrays::read_array(in, TermUnits);
        Arrays::read_array(in, TermSteps);
        Arrays::read_array(in, MeanReversionRates);
        Arrays::read_array(in, Volatilities);

        Arrays::read_array(in, Maturities);
        Arrays::read_array(in, Cashflows);
        Arrays::read_array(in, CashflowSteps);
        Arrays::read_array(in, Repayments);
        Arrays::read_array(in, Coupons);
        Arrays::read_array(in, Spreads);

        Arrays::read_array(in, OptionTypes);
        Arrays::read_array(in, StrikePrices);
        Arrays::read_array(in, FirstExerciseSteps);
        Arrays::read_array(in, LastExerciseSteps);
        Arrays::read_array(in, ExerciseStepFrequencies);
        Arrays::read_array(in, YieldCurveIndices);

        Arrays::read_array(in, YieldCurveTerms);
        Arrays::read_array(in, YieldCurveRates);
        Arrays::read_array(in, YieldCurveTimeSteps);

        ValuationCount = StrikePrices.size();
        YieldCurveCount = YieldCurveTerms.size();

        in.close();
    }

    void writeToFile(const std::string &filename)
    {
        if (filename.empty())
        {
            throw std::invalid_argument("File not specified.");
        }

        std::ofstream out(filename);

        if (!out)
        {
            throw std::invalid_argument("File does not exist.");
        }

        Arrays::write_array(out, TermUnits);
        Arrays::write_array(out, TermSteps);
        Arrays::write_array(out, MeanReversionRates);
        Arrays::write_array(out, Volatilities);

        Arrays::write_array(out, Maturities);
        Arrays::write_array(out, Cashflows);
        Arrays::write_array(out, CashflowSteps);
        Arrays::write_array(out, Repayments);
        Arrays::write_array(out, Coupons);
        Arrays::write_array(out, Spreads);

        Arrays::write_array(out, OptionTypes);
        Arrays::write_array(out, StrikePrices);
        Arrays::write_array(out, FirstExerciseSteps);
        Arrays::write_array(out, LastExerciseSteps);
        Arrays::write_array(out, ExerciseStepFrequencies);

        Arrays::write_array(out, YieldCurveIndices);

        Arrays::write_array(out, YieldCurveTerms);
        Arrays::write_array(out, YieldCurveRates);
        Arrays::write_array(out, YieldCurveTimeSteps);

        out.close();
    }
};
} // namespace trinom

#endif