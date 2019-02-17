#ifndef CUDA_DOMAIN_CUH
#define CUDA_DOMAIN_CUH

#include <chrono>

#include <thrust/device_vector.h>
#include <thrust/transform_scan.h>

#include "../common/Valuations.hpp"
#include "../common/ValuationConstants.hpp"
#include "../common/Domain.hpp"
#include "../cuda/CudaErrors.cuh"

using namespace trinom;

namespace cuda
{

struct KernelValuations
{
    int ValuationCount;
    int YieldCurveCount;

    // Model parameters
    uint16_t *TermUnits;
    uint16_t *TermSteps;
    real *MeanReversionRates;
    real *Volatilities;
    // Bond parameters
    real *Maturities;
    uint16_t *Cashflows;
    uint16_t *CashflowSteps;
    real *Repayments;
    real *Coupons;
    real *Spreads;
    // Option parameters
    OptionTypeE *OptionTypes;
    real *StrikePrices;
    uint16_t *FirstExerciseSteps;
    uint16_t *LastExerciseSteps;
    uint16_t *ExerciseStepFrequencies;
    // Yield Curve parameters
    uint16_t *YieldCurveIndices;
    uint16_t *YieldCurveTerms;
    real *YieldCurveRates;
    uint16_t *YieldCurveTimeSteps;
    // Kernel parameters
    int32_t *Widths;
    int32_t *Heights;
    int32_t *CashflowIndices;
    int32_t *YieldCurveTermIndices;
};

struct compute_width_height
{
    template <typename Tuple>
    __host__ __device__ void operator()(Tuple t)
    {
        // Tuple(TermUnit, TermStepCount, Maturity, ReversionRate, Widths, Height)
        real termUnit = thrust::get<0>(t);
        real termStepCount = thrust::get<1>(t);
        real maturity = thrust::get<2>(t);
        real a = thrust::get<3>(t);
        int termUnitsInYearCount = (int)ceil((real)year / termUnit);
        real dt = termUnitsInYearCount / termStepCount;               // [years]
        real M = exp(-a * dt) - one;
        int jmax = (int)(minus184 / M) + 1;

        thrust::get<4>(t) = 2 * jmax + 1;                                          // width
        thrust::get<5>(t) = termStepCount * termUnitsInYearCount * maturity + 1;   // height + 1
    }
};

struct sort_tuple_asc
{
    typedef thrust::tuple<int32_t, int32_t> Tuple;
    __host__ __device__ bool operator()(const Tuple& t1, const Tuple& t2)
    {
        return (t1.get<0>() < t2.get<0>() || (t1.get<0>() == t2.get<0>() && t1.get<1>() < t2.get<1>()));
    }
};

struct sort_tuple_desc
{
    typedef thrust::tuple<int32_t, int32_t> Tuple;
    __host__ __device__ bool operator()(const Tuple& t1, const Tuple& t2)
    {
        return (t1.get<0>() > t2.get<0>() || (t1.get<0>() == t2.get<0>() && t1.get<1>() > t2.get<1>()));
    }
};

template<typename T>
size_t vectorsizeof(const typename thrust::device_vector<T>& vec)
{
    return sizeof(T) * vec.size();
}

class CudaValuations
{
private:
    // Model parameters
    thrust::device_vector<uint16_t> TermUnits;
    thrust::device_vector<uint16_t> TermSteps;
    thrust::device_vector<real> MeanReversionRates;
    thrust::device_vector<real> Volatilities;
    // Bond parameters
    thrust::device_vector<real> Maturities;
    thrust::device_vector<uint16_t> Cashflows; // Not used in kernels
    thrust::device_vector<uint16_t> CashflowSteps;
    thrust::device_vector<real> Repayments;
    thrust::device_vector<real> Coupons;
    thrust::device_vector<real> Spreads;
    // Option parameters
    thrust::device_vector<OptionTypeE> OptionTypes;
    thrust::device_vector<real> StrikePrices;
    thrust::device_vector<uint16_t> FirstExerciseSteps;
    thrust::device_vector<uint16_t> LastExerciseSteps;
    thrust::device_vector<uint16_t> ExerciseStepFrequencies;
    // Yield Curve parameters
    thrust::device_vector<uint16_t> YieldCurveIndices;
    thrust::device_vector<uint16_t> YieldCurveTerms;
    thrust::device_vector<real> YieldCurveRates;
    thrust::device_vector<uint16_t> YieldCurveTimeSteps;

public:
    const int ValuationCount;
    const int YieldCurveCount;
    KernelValuations KernelValuations;

    thrust::device_vector<int32_t> Widths;
    thrust::device_vector<int32_t> Heights;

    thrust::device_vector<int32_t> CashflowIndices;
    thrust::device_vector<int32_t> YieldCurveTermIndices;
    thrust::device_vector<int32_t> ValuationIndices;

    long DeviceMemory = 0;

    CudaValuations(const Valuations &valuations) : 
        TermUnits(valuations.TermUnits.begin(), valuations.TermUnits.end()),
        TermSteps(valuations.TermSteps.begin(), valuations.TermSteps.end()),
        MeanReversionRates(valuations.MeanReversionRates.begin(), valuations.MeanReversionRates.end()),
        Volatilities(valuations.Volatilities.begin(), valuations.Volatilities.end()),
        Maturities(valuations.Maturities.begin(), valuations.Maturities.end()),
        Cashflows(valuations.Cashflows.begin(), valuations.Cashflows.end()),
        CashflowSteps(valuations.CashflowSteps.begin(), valuations.CashflowSteps.end()),
        Repayments(valuations.Repayments.begin(), valuations.Repayments.end()),
        Coupons(valuations.Coupons.begin(), valuations.Coupons.end()),
        Spreads(valuations.Spreads.begin(), valuations.Spreads.end()),
        OptionTypes(valuations.OptionTypes.begin(), valuations.OptionTypes.end()),
        StrikePrices(valuations.StrikePrices.begin(), valuations.StrikePrices.end()),
        FirstExerciseSteps(valuations.FirstExerciseSteps.begin(), valuations.FirstExerciseSteps.end()),
        LastExerciseSteps(valuations.LastExerciseSteps.begin(), valuations.LastExerciseSteps.end()),
        ExerciseStepFrequencies(valuations.ExerciseStepFrequencies.begin(), valuations.ExerciseStepFrequencies.end()),
        YieldCurveIndices(valuations.YieldCurveIndices.begin(), valuations.YieldCurveIndices.end()),
        YieldCurveTerms(valuations.YieldCurveTerms.begin(), valuations.YieldCurveTerms.end()),
        YieldCurveRates(valuations.YieldCurveRates.begin(), valuations.YieldCurveRates.end()),
        YieldCurveTimeSteps(valuations.YieldCurveTimeSteps.begin(), valuations.YieldCurveTimeSteps.end()),
        ValuationCount(valuations.ValuationCount),
        YieldCurveCount(valuations.YieldCurveCount)
    {

    }

    void initialize()
    {
        Widths.resize(ValuationCount);
        Heights.resize(ValuationCount);
        CashflowIndices.resize(ValuationCount);
        YieldCurveTermIndices.resize(ValuationCount);

        KernelValuations.ValuationCount = ValuationCount;
        KernelValuations.YieldCurveCount = YieldCurveCount,

        KernelValuations.TermUnits = thrust::raw_pointer_cast(TermUnits.data());
        KernelValuations.TermSteps = thrust::raw_pointer_cast(TermSteps.data());
        KernelValuations.MeanReversionRates = thrust::raw_pointer_cast(MeanReversionRates.data());
        KernelValuations.Volatilities = thrust::raw_pointer_cast(Volatilities.data());

        KernelValuations.Maturities = thrust::raw_pointer_cast(Maturities.data());
        KernelValuations.Cashflows = thrust::raw_pointer_cast(Cashflows.data());
        KernelValuations.CashflowSteps = thrust::raw_pointer_cast(CashflowSteps.data());
        KernelValuations.Repayments = thrust::raw_pointer_cast(Repayments.data());
        KernelValuations.Coupons = thrust::raw_pointer_cast(Coupons.data());
        KernelValuations.Spreads = thrust::raw_pointer_cast(Spreads.data());

        KernelValuations.OptionTypes = thrust::raw_pointer_cast(OptionTypes.data());
        KernelValuations.StrikePrices = thrust::raw_pointer_cast(StrikePrices.data());
        KernelValuations.FirstExerciseSteps = thrust::raw_pointer_cast(FirstExerciseSteps.data());
        KernelValuations.LastExerciseSteps = thrust::raw_pointer_cast(LastExerciseSteps.data());
        KernelValuations.ExerciseStepFrequencies = thrust::raw_pointer_cast(ExerciseStepFrequencies.data());

        KernelValuations.YieldCurveIndices = thrust::raw_pointer_cast(YieldCurveIndices.data());
        KernelValuations.YieldCurveTerms = thrust::raw_pointer_cast(YieldCurveTerms.data());
        KernelValuations.YieldCurveRates = thrust::raw_pointer_cast(YieldCurveRates.data());
        KernelValuations.YieldCurveTimeSteps = thrust::raw_pointer_cast(YieldCurveTimeSteps.data());

        KernelValuations.Widths = thrust::raw_pointer_cast(Widths.data());
        KernelValuations.Heights = thrust::raw_pointer_cast(Heights.data());
        KernelValuations.CashflowIndices = thrust::raw_pointer_cast(CashflowIndices.data());
        KernelValuations.YieldCurveTermIndices = thrust::raw_pointer_cast(YieldCurveTermIndices.data());

        DeviceMemory += vectorsizeof(TermUnits);
        DeviceMemory += vectorsizeof(TermSteps);
        DeviceMemory += vectorsizeof(MeanReversionRates);
        DeviceMemory += vectorsizeof(Volatilities);

        DeviceMemory += vectorsizeof(Maturities);
        DeviceMemory += vectorsizeof(Cashflows);
        DeviceMemory += vectorsizeof(CashflowSteps);
        DeviceMemory += vectorsizeof(Coupons);
        DeviceMemory += vectorsizeof(Spreads);
        
        DeviceMemory += vectorsizeof(OptionTypes);
        DeviceMemory += vectorsizeof(StrikePrices);
        DeviceMemory += vectorsizeof(FirstExerciseSteps);
        DeviceMemory += vectorsizeof(LastExerciseSteps);
        DeviceMemory += vectorsizeof(ExerciseStepFrequencies);
        
        DeviceMemory += vectorsizeof(YieldCurveIndices);
        DeviceMemory += vectorsizeof(YieldCurveTerms);
        DeviceMemory += vectorsizeof(YieldCurveRates);
        DeviceMemory += vectorsizeof(YieldCurveTimeSteps);
               
        DeviceMemory += vectorsizeof(Widths);
        DeviceMemory += vectorsizeof(Heights);
        DeviceMemory += vectorsizeof(CashflowIndices);
        DeviceMemory += vectorsizeof(YieldCurveTermIndices);

        // Fill in widths and heights for all valuations.
        thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(TermUnits.begin(), TermSteps.begin(), Maturities.begin(), MeanReversionRates.begin(), Widths.begin(), Heights.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(TermUnits.end(), TermSteps.end(), Maturities.end(), MeanReversionRates.end(), Widths.end(), Heights.end())),
                     compute_width_height());

        thrust::exclusive_scan(Cashflows.begin(), Cashflows.end(), CashflowIndices.begin());
        thrust::exclusive_scan(YieldCurveTerms.begin(), YieldCurveTerms.end(), YieldCurveTermIndices.begin());

        cudaDeviceSynchronize();
    }

    void sortValuations(const SortType sort, const bool isTest)
    {
        if (sort != SortType::NONE)
        {
            // Create indices
            ValuationIndices = thrust::device_vector<int32_t>(ValuationCount);
            thrust::sequence(ValuationIndices.begin(), ValuationIndices.end());
            DeviceMemory += vectorsizeof(ValuationIndices);

            auto optionBegin = thrust::make_zip_iterator(thrust::make_tuple(
                //TermUnits.begin(),
                //TermSteps.begin(),
                MeanReversionRates.begin(),
                Volatilities.begin(),
                Cashflows.begin(),
                //Spreads.begin(),
                //OptionTypes.begin(),
                StrikePrices.begin(),
                FirstExerciseSteps.begin(),
                LastExerciseSteps.begin(),
                ExerciseStepFrequencies.begin(),
                YieldCurveIndices.begin(),
                CashflowIndices.begin(),
                ValuationIndices.begin()
            ));
    
            auto keysBegin = (sort == SortType::WIDTH_ASC || sort == SortType::WIDTH_DESC) 
                ? thrust::make_zip_iterator(thrust::make_tuple(Widths.begin(), Heights.begin()))
                : thrust::make_zip_iterator(thrust::make_tuple(Heights.begin(), Widths.begin()));
            auto keysEnd = (sort == SortType::WIDTH_ASC || sort == SortType::WIDTH_DESC) 
                ? thrust::make_zip_iterator(thrust::make_tuple(Widths.end(), Heights.end()))
                : thrust::make_zip_iterator(thrust::make_tuple(Heights.end(), Widths.end()));

            // Sort options
            switch (sort)
            {
                case SortType::WIDTH_ASC:
                    if (isTest) std::cout << "Ascending sort, width first, height second" << std::endl;
                case SortType::HEIGHT_ASC:
                    if (isTest && sort == SortType::HEIGHT_ASC) std::cout << "Ascending sort, height first, width second" << std::endl;
                    thrust::sort_by_key(keysBegin, keysEnd, optionBegin, sort_tuple_asc());
                    break;
                case SortType::WIDTH_DESC:
                    if (isTest) std::cout << "Descending sort, width first, height second" << std::endl;
                case SortType::HEIGHT_DESC:
                    if (isTest && sort == SortType::HEIGHT_DESC) std::cout << "Descending sort, height first, width second" << std::endl;
                    thrust::sort_by_key(keysBegin, keysEnd, optionBegin, sort_tuple_desc());
                    break;
            }
            cudaDeviceSynchronize();
        }
    }

    void sortResult(thrust::device_vector<real> &deviceResults)
    {
        // Sort result
        if (!ValuationIndices.empty())
        {
            thrust::sort_by_key(ValuationIndices.begin(), ValuationIndices.end(), deviceResults.begin());
            cudaDeviceSynchronize();
        }
    }
};

struct CudaRuntime
{
    long KernelRuntime = std::numeric_limits<long>::max();
    long TotalRuntime = std::numeric_limits<long>::max();
    long DeviceMemory = 0;
};

bool operator <(const CudaRuntime& x, const CudaRuntime& y) {
    return std::tie(x.KernelRuntime, x.TotalRuntime) < std::tie(y.KernelRuntime, y.TotalRuntime);
}

__device__ void computeConstants(ValuationConstants &c, const KernelValuations &valuations, const int idx)
{
    c.termUnit = valuations.TermUnits[idx];
    const auto T = valuations.Maturities[idx];
    const auto termUnitsInYearCount = (int)ceil((real)year / c.termUnit);
    const auto termStepCount = valuations.TermSteps[idx];
    c.n = (int)ceil(termStepCount * termUnitsInYearCount * T);
    c.dt = termUnitsInYearCount / (real)termStepCount; // [years]
    c.type = valuations.OptionTypes[idx];
    c.X = valuations.StrikePrices[idx];

    const auto a = valuations.MeanReversionRates[idx];
    const auto sigma = valuations.Volatilities[idx];
    const auto V = sigma * sigma * (one - exp(-two * a * c.dt)) / (two * a);
    const auto dr = sqrt(three * V);
    c.M = exp(-a * c.dt) - one;

    // simplified computations
    // c.dr = sigma * sqrt(three * c.dt);
    // c.M = -a * c.dt;

    c.mdrdt = -dr * c.dt;
    c.jmax = (int)(minus184 / c.M) + 1;
    c.expmOasdt = exp(-(valuations.Spreads[idx] / hundred)*c.dt);

    c.width = 2 * c.jmax + 1;

    c.lastExerciseStep = valuations.LastExerciseSteps[idx];
    c.firstExerciseStep = valuations.FirstExerciseSteps[idx];
    c.exerciseStepFrequency = valuations.ExerciseStepFrequencies[idx];

    const auto firstYCTermIdx = valuations.YieldCurveTermIndices[valuations.YieldCurveIndices[idx]];
    c.firstYieldCurveRate = &valuations.YieldCurveRates[c.firstYCTermIdx];
    c.firstYieldCurveTimeStep = &valuations.YieldCurveTimeSteps[c.firstYCTermIdx];
    c.yieldCurveTermCount = valuations.YieldCurveTerms[valuations.YieldCurveIndices[idx]];
}

}
#endif
