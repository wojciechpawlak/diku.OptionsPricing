#include "../cuda-option/Version1.cuh"
#include "../cuda-option/Version2.cuh"
#include "../cuda-option/Version3.cuh"
#include "../cuda-multi/Version1.cuh"
#include "../cuda-multi/Version2.cuh"
#include "../cuda-multi/Version3.cuh"
#include "../seq/Seq.hpp"

#include "Compare.hpp"

using namespace std;
using namespace trinom;

TEST_CASE("Book options")
{
    Valuations valuations(100);
    for (int i = 1; i <= valuations.ValuationCount; ++i)
    {
        valuations.TermUnits.push_back(360);
        valuations.TermSteps.push_back(i);
        valuations.MeanReversionRates.push_back(0.1);
        valuations.Volatilities.push_back(0.01);

        valuations.Maturities.push_back(9);
        valuations.Cashflows.push_back(3);
        valuations.CashflowSteps.push_back(3);
        valuations.Repayments.push_back(3);
        valuations.Coupons.push_back(3);
        valuations.Spreads.push_back(3);

        valuations.OptionTypes.push_back(OptionType::PUT_VANILLA);
        valuations.StrikePrices.push_back(63);
        valuations.FirstExerciseSteps.push_back(63);
        valuations.LastExerciseSteps.push_back(63);
        valuations.ExerciseStepFrequencies.push_back(63);

        valuations.ExerciseStepFrequencies.push_back(63);

        valuations.ExerciseStepFrequencies.push_back(63);
        valuations.ExerciseStepFrequencies.push_back(63);
        valuations.ExerciseStepFrequencies.push_back(63);
    }
    
    vector<real> seqResults, cudaResults;
    seqResults.resize(valuations.ValuationCount);
    seq::computeOptions(valuations, seqResults);

    SECTION("CUDA option version 1")
    {
        vector<real> results;
        results.resize(valuations.ValuationCount);
        cuda::option::KernelRunNaive kernelRun;
        kernelRun.run(valuations, results, 64);
        compareVectors(results, seqResults);
    }

    SECTION("CUDA option version 2")
    {
        vector<real> results;
        results.resize(valuations.ValuationCount);
        cuda::option::KernelRunCoalesced kernelRun;
        kernelRun.run(valuations, results, 64);
        compareVectors(results, seqResults);
    }

    SECTION("CUDA option version 3")
    {
        vector<real> results;
        results.resize(valuations.ValuationCount);
        cuda::option::KernelRunCoalescedGranular kernelRun(64);
        kernelRun.run(valuations, results, 64);
        compareVectors(results, seqResults);
    }

    SECTION("CUDA option version 4")
    {
        vector<real> results;
        results.resize(valuations.ValuationCount);
        cuda::option::KernelRunCoalescedGranular kernelRun(32);
        kernelRun.run(valuations, results, 64);
        compareVectors(results, seqResults);
    }

    SECTION("CUDA multi version 1")
    {
        vector<real> results;
        results.resize(valuations.ValuationCount);
        cuda::multi::KernelRunNaive kernelRun;
        kernelRun.run(valuations, results, 512);
        compareVectors(results, seqResults);
    }

    SECTION("CUDA multi version 2")
    {
        vector<real> results;
        results.resize(valuations.ValuationCount);
        cuda::multi::KernelRunCoalesced kernelRun;
        kernelRun.run(valuations, results, 512);
        compareVectors(results, seqResults);
    }

    SECTION("CUDA multi version 3")
    {
        vector<real> results;
        results.resize(valuations.ValuationCount);
        cuda::multi::KernelRunCoalescedBlock kernelRun;
        kernelRun.run(valuations, results, 512);
        compareVectors(results, seqResults);
    }
}
