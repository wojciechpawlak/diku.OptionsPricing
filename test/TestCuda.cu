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
    Valuations valuations("../data/0_UNIFORM_1_CALL_PUT_EU_Berm_US_oas.in");
    
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
