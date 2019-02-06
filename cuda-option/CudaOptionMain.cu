
#include "Version1.cuh"
#include "Version2.cuh"
#include "Version3.cuh"
#include "../common/Args.hpp"

using namespace std;
using namespace trinom;

cuda::CudaRuntime run(const Valuations &valuations, vector<real> &results,
    const int version, const int blockSize, const SortType sortType, const bool isTest)
{
    switch (version)
    {
    case 1:
    {
        cuda::option::KernelRunNaive kernelRun;
        kernelRun.run(valuations, results, blockSize, sortType, isTest);
        return kernelRun.runtime;
    }
    case 2:
    {
        cuda::option::KernelRunCoalesced kernelRun;
        kernelRun.run(valuations, results, blockSize, sortType, isTest);
        return kernelRun.runtime;
    }
    case 3:
    {
        cuda::option::KernelRunCoalescedGranular kernelRun(blockSize);  // block-level padding granularity
        kernelRun.run(valuations, results, blockSize, sortType, isTest);
        return kernelRun.runtime;
    }
    case 4:
    {
        cuda::option::KernelRunCoalescedGranular kernelRun(32);         // warp-level padding granularity
        kernelRun.run(valuations, results, blockSize, sortType, isTest);
        return kernelRun.runtime;
    }
    }
    return cuda::CudaRuntime();
}

void computeValuations(const Valuations &valuations, const int version,
    const int blockSize, const SortType sortType, const int runs, const bool isTest)
{
    if (isTest)
    {
        cout << "Cuda one valuation per thread version " << version << endl;
    }

    if (runs > 0)
    {
        if (isTest)
        {
            cout << "Performing " << runs << " runs..." << endl;
        }
        cuda::CudaRuntime best;
        for (auto i = 0; i < runs; ++i)
        {
            vector<real> results;
            results.resize(valuations.ValuationCount);
            auto runtime = run(valuations, results, version, blockSize, sortType, isTest);
            if (runtime < best)
            {
                best = runtime;
            }
        }
        if (isTest)
        {
            cout << "Best times: kernel " << best.KernelRuntime << " microsec, total " << best.TotalRuntime << " microsec." << endl;
        }
        else
        {
            cout << version << ',' << blockSize << ',' << (char)sortType << ',' << best.KernelRuntime << ',' << best.TotalRuntime << ',' << best.DeviceMemory << endl;
        }
    }
    else
    {
        vector<real> results;
        results.resize(valuations.ValuationCount);
        run(valuations, results, version, blockSize, sortType, isTest);

        if (!isTest)
        {
            Arrays::write_array(cout, results);
        }
    }
}

int main(int argc, char *argv[])
{
    Args args(argc, argv);

    if (args.test)
    {
        cout << "Loading valuations " << args.valuations << endl;
    }

    // Read valuations and yield curve.
    Valuations valuations(args.valuations);

    // Initialize cuda device.
    cudaSetDevice(args.device);
    cudaFree(0);

    for (auto &version : args.versions)
    {
        for (auto &blockSize : args.blockSizes)
        {
            for (auto &sortType : args.sorts)
            {
                computeValuations(valuations, version, blockSize, sortType, args.runs, args.test);
            }
        }
    }

    return 0;
}
