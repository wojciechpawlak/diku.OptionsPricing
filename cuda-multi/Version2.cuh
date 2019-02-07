#ifndef CUDA_MULTI_VERSION_2_CUH
#define CUDA_MULTI_VERSION_2_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

namespace cuda
{

namespace multi
{

struct KernelArgsValuesCoalesced
{
    real *res;
    real *alphas;
    int32_t *inds;
    int32_t maxHeight;
    int32_t maxValuationsBlock;
};

class KernelArgsCoalesced : public KernelArgsBase<KernelArgsValuesCoalesced>
{

private:

    int valuationIdx;
    int valuationCount;
    int valuationCountBlock;

public:

    KernelArgsCoalesced(KernelArgsValuesCoalesced &v) : KernelArgsBase(v) { }
    
    __device__ inline void init(const int valuationIdxBlock, const int idxBlock, const int idxBlockNext, const int valuationCount)
    {
        this->valuationIdx = idxBlock + valuationIdxBlock;
        this->valuationCount = valuationCount;
        this->valuationCountBlock = idxBlockNext - idxBlock;
    }

    __device__ inline void setAlphaAt(const int index, const real value, const int valuationIndex = 0) override
    {
        values.alphas[valuationCount * index + valuationIdx] = value;
    }

    __device__ inline real getAlphaAt(const int index, const int valuationIndex = 0) const override
    {
        return values.alphas[valuationCount * index + valuationIdx];
    }

    __device__ inline int getMaxHeight() const override
    {
        return values.maxHeight;
    }

    __device__ inline int getValuationIdx() const override
    {
        return valuationIdx;
    }
};

class KernelRunCoalesced : public KernelRunBase
{

protected:
    void runPreprocessing(CudaValuations &valuations, std::vector<real> &results) override
    {
        // Compute indices.
        thrust::host_vector<int32_t> hostWidths = valuations.Widths;
        thrust::host_vector<int32_t> hInds;

        auto counter = 0;
        auto prevInd = 0;
        auto maxValuationsBlock = 0;
        for (auto i = 0; i < valuations.ValuationCount; ++i)
        {
            auto w = hostWidths[i];
            counter += w;
            if (counter > BlockSize)
            {
                hInds.push_back(i);
                counter = w;

                auto valuationsBlock = i - prevInd;
                if (valuationsBlock > maxValuationsBlock) {
                    maxValuationsBlock = valuationsBlock;
                }
                prevInd = i;
            }
        }
        hInds.push_back(valuations.ValuationCount);

        auto valuationsBlock = valuations.ValuationCount - prevInd;
        if (valuationsBlock > maxValuationsBlock) {
            maxValuationsBlock = valuationsBlock;
        }

        thrust::device_vector<int32_t> dInds = hInds;

        KernelArgsValuesCoalesced values;

        // Get the max height
        values.maxHeight = thrust::max_element(valuations.Heights.begin(), valuations.Heights.end())[0];
        const int totalAlphasCount = valuations.ValuationCount * values.maxHeight;

        runKernel<KernelArgsCoalesced>(valuations, results, dInds, values, totalAlphasCount, maxValuationsBlock);
    }
};

}

}

#endif
