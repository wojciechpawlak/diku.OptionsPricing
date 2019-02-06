#ifndef CUDA_MULTI_VERSION_3_CUH
#define CUDA_MULTI_VERSION_3_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

namespace cuda
{

namespace multi
{  

struct KernelArgsValuesCoalescedBlock
{
    real *res;
    real *alphas;
    int32_t *inds;
    int32_t *alphaInds;
    int32_t maxOptionsBlock;
};

class KernelArgsCoalescedBlock : public KernelArgsBase<KernelArgsValuesCoalescedBlock>
{

private:

    int optionIdx;
    int optionCountBlock;
    int alphaIdx;
    int alphaIdxBlock;
    int maxHeight;

public:

    KernelArgsCoalescedBlock(KernelArgsValuesCoalescedBlock &v) : KernelArgsBase(v) { }
    
    __device__ inline void init(const int optionIdxBlock, const int idxBlock, const int idxBlockNext, const int optionCount)
    {
        optionIdx = idxBlock + optionIdxBlock;
        optionCountBlock = idxBlockNext - idxBlock;
        alphaIdxBlock = (blockIdx.x == 0 ? 0 : values.alphaInds[blockIdx.x - 1]);
        maxHeight = (values.alphaInds[blockIdx.x] - alphaIdxBlock) / optionCountBlock;
        alphaIdx = alphaIdxBlock + optionIdxBlock;
    }

    __device__ inline void setAlphaAt(const int index, const real value, const int optionIndex = 0) override
    {
        values.alphas[optionCountBlock * index + alphaIdxBlock + optionIndex] = value;
    }

    __device__ inline real getAlphaAt(const int index, const int optionIndex = 0) const override
    {
        return values.alphas[optionCountBlock * index + alphaIdxBlock + optionIndex];
    }

    __device__ inline int getMaxHeight() const override
    {
        return maxHeight;
    }

    __device__ inline int getOptionIdx() const override
    {
        return optionIdx;
    }
};

class KernelRunCoalescedBlock : public KernelRunBase
{

protected:
    void runPreprocessing(CudaValuations &valuations, std::vector<real> &results) override
    {
        // Compute indices.
        thrust::host_vector<int32_t> hostWidths = valuations.Widths;
        thrust::host_vector<int32_t> hostHeights = valuations.Heights;
        thrust::host_vector<int32_t> hInds;
        thrust::host_vector<int32_t> hAlphaInds;

        auto counter = 0;
        auto maxHeightBlock = 0;
        auto prevInd = 0;
        auto maxOptionsBlock = 0;
        for (auto i = 0; i < valuations.ValuationCount; ++i)
        {
            auto w = hostWidths[i];
            auto h = hostHeights[i];

            counter += w;
            if (counter > BlockSize)
            {
                auto alphasBlock = maxHeightBlock * (i - (hInds.empty() ? 0 : hInds.back()));
                hAlphaInds.push_back((hAlphaInds.empty() ? 0 : hAlphaInds.back()) + alphasBlock);
                hInds.push_back(i);
                counter = w;
                maxHeightBlock = 0;

                auto optionsBlock = i - prevInd;
                if (optionsBlock > maxOptionsBlock) {
                    maxOptionsBlock = optionsBlock;
                }
                prevInd = i;
            }
            if (h > maxHeightBlock) {
                maxHeightBlock = h;
            }
        }
        auto alphasBlock = maxHeightBlock * (valuations.ValuationCount - (hInds.empty() ? 0 : hInds.back()));
        hAlphaInds.push_back((hAlphaInds.empty() ? 0 : hAlphaInds.back()) + alphasBlock);
        hInds.push_back(valuations.ValuationCount);

        auto optionsBlock = valuations.ValuationCount - prevInd;
        if (optionsBlock > maxOptionsBlock) {
            maxOptionsBlock = optionsBlock;
        }

        thrust::device_vector<int32_t> dInds = hInds;
        thrust::device_vector<int32_t> dAlphaInds = hAlphaInds;
        auto totalAlphasCount = hAlphaInds.back();

        KernelArgsValuesCoalescedBlock values;
        values.alphaInds = thrust::raw_pointer_cast(dAlphaInds.data());

        valuations.DeviceMemory += vectorsizeof(dAlphaInds);

        runKernel<KernelArgsCoalescedBlock>(valuations, results, dInds, values, totalAlphasCount, maxOptionsBlock);
    }
};

}

}

#endif
