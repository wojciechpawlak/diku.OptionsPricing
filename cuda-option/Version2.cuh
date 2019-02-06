#ifndef CUDA_VERSION_2_CUH
#define CUDA_VERSION_2_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

namespace cuda
{

namespace option
{

class KernelArgsCoalesced : public KernelArgsBase<KernelArgsValues>
{
private:
    int N;

public:

    KernelArgsCoalesced(KernelArgsValues &v) : KernelArgsBase(v) { }

    __device__ inline void init(const KernelValuations &valuations) override { N = valuations.ValuationCount; }

    __device__ void fillQs(const int count, const int value) override
    {
        auto ptr = values.QsAll + getIdx();

        for (auto i = 0; i < count; ++i)
        {
            *ptr = value;
            ptr += N;
        }
    }

    __device__ inline real getRateAt(const int index) const override { return values.ratesAll[index * N + getIdx()]; }

    __device__ inline void setRateAt(const int index, const real value) override { values.ratesAll[index * N + getIdx()] = value; }

    __device__ inline real getPAt(const int index, const int branch) override
    {
        switch (branch)
        {
        case 1:
            return values.pusAll[index * N + getIdx()]; // up
        case 2:
            return values.pmsAll[index * N + getIdx()]; // mid
        case 3:
            return values.pdsAll[index * N + getIdx()]; // down
        }
        return 0;
    }

    __device__ inline void setPAt(const int index, const int branch, const real value) override
    {
        switch (branch)
        {
        case 1:
            values.pusAll[index * N + getIdx()] = value; // up
        case 2:
            values.pmsAll[index * N + getIdx()] = value; // mid
        case 3:
            values.pdsAll[index * N + getIdx()] = value; // down
        }
    }

    __device__ inline real getQAt(const int index) const override { return values.QsAll[index * N + getIdx()]; }

    __device__ inline void setQAt(const int index, const real value) override { values.QsAll[index * N + getIdx()] = value; }

    __device__ inline void setQCopyAt(const int index, const real value) override { values.QsCopyAll[index * N + getIdx()] = value; }

    __device__ inline real getAlphaAt(const int index) const override { return values.alphasAll[index * N + getIdx()]; }

    __device__ inline void setAlphaAt(const int index, const real value) override { values.alphasAll[index * N + getIdx()] = value; }

    //__device__ inline real getOptPAt(const int index) const override { return values.OptPsAll[index * N + getIdx()]; }

    //__device__ inline void setOptPAt(const int index, const real value) override { values.OptPsAll[index * N + getIdx()] = value; }

    //__device__ inline void setOptPCopyAt(const int index, const real value) override { values.OptPsCopyAll[index * N + getIdx()] = value; }

    __device__ inline void setResult(const int jmax) override { values.res[getIdx()] = values.QsAll[jmax * N + getIdx()]; }
};

class KernelRunCoalesced : public KernelRunBase
{

protected:
    void runPreprocessing(CudaValuations &valuations, std::vector<real> &results) override
    {
        // Compute padding
        int maxWidth = thrust::max_element(valuations.Widths.begin(), valuations.Widths.end())[0];
        int maxHeight = thrust::max_element(valuations.Heights.begin(), valuations.Heights.end())[0];
        int totalQsCount = valuations.ValuationCount * maxWidth;
        int totalAlphasCount = valuations.ValuationCount * maxHeight;
        KernelArgsValues values;

        runKernel<KernelArgsCoalesced>(valuations, results, totalQsCount, totalAlphasCount, values);
    }
};

}

}

#endif
