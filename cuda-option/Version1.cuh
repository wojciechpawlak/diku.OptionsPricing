#ifndef CUDA_VERSION_1_CUH
#define CUDA_VERSION_1_CUH

#include "Kernel.cuh"
#include "../cuda/CudaDomain.cuh"

namespace cuda
{

namespace option
{

class KernelArgsNaive : public KernelArgsBase<KernelArgsValues>
{
private:
    real *rates;
    real *pus;
    real *pms;
    real *pds;
    real *Qs;
    real *QsCopy;
    //real *OptPs;
    //real *OptPsCopy;
    real *alphas;

public:

    KernelArgsNaive(KernelArgsValues &v) : KernelArgsBase(v) { }

    __device__ inline void init(const KernelValuations &valuations) override
    {
        auto idx = getIdx();
        auto QsInd = idx == 0 ? 0 : valuations.Widths[idx - 1];
        auto alphasInd = idx == 0 ? 0 : valuations.Heights[idx - 1];

        rates = values.ratesAll + QsInd;
        pus = values.pusAll + QsInd;
        pms = values.pmsAll + QsInd;
        pds = values.pdsAll + QsInd;
        Qs = values.QsAll + QsInd;
        QsCopy = values.QsCopyAll + QsInd;
        alphas = values.alphasAll + alphasInd;
        //OptPs = values.OptPsAll + QsInd;
        //OptPsCopy = values.OptPsCopyAll + QsInd;
    }

    __device__ inline void switchQs()
    {
        auto QsT = Qs;
        Qs = QsCopy;
        QsCopy = QsT;
    }

    //__device__ inline void switchOptPs()
    //{
    //    auto OptPsT = OptPs;
    //    OptPs = OptPsCopy;
    //    OptPsCopy = OptPsT;
    //}

    __device__ void fillQs(const int count, const int value) override
    {
        for (auto i = 0; i < count; ++i)
        {
            Qs[i] = value;
        }
    }

    __device__ inline real getRateAt(const int index) const override { return rates[index]; }

    __device__ inline void setRateAt(const int index, const real value) override { rates[index] = value; }

    __device__ inline real getPAt(const int index, const int branch) override
    {
        switch (branch)
        {
        case 1:
            return pus[index]; // up
        case 2:
            return pms[index]; // mid
        case 3:
            return pds[index]; // down
        }
        return zero;
    }

    __device__ inline void setPAt(const int index, const int branch, const real value) override
    {
        switch (branch)
        {
        case 1:
            pus[index] = value; // up
        case 2:
            pms[index] = value; // mid
        case 3:
            pds[index] = value; // down
        }
    }

    __device__ inline real getQAt(const int index) const override { return Qs[index]; }

    __device__ inline void setQAt(const int index, const real value) override { Qs[index] = value; }

    __device__ inline void setQCopyAt(const int index, const real value) override { QsCopy[index] = value; }

    __device__ inline real getAlphaAt(const int index) const override { return alphas[index]; }

    __device__ inline void setAlphaAt(const int index, const real value) override { alphas[index] = value; }

    //__device__ inline real getOptPAt(const int index) const override { return OptPs[index]; }

    //__device__ inline void setOptPAt(const int index, const real value) override { OptPs[index] = value; }

    //__device__ inline void setOptPCopyAt(const int index, const real value) override { OptPsCopy[index] = value; }

    __device__ inline void setResult(const int jmax) override { values.res[getIdx()] = Qs[jmax]; }
};

class KernelRunNaive : public KernelRunBase
{

protected:
    void runPreprocessing(CudaValuations &valuations, std::vector<real> &results) override
    {
        // Compute indices.
        thrust::inclusive_scan(valuations.Widths.begin(), valuations.Widths.end(), valuations.Widths.begin());
        thrust::inclusive_scan(valuations.Heights.begin(), valuations.Heights.end(), valuations.Heights.begin());

        // Allocate temporary vectors.
        const int totalQsCount = valuations.Widths[valuations.ValuationCount - 1];
        const int totalAlphasCount = valuations.Heights[valuations.ValuationCount - 1];
        KernelArgsValues values;

        runKernel<KernelArgsNaive>(valuations, results, totalQsCount, totalAlphasCount, values);
    }
};

}

}

#endif
