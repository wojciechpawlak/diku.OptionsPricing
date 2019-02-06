#ifndef CUDA_KERNEL_OPTION_CUH
#define CUDA_KERNEL_OPTION_CUH

#include "../cuda/CudaDomain.cuh"

#define DEFAULT_BLOCK_SIZE 256
#define PRINT_IDX 4
#define DEV

using namespace trinom;

namespace cuda
{

namespace option
{

struct KernelArgsValues
{
    real *ratesAll;
    real *pusAll;
    real *pmsAll;
    real *pdsAll;
    real *QsAll;        // intermediate results: forward propagation: evolution of the short-rate process, backward propagation: bond prices
    real *QsCopyAll;    // intermediate results: buffer, forward propagation: evolution of the short-rate process, backward propagation: bond prices
    real *alphasAll;    // intermediate results: analytical displacement of the short-rate process tree
    //real *OptPsAll;     // backward propagation: option prices
    //real *OptPsCopyAll; // buffer, backward propagation: option prices
    real *res;          // pricing result
};  

/**
Base class for kernel arguments.
Important! Don't call defined pure virtual functions within your implementation.
**/
template<class KernelArgsValuesT>
class KernelArgsBase
{

protected:
    KernelArgsValuesT values;

public:

    KernelArgsBase(KernelArgsValuesT &v) : values(v) { }
    
    __device__ inline int getIdx() const { return threadIdx.x + blockDim.x * blockIdx.x; }

    __device__ virtual void init(const KernelValuations &valuations) = 0;

    __device__ virtual void switchQs()
    {
        auto QsT = values.QsAll;
        values.QsAll = values.QsCopyAll;
        values.QsCopyAll = QsT;
    }

    //__device__ virtual void switchOptPs()
    //{
    //    auto OptPsT = values.OptPsAll;
    //    values.OptPsAll = values.OptPsCopyAll;
    //    values.OptPsCopyAll = OptPsT;
    //}

    __device__ virtual void fillQs(const int count, const int value) = 0;

    __device__ virtual real getRateAt(const int index) const = 0;

    __device__ virtual void setRateAt(const int index, const real value) = 0;

    __device__ virtual real getPAt(const int index, const int branch) = 0;

    __device__ virtual void setPAt(const int index, const int branch, const real value) = 0;

    __device__ virtual real getQAt(const int index) const = 0;

    __device__ virtual void setQAt(const int index, const real value) = 0;

    __device__ virtual void setQCopyAt(const int index, const real value) = 0;

    __device__ virtual real getAlphaAt(const int index) const = 0;

    __device__ virtual void setAlphaAt(const int index, const real value) = 0;

    //__device__ virtual real getOptPAt(const int index) const = 0;

    //__device__ virtual void setOptPAt(const int index, const real value) = 0;

    //__device__ virtual void setOptPCopyAt(const int index, const real value) = 0;

    __device__ virtual void setResult(const int jmax) = 0;
};

template<class KernelArgsT>
__global__ void kernelOneOptionPerThread(const KernelValuations valuations, KernelArgsT args)
{
    auto idx = args.getIdx();

    // Out of valuations check
    if (idx >= valuations.ValuationCount) return;

    ValuationConstants c;
    computeConstants(c, valuations, idx);
#ifdef DEV
    printf("%d: %d %d %d %d\n", idx, c.firstYCTermIdx, c.LastExerciseStep, c.FirstExerciseStep, c.ExerciseStepFrequency);
    printf("%d: %f %d %d\n", idx, valuations.YieldCurveRates[c.firstYCTermIdx], valuations.YieldCurveTimeSteps[c.firstYCTermIdx], valuations.YieldCurveTerms[valuations.YieldCurveIndices[idx]]);
#endif
    args.init(valuations);
    args.setQAt(c.jmax, one);
    int lastUsedYCTermIdx = 0;
    auto alpha = interpolateRateAtTimeStep(c.dt, c.termUnit, c.firstYieldCurveRate, c.firstYieldCurveTimeStep, c.yieldCurveTermCount, &lastUsedYCTermIdx);
    args.setAlphaAt(0, exp(-alpha * c.dt));

    //if (args.getIdx() == 2)
    //    printf("0 %d alpha %f alpha g %f alpha sh %f OptionIdx %d OptionInBlockIdx %d idxBlock %d idxBlockNext %d idx %d scannedWidthIdx %d threadIdx %d  Qexp %f dt %f termUnit %d n %d  optIdx %d\n",
    //        0, alpha, args.getAlphaAt(0), 0.0, args.getIdx(), 0, 0, 0, idx, 0, threadIdx.x, 0.0, c.dt, c.termUnit, c.n, 0);

    // Precompute exponent of rates for each node on the width on the tree (constant over forward propagation)
    for (auto j = -c.jmax; j <= c.jmax; ++j)
    {
        auto jind = j + c.jmax;
        args.setRateAt(jind, exp(-(real)j*c.dr*c.dt));
    }

    // Precompute probabilities for each node on the width on the tree (constant over forward and backward propagation)
    args.setPAt(0, 1, PU_B(-c.jmax, c.M)); args.setPAt(0, 2, PM_B(-c.jmax, c.M)); args.setPAt(0, 3, PD_B(-c.jmax, c.M));
#ifdef DEV1
    if (idx == PRINT_IDX) printf("%d: %d: %f %f %f\n", idx, 0, args.getPAt(0, 1), args.getPAt(0, 2), args.getPAt(0, 3));
#endif
    for (auto j = -c.jmax + 1; j < c.jmax; ++j)
    {
        auto jind = j + c.jmax;
        args.setPAt(jind, 1, PU_A(j, c.M)); args.setPAt(jind, 2, PM_A(j, c.M)); args.setPAt(jind, 3, PD_A(j, c.M));
#ifdef DEV1
        if (idx == PRINT_IDX) printf("%d: %d: %f %f %f\n", idx, jind, args.getPAt(jind, 1), args.getPAt(jind, 2), args.getPAt(jind, 3));
#endif
    }
    args.setPAt(c.width - 1, 1, PU_C(c.jmax, c.M)); args.setPAt(c.width - 1, 2, PM_C(c.jmax, c.M)); args.setPAt(c.width - 1, 3, PD_C(c.jmax, c.M));
#ifdef DEV1
    if (idx == PRINT_IDX) printf("%d: %d: %f %f %f\n", idx, c.width - 1, args.getPAt(c.width - 1, 1), args.getPAt(c.width - 1, 2), args.getPAt(c.width - 1, 3));
#endif

    // Forward propagation
    for (auto i = 1; i <= c.n; ++i)
    {
        const auto jhigh = min(i, c.jmax);

        //if (args.getIdx() == 2)
        //    printf("1 %d alpha %f alpha g %f alpha sh %f OptionIdx %d OptionInBlockIdx %d idxBlock %d idxBlockNext %d idx %d scannedWidthIdx %d threadIdx %d  Qexp %f dt %f termUnit %d n %d  optIdx %d\n",
        //        i - 1, alpha, args.getAlphaAt(i - 1), 0.0, args.getIdx(), 0, 0, 0, idx, 0, threadIdx.x, 0.0, c.dt, c.termUnit, c.n, 0);
        
        // Precompute Qexp
        const auto expmAlphadt = args.getAlphaAt(i - 1);
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            const auto jind = j + c.jmax;      // array index for j
            real Qexp = args.getQAt(jind) * expmAlphadt * args.getRateAt(jind);
            args.setQAt(jind, Qexp);
        }
        // Forward iteration step, compute Qs in the next time step
        real aggregatedQs = zero;
        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            const auto jind = j + c.jmax;      // array index for j            

            const auto expu1 = j == jhigh ? zero : args.getQAt(jind + 1);
            const auto expm = args.getQAt(jind);
            const auto expd1 = j == -jhigh ? zero : args.getQAt(jind - 1);

            real Q;

            if (i == 1)
            {
                if (j == -jhigh) {
                    Q = args.getPAt(jind + 1, 3) * expu1;
                } else if (j == jhigh) {
                    Q = args.getPAt(jind - 1, 1) * expd1;
                } else {
                    Q = args.getPAt(jind, 2) * expm;
                }
            } 
            else if (i <= c.jmax)
            {
                if (j == -jhigh) {
                    Q = args.getPAt(jind + 1, 3) * expu1;
                } else if (j == -jhigh + 1) {
                    Q = args.getPAt(jind, 2) * expm +
                        args.getPAt(jind + 1, 3) * expu1;
                } else if (j == jhigh) {
                    Q = args.getPAt(jind - 1, 1) * expd1;
                } else if (j == jhigh - 1) {
                    Q = args.getPAt(jind - 1, 1) * expd1 +
                        args.getPAt(jind, 2) * expm;
                } else {
                    Q = args.getPAt(jind - 1, 1) * expd1 +
                        args.getPAt(jind, 2) * expm +
                        args.getPAt(jind + 1, 3) * expu1;
                }
            }
            else
            {
                if (j == -jhigh) {
                    Q = args.getPAt(jind, 3) * expm +
                        args.getPAt(jind + 1, 3) * expu1;
                } else if (j == -jhigh + 1) {
                    Q = args.getPAt(jind - 1, 2) * expd1 +
                        args.getPAt(jind, 2) * expm +
                        args.getPAt(jind + 1, 3) * expu1;
                            
                } else if (j == jhigh) {
                    Q = args.getPAt(jind - 1, 1) * expd1 +
                        args.getPAt(jind, 1) * expm;
                } else if (j == jhigh - 1) {
                    Q = args.getPAt(jind - 1, 1) * expd1 +
                        args.getPAt(jind, 2) * expm +
                        args.getPAt(jind + 1, 2) * expu1;
                } else {
                    Q = ((j == -jhigh + 2) ? args.getPAt(jind - 2,  1) * args.getQAt(jind - 2) : zero) +
                        args.getPAt(jind - 1, 1) * expd1 +
                        args.getPAt(jind, 2) * expm +
                        args.getPAt(jind + 1, 3) * expu1 +
                        ((j == jhigh - 2) ? args.getPAt(jind + 2, 3) * args.getQAt(jind + 2) : zero);
                }
            }
            // Determine the new alpha using equation 30.22
            // by summing up Qs from the next time step
            args.setQCopyAt(jind, Q); 
            aggregatedQs += Q * args.getRateAt(jind);
        }

        alpha = computeAlpha(aggregatedQs, i - 1, c.dt, c.termUnit, c.firstYieldCurveRate, c.firstYieldCurveTimeStep, c.yieldCurveTermCount, &lastUsedYCTermIdx);
        args.setAlphaAt(i, exp(-alpha * c.dt));
#ifdef DEV
        if (idx == PRINT_IDX) printf("%d: %.18f %.18f\n", i, alpha, args.getAlphaAt(i));
#endif
        //if (args.getIdx() == 2)
        //    printf("2 %d alpha %f alpha g %f alpha sh %f OptionIdx %d OptionInBlockIdx %d idxBlock %d idxBlockNext %d idx %d scannedWidthIdx %d threadIdx %d  Qexp %f dt %f termUnit %d n %d  optIdx %d\n",
        //        i, alpha2, args.getAlphaAt(i), 0.0, args.getIdx(), 0, 0, 0, idx, 0, threadIdx.x, 0.0, c.dt, c.termUnit, c.n, 0);

        // Switch Qs
        args.switchQs();
    }
    
    // Backward propagation
    // TODO Find out how to handle oas spread
    auto lastUsedCIdx = valuations.CashflowIndices[idx] + valuations.Cashflows[idx] - 1;
    printf("%d: %d %f %f %d\n", idx, lastUsedCIdx, valuations.Repayments[lastUsedCIdx], valuations.Coupons[lastUsedCIdx], valuations.CashflowSteps[lastUsedCIdx]);
    args.fillQs(c.width, valuations.Repayments[lastUsedCIdx] + valuations.Coupons[lastUsedCIdx]); // initialize to par/face value: last repayment + last coupon
    auto lastUsedCStep = valuations.CashflowSteps[--lastUsedCIdx];

    for (auto i = c.n - 1; i >= 0; --i)
    {
        const auto jhigh = min(i, c.jmax);
        const auto expmAlphadt = args.getAlphaAt(i);

        //if (args.getIdx() == 2)
        //    printf("3 %d alpha %f alpha g %f alpha sh %f OptionIdx %d OptionInBlockIdx %d idxBlock %d idxBlockNext %d idx %d scannedWidthIdx %d threadIdx %d  Qexp %f dt %f termUnit %d n %d  optIdx %d\n",
        //        i, alpha, args.getAlphaAt(i), 0.0, args.getIdx(), 0, 0, 0, idx, 0, threadIdx.x, 0.0, c.dt, c.termUnit, c.n, 0);
        
        // check if there is an option exercise at the current step
        const auto isExerciseStep = i <= c.LastExerciseStep && i >= c.FirstExerciseStep && (lastUsedCStep - i) % c.ExerciseStepFrequency == 0;
#ifdef DEV
        if (idx == PRINT_IDX) printf("%d: %d %d %d %d\n", i, isExerciseStep, lastUsedCStep, (lastUsedCStep - i) % c.ExerciseStepFrequency, (lastUsedCStep - i) % c.ExerciseStepFrequency == 0);
#endif
        // add coupon and repayments
        if (i == lastUsedCStep - 1)
        {
#ifdef DEV
            if (idx == PRINT_IDX) printf("%d: %d coupon: %.18f\n", i, lastUsedCIdx, args.getQAt(c.jmax));
#endif
            for (auto j = -jhigh; j <= jhigh; ++j)
            {
                const auto jind = j + c.jmax;      // array index for j
                args.setQAt(jind, args.getQAt(jind) + valuations.Repayments[lastUsedCIdx] + valuations.Coupons[lastUsedCIdx]);
            }
#ifdef DEV
            if (idx == PRINT_IDX) printf("%d: %d coupon: %.18f\n", i, lastUsedCIdx, args.getQAt(c.jmax));
#endif
            lastUsedCStep = (--lastUsedCIdx >= 0) ? valuations.CashflowSteps[lastUsedCIdx] : 0;
        }

        // calculate accrued interest from cashflow
        const auto ai = isExerciseStep && lastUsedCStep != 0 ? computeAccruedInterest(c.termStepCount, i, lastUsedCStep, valuations.CashflowSteps[lastUsedCIdx + 1], valuations.Coupons[lastUsedCIdx]) : zero;
#ifdef DEV
        if (idx == PRINT_IDX && isExerciseStep) 
            printf("%d: ai %f %d %d %d %f %d %d %f\n", i, ai, c.termStepCount, lastUsedCStep, valuations.CashflowSteps[lastUsedCIdx + 1], valuations.Coupons[lastUsedCIdx],
                valuations.CashflowSteps[lastUsedCIdx + 1] - lastUsedCStep, valuations.CashflowSteps[lastUsedCIdx + 1] - i, 
                (real)(valuations.CashflowSteps[lastUsedCIdx + 1] - lastUsedCStep - valuations.CashflowSteps[lastUsedCIdx + 1] - i) / (valuations.CashflowSteps[lastUsedCIdx + 1] - lastUsedCStep));
#endif

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            const auto jind = j + c.jmax;      // array index for j
            const auto discFactor = expmAlphadt * args.getRateAt(jind) * c.expmOasdt;

            real res;
            if (j == c.jmax)
            {
                // Top edge branching
                res = (args.getPAt(jind, 1) * args.getQAt(jind) +
                    args.getPAt(jind, 2) * args.getQAt(jind - 1) +
                    args.getPAt(jind, 3) * args.getQAt(jind - 2)) *
                        discFactor;
            }
            else if (j == -c.jmax)
            {  
                res = (args.getPAt(jind, 1) * args.getQAt(jind + 2) +
                    args.getPAt(jind, 2) * args.getQAt(jind + 1) +
                    args.getPAt(jind, 3) * args.getQAt(jind)) *
                        discFactor;
            }
            else
            {
                // Standard branching
                res = (args.getPAt(jind, 1) * args.getQAt(jind + 1) +
                    args.getPAt(jind, 2) * args.getQAt(jind) +
                    args.getPAt(jind, 3) * args.getQAt(jind - 1)) *
                        discFactor;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            args.setQCopyAt(jind, getOptionPayoff(isExerciseStep, c.X, c.type, res, ai));
        }

        // Switch Qs
        args.switchQs();
#ifdef DEV
        if (idx == PRINT_IDX) printf("%d: %.18f\n", i, args.getQAt(c.jmax));
#endif
    }

    args.setResult(c.jmax);
    //if (args.getOptionIdx() == 2) printf("res: %f\n", args.getQAt(c.jmax));
}

class KernelRunBase
{

private:
    std::chrono::time_point<std::chrono::steady_clock> TimeBegin;
    bool IsTest;

protected:
    int BlockSize;

    virtual void runPreprocessing(CudaValuations &valuations, std::vector<real> &results) = 0;

    template<class KernelArgsT, class KernelArgsValuesT>
    void runKernel(CudaValuations &valuations, std::vector<real> &results, const int totalQsCount, const int totalAlphasCount, KernelArgsValuesT &values)
    {
        thrust::device_vector<real> rates(totalQsCount);
        thrust::device_vector<real> pus(totalQsCount);
        thrust::device_vector<real> pms(totalQsCount);
        thrust::device_vector<real> pds(totalQsCount);
        thrust::device_vector<real> Qs(totalQsCount);
        thrust::device_vector<real> QsCopy(totalQsCount);
        thrust::device_vector<real> alphas(totalAlphasCount);
        thrust::device_vector<real> result(valuations.ValuationCount);

        const auto blockCount = (int)ceil(valuations.ValuationCount / ((real)BlockSize));

        valuations.DeviceMemory += vectorsizeof(rates);
        valuations.DeviceMemory += vectorsizeof(pus);
        valuations.DeviceMemory += vectorsizeof(pms);
        valuations.DeviceMemory += vectorsizeof(pds);
        valuations.DeviceMemory += vectorsizeof(Qs);
        valuations.DeviceMemory += vectorsizeof(QsCopy);
        valuations.DeviceMemory += vectorsizeof(alphas);
        valuations.DeviceMemory += vectorsizeof(result);
        runtime.DeviceMemory = valuations.DeviceMemory;

        if (IsTest)
        {
            std::cout << "Running " << valuations.ValuationCount << 
            #ifdef USE_DOUBLE
            " double"
            #else
            " float"
            #endif
            << " valuations with block size " << BlockSize << std::endl;
            std::cout << "Qs count " << totalQsCount << ", alphas count " << totalAlphasCount << std::endl;
            std::cout << "Global memory size " << valuations.DeviceMemory / (1024.0 * 1024.0) << " MB" << std::endl;

            cudaDeviceSynchronize();
            size_t memoryFree, memoryTotal;
            cudaMemGetInfo(&memoryFree, &memoryTotal);
            std::cout << "Current GPU memory usage " << (memoryTotal - memoryFree) / (1024.0 * 1024.0) << " MB out of " << memoryTotal / (1024.0 * 1024.0) << " MB " << std::endl;
        }

        values.ratesAll = thrust::raw_pointer_cast(rates.data());
        values.pusAll = thrust::raw_pointer_cast(pus.data());
        values.pmsAll = thrust::raw_pointer_cast(pms.data());
        values.pdsAll = thrust::raw_pointer_cast(pds.data());
        values.QsAll = thrust::raw_pointer_cast(Qs.data());
        values.QsCopyAll = thrust::raw_pointer_cast(QsCopy.data());
        values.alphasAll = thrust::raw_pointer_cast(alphas.data());
        values.res = thrust::raw_pointer_cast(result.data());
        KernelArgsT kernelArgs(values);

        auto time_begin_kernel = std::chrono::steady_clock::now();
        kernelOneOptionPerThread<<<blockCount, BlockSize>>>(valuations.KernelValuations, kernelArgs);
        cudaDeviceSynchronize();
        auto time_end_kernel = std::chrono::steady_clock::now();
        runtime.KernelRuntime = std::chrono::duration_cast<std::chrono::microseconds>(time_end_kernel - time_begin_kernel).count();

        CudaCheckError();

        if (IsTest)
        {
            std::cout << "Kernel executed in " << runtime.KernelRuntime << " microsec" << std::endl;
        }

        // Sort result
        valuations.sortResult(result);

        // Stop timing
        auto timeEnd = std::chrono::steady_clock::now();
        runtime.TotalRuntime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - TimeBegin).count();

        if (IsTest)
        {
            std::cout << "Total execution time " << runtime.TotalRuntime << " microsec" << std::endl;
        }

        // Copy result to host
        thrust::copy(result.begin(), result.end(), results.begin());
        cudaDeviceSynchronize();
    }

public:
    CudaRuntime runtime;
    
    void run(
        const Valuations &valuations,
        std::vector<real> &results,
        const int blockSize = -1,
        const SortType sortType = SortType::NONE,
        const bool isTest = false)
    {
        CudaValuations cudaValuations(valuations);

        // Start timing when input is copied to device
        cudaDeviceSynchronize();
        auto timeBegin = std::chrono::steady_clock::now();

        cudaValuations.initialize();
        run(cudaValuations, results, timeBegin, blockSize, sortType, isTest);
    }

    void run(
        CudaValuations &cudaValuations,
        std::vector<real> &results,
        const std::chrono::time_point<std::chrono::steady_clock> timeBegin,
        const int blockSize = -1,
        const SortType sortType = SortType::NONE,
        const bool isTest = false)
    {
        TimeBegin = timeBegin;
        IsTest = isTest;
        BlockSize = blockSize > 0 ? blockSize : DEFAULT_BLOCK_SIZE;

        cudaValuations.sortValuations(sortType, isTest);
        runPreprocessing(cudaValuations, results);
    }

};

}

}

#endif