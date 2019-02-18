#ifndef CUDA_KERNEL_OPTION_CUH
#define CUDA_KERNEL_OPTION_CUH

#include "../cuda/CudaDomain.cuh"

#define DEFAULT_BLOCK_SIZE 256

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
#ifdef DEV_LIMIT
    if (idx != 0 && idx != 1) return;
#endif

    ValuationConstants c;
    computeConstants(c, valuations, idx);
    // Helper variables
    volatile uint16_t lastUsedYCTermIdx = 0;
    const auto sortedIdx = valuations.ValuationIndices[0] != -1 ? valuations.ValuationIndices[idx] : idx;
    auto lastUsedCIdx = valuations.CashflowIndices[sortedIdx] + (int)valuations.Cashflows[sortedIdx] - 1;
    auto lastCashflow = valuations.Repayments[lastUsedCIdx] + valuations.Coupons[lastUsedCIdx];
    auto cashflowsRemaining = valuations.Cashflows[sortedIdx];
    auto lastCStep = valuations.CashflowSteps[lastUsedCIdx];
    auto alpha = 0.0;
#ifdef DEV
    const auto ycIndex = valuations.YieldCurveIndices[sortedIdx];
    const auto firstYCTermIndex = valuations.YieldCurveTermIndices[ycIndex];
    if (idx == PRINT_IDX)
        printf("%d %d %d: Input %d %.18f %d %.18f %d %.18f %.18f %d %.18f %d %d %d %d %.18f %d %d %d %.18f %d %d %d %d %.18f %d %d\n", idx, threadIdx.x, blockIdx.x,
        c.termUnit, c.dt, c.n, c.X, c.type, c.M, c.mdrdt, c.jmax, c.expmOasdt, c.lastExerciseStep, c.firstExerciseStep, c.exerciseStepFrequency,
        c.yieldCurveTermCount, *c.firstYieldCurveRate, *c.firstYieldCurveTimeStep, lastUsedYCTermIdx,
        lastUsedCIdx, lastCashflow, lastCStep, cashflowsRemaining,
        ycIndex, firstYCTermIndex, valuations.YieldCurveRates[firstYCTermIndex], valuations.YieldCurveTimeSteps[firstYCTermIndex], sortedIdx);
#endif
#ifdef DEV_EXTRA
    // problem with exponent function on single precision
    // exp function on GPU yields slightly different result than exp on CPU
    // the difference propagates to all the other computations 
    // as all subsequent computations depend on this calculation
    const auto a = valuations.MeanReversionRates[idx];
    const auto sigma = valuations.Volatilities[idx];
    const double tmp = -two * a * c.dt;
    const auto exp_tmp = exp(tmp);
    if (idx == PRINT_IDX) printf("%d: %d %.18f %.18f %.18f %.18f %.18f %.18f %d %d %d %d\n",
        idx, c.n, a, sigma, tmp, exp_tmp, sigma * sigma * (one - exp_tmp) / (two * a), c.dt, c.firstYCTermIdx, c.lastExerciseStep, c.firstExerciseStep, c.exerciseStepFrequency);
    if (idx == PRINT_IDX) printf("%d: %.18f %d %d\n", idx, valuations.YieldCurveRates[c.firstYCTermIdx], valuations.YieldCurveTimeSteps[c.firstYCTermIdx], valuations.YieldCurveTerms[valuations.YieldCurveIndices[idx]]);
#endif
    args.init(valuations);

    // --------------------------------------------------

    // Precompute exponent of rates for each node on the width on the tree (constant over forward propagation)
    for (auto j = -c.jmax; j <= c.jmax; ++j)
    {
        auto jind = j + c.jmax;
        args.setRateAt(jind, exp(c.mdrdt*j));
    }

    // Precompute probabilities for each node on the width on the tree (constant over forward and backward propagation)
    args.setPAt(0, 1, PU_B(-c.jmax, c.M)); args.setPAt(0, 2, PM_B(-c.jmax, c.M)); args.setPAt(0, 3, PD_B(-c.jmax, c.M));
#ifdef DEV0
    if (idx == PRINT_IDX) printf("%d: %d: %.18f %.18f %.18f %.18f\n", idx, 0, args.getRateAt(0), args.getPAt(0, 1), args.getPAt(0, 2), args.getPAt(0, 3));
#endif
    for (auto j = -c.jmax + 1; j < c.jmax; ++j)
    {
        auto jind = j + c.jmax;
        args.setPAt(jind, 1, PU_A(j, c.M)); args.setPAt(jind, 2, PM_A(j, c.M)); args.setPAt(jind, 3, PD_A(j, c.M));
#ifdef DEV0
        if (idx == PRINT_IDX) printf("%d: %d: %.18f %.18f %.18f %.18f\n", idx, jind, args.getRateAt(jind), args.getPAt(jind, 1), args.getPAt(jind, 2), args.getPAt(jind, 3));
#endif
    }
    args.setPAt(c.width - 1, 1, PU_C(c.jmax, c.M)); args.setPAt(c.width - 1, 2, PM_C(c.jmax, c.M)); args.setPAt(c.width - 1, 3, PD_C(c.jmax, c.M));
#ifdef DEV0
    if (idx == PRINT_IDX) printf("%d: %d: %.18f %.18f %.18f %.18f\n", idx, c.width - 1, args.getRateAt(c.width - 1), args.getPAt(c.width - 1, 1), args.getPAt(c.width - 1, 2), args.getPAt(c.width - 1, 3));
#endif

    // Forward propagation
    args.setQAt(c.jmax, one); // Initialize the root of the tree
    alpha = interpolateRateAtTimeStep(c.dt, c.termUnit, c.firstYieldCurveRate, c.firstYieldCurveTimeStep, c.yieldCurveTermCount, &lastUsedYCTermIdx);
    args.setAlphaAt(0, exp(-alpha * c.dt));

#ifdef DEV1
    if (idx == PRINT_IDX)
        printf("%d %d: %.18f %.18f %.18f %d\n", idx, 0, 1.0, alpha, args.getAlphaAt(0), lastUsedYCTermIdx);
#endif
#ifdef DEV1
    if (idx == PRINT_IDX && 0 >= PRINT_FIRST_ITER && 0 <= PRINT_LAST_ITER)
    {
        printf("%d %d: ", idx, 0);
        for (auto k = 0; k < c.width; ++k)
        {
            printf("%d: %.18f ", k, args.getQAt(k));
        }
        printf("\n");
    }
#endif

    for (auto i = 1; i <= c.n; ++i)
    {
        const auto jhigh = min(i, c.jmax);

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

            const auto expu = j == jhigh ? zero : args.getQAt(jind + 1);
            const auto expm = args.getQAt(jind);
            const auto expd = j == -jhigh ? zero : args.getQAt(jind - 1);

            real Q;
            if (i == 1)
            {
                if (j == -jhigh) {
                    Q = args.getPAt(jind + 1, 3) * expu;
                }
                else if (j == jhigh) {
                    Q = args.getPAt(jind - 1, 1) * expd;
                }
                else {
                    Q = args.getPAt(jind, 2) * expm;
                }
            }
            else if (i <= c.jmax)
            {
                if (j == -jhigh) {
                    Q = args.getPAt(jind + 1, 3) * expu;
                }
                else if (j == -jhigh + 1) {
                    Q = args.getPAt(jind, 2) * expm +
                        args.getPAt(jind + 1, 3) * expu;
                }
                else if (j == jhigh) {
                    Q = args.getPAt(jind - 1, 1) * expd;
                }
                else if (j == jhigh - 1) {
                    Q = args.getPAt(jind - 1, 1) * expd +
                        args.getPAt(jind, 2) * expm;
                }
                else {
                    Q = args.getPAt(jind - 1, 1) * expd +
                        args.getPAt(jind, 2) * expm +
                        args.getPAt(jind + 1, 3) * expu;
                }
            }
            else
            {
                if (j == -jhigh) {
                    Q = args.getPAt(jind, 3) * expm +
                        args.getPAt(jind + 1, 3) * expu;
                }
                else if (j == -jhigh + 1) {
                    Q = args.getPAt(jind - 1, 2) * expd +
                        args.getPAt(jind, 2) * expm +
                        args.getPAt(jind + 1, 3) * expu;

                }
                else if (j == jhigh) {
                    Q = args.getPAt(jind - 1, 1) * expd +
                        args.getPAt(jind, 1) * expm;
                }
                else if (j == jhigh - 1) {
                    Q = args.getPAt(jind - 1, 1) * expd +
                        args.getPAt(jind, 2) * expm +
                        args.getPAt(jind + 1, 2) * expu;
                }
                else {
                    Q = ((j == -jhigh + 2) ? args.getPAt(jind - 2, 1) * args.getQAt(jind - 2) : zero) +
                        args.getPAt(jind - 1, 1) * expd +
                        args.getPAt(jind, 2) * expm +
                        args.getPAt(jind + 1, 3) * expu +
                        ((j == jhigh - 2) ? args.getPAt(jind + 2, 3) * args.getQAt(jind + 2) : zero);
                }
            }
            // Determine the new alpha using equation 30.22
            // by summing up Qs from the next time step
            args.setQCopyAt(jind, Q);
            aggregatedQs += Q * args.getRateAt(jind);
            //#ifdef DEV1
            //            if (idx == PRINT_IDX && i == 1) printf("%d %d: %.18f %.18f %.18f\n", idx, jind, aggregatedQs, Q, args.getRateAt(jind));
            //#endif
        }

        alpha = computeAlpha(aggregatedQs, i - 1, c.dt, c.termUnit, c.firstYieldCurveRate, c.firstYieldCurveTimeStep, c.yieldCurveTermCount, &lastUsedYCTermIdx);
        args.setAlphaAt(i, exp(-alpha * c.dt));
#ifdef DEV1
        if (idx == PRINT_IDX) printf("%d %d: %.18f %.18f %.18f %d\n", idx, i, aggregatedQs, alpha, args.getAlphaAt(i), lastUsedYCTermIdx);
#endif

        // Switch Qs
        args.switchQs();

#ifdef DEV1
        if (idx == PRINT_IDX && i > PRINT_FIRST_ITER && i < PRINT_LAST_ITER)
        {
            printf("%d %d: ", idx, i);
            for (auto k = 0; k < c.width; ++k)
            {
                printf("%d: %.18f ", k, args.getQAt(k));
            }
            printf("\n");
        }
#endif
    }

    // Backward propagation
#ifdef DEV2
    if (idx == PRINT_IDX) printf("%d %d: %d %d %.18f %.18f %.18f %d\n", idx, c.n, lastUsedCIdx, cashflowsRemaining, lastCashflow, valuations.Repayments[lastUsedCIdx], valuations.Coupons[lastUsedCIdx], valuations.CashflowSteps[lastUsedCIdx]);
#endif
    args.fillQs(c.width, lastCashflow); // initialize to par/face value: last repayment + last coupon
    cashflowsRemaining--;
    if (lastUsedCIdx > 0 && lastCStep <= c.n && cashflowsRemaining > 0)
    {
        lastUsedCIdx--;
        lastCStep = valuations.CashflowSteps[lastUsedCIdx];
    }
    
    for (auto i = c.n - 1; i >= 0; --i)
    {
        const auto jhigh = min(i, c.jmax);
        const auto expmAlphadt = args.getAlphaAt(i);

        // check if there is an option exercise at the current step
        const auto isExerciseStep = i <= c.lastExerciseStep && i >= c.firstExerciseStep && (lastCStep - i) % c.exerciseStepFrequency == 0;
#ifdef DEV2
        if (idx == PRINT_IDX)
            printf("%d %d: %.18f %d %d %d %d\n", idx, i, expmAlphadt, isExerciseStep, lastCStep, (lastCStep - i) % c.exerciseStepFrequency, (lastCStep - i) % c.exerciseStepFrequency == 0);
#endif
        // add coupon and repayment if crossed a time step with a cashflow
        if (i == lastCStep - 1 && cashflowsRemaining > 0)
        {
            lastCashflow = valuations.Repayments[lastUsedCIdx] + valuations.Coupons[lastUsedCIdx];
#ifdef DEV2
            if (idx == PRINT_IDX) printf("%d %d: %d %d coupon: %.18f %.18f\n", idx, i, lastUsedCIdx, cashflowsRemaining, args.getQAt(c.jmax), lastCashflow);
#endif
            for (auto j = -jhigh; j <= jhigh; ++j)
            {
                const auto jind = j + c.jmax;      // array index for j
                args.setQAt(jind, args.getQAt(jind) + lastCashflow);
            }
#ifdef DEV2
            if (idx == PRINT_IDX) printf("%d %d: %d %d coupon: %.18f\n", idx, i, lastUsedCIdx, cashflowsRemaining, args.getQAt(c.jmax));
#endif
            cashflowsRemaining--;
            if (lastUsedCIdx > 0 && lastCStep <= c.n && cashflowsRemaining > 0)
            {
                lastUsedCIdx--;
                lastCStep = valuations.CashflowSteps[lastUsedCIdx];
            }
        }

        // calculate accrued interest from last cashflow
        const auto ai = isExerciseStep && lastCStep != 0 && cashflowsRemaining > 0 ? computeAccruedInterest(i, lastCStep, valuations.CashflowSteps[lastUsedCIdx + 1], valuations.Coupons[lastUsedCIdx]) : zero;
#ifdef DEV2
        if (idx == PRINT_IDX && isExerciseStep && lastCStep != 0 && cashflowsRemaining > 0)
            printf("%d %d: ai %.18f %d %d %.18f %d %d %.18f\n", idx, i, ai, lastCStep, valuations.CashflowSteps[lastUsedCIdx + 1], valuations.Coupons[lastUsedCIdx],
                valuations.CashflowSteps[lastUsedCIdx + 1] - lastCStep, valuations.CashflowSteps[lastUsedCIdx + 1] - i,
                (real)(valuations.CashflowSteps[lastUsedCIdx + 1] - lastCStep - valuations.CashflowSteps[lastUsedCIdx + 1] - i) / (valuations.CashflowSteps[lastUsedCIdx + 1] - lastCStep));
#endif

        for (auto j = -jhigh; j <= jhigh; ++j)
        {
            const auto jind = j + c.jmax;      // array index for j
            const auto discFactor = expmAlphadt * args.getRateAt(jind) * c.expmOasdt;

            real currentStepPrice;
            if (j == c.jmax)
            {
                // Top edge branching
                currentStepPrice = (args.getPAt(jind, 1) * args.getQAt(jind) +
                    args.getPAt(jind, 2) * args.getQAt(jind - 1) +
                    args.getPAt(jind, 3) * args.getQAt(jind - 2)) *
                    discFactor;
            }
            else if (j == -c.jmax)
            {
                // Bottom edge branching
                currentStepPrice = (args.getPAt(jind, 1) * args.getQAt(jind + 2) +
                    args.getPAt(jind, 2) * args.getQAt(jind + 1) +
                    args.getPAt(jind, 3) * args.getQAt(jind)) *
                    discFactor;
            }
            else
            {
                // Standard branching
                currentStepPrice = (args.getPAt(jind, 1) * args.getQAt(jind + 1) +
                    args.getPAt(jind, 2) * args.getQAt(jind) +
                    args.getPAt(jind, 3) * args.getQAt(jind - 1)) *
                    discFactor;
            }

            // after obtaining the result from (i+1) nodes, set the call for ith node
            args.setQCopyAt(jind, getOptionPayoff(isExerciseStep, c.X, c.type, currentStepPrice, ai));
        }

        // Switch Qs
        args.switchQs();
#ifdef DEV2
        if (idx == PRINT_IDX) printf("%d %d: %.18f\n", idx, i, args.getQAt(c.jmax));
#endif
    }

    args.setResult(c.jmax);
#ifdef DEV
    if (idx == PRINT_IDX) printf("%d: res %.18f\n", idx, args.getQAt(c.jmax));
#endif
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
        kernelOneOptionPerThread << <blockCount, BlockSize >> > (valuations.KernelValuations, kernelArgs);
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