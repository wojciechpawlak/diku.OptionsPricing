#ifndef CUDA_KERNEL_MULTI_CUH
#define CUDA_KERNEL_MULTI_CUH

#include <sstream>

#include "../cuda/CudaDomain.cuh"
#include "../cuda/ScanKernels.cuh"

using namespace trinom;

namespace cuda
{

namespace multi
{

volatile extern __shared__ char sh_mem[];

/**
Base class for kernel arguments.
Important! Don't call defined pure virtual functions within your implementation.
**/
template<class KernelArgsValuesT>
class KernelArgsBase
{

public:

    KernelArgsValuesT values;

    KernelArgsBase(KernelArgsValuesT &v) : values(v) { }

    __device__ virtual void init(const int valuationIdxBlock, const int idxBlock, const int idxBlockNext, const int valuationCount) = 0;

    __device__ virtual void setAlphaAt(const int index, const real value, const int valuationIndex = 0) = 0;

    __device__ virtual real getAlphaAt(const int index, const int valuationIndex = 0) const = 0;

    __device__ virtual int getMaxHeight() const = 0;

    __device__ virtual int getValuationIdx() const = 0;

    // 8-byte size variables, aligned to 8-byte (64-bit) border

    __device__ inline volatile real* getQs()
    {
        return (real *)&sh_mem;
    }

    __device__ inline volatile int32_t* getValuationInds()
    {
        return (int32_t *)&sh_mem;  // Sharing the same array with Qs!
    }

    __device__ inline volatile real* getRates()
    {
        return (real *)&sh_mem[blockDim.x * (sizeof(real))];
    }

    __device__ inline volatile real* getPus()
    {
        return (real *)&sh_mem[blockDim.x * (2 * sizeof(real))];
    }

    __device__ inline volatile real* getPms()
    {
        return (real *)&sh_mem[blockDim.x * (3 * sizeof(real))];
    }

    __device__ inline volatile real* getPds()
    {
        return (real *)&sh_mem[blockDim.x * (4 * sizeof(real))];
    }

    __device__ inline volatile real* getAlphas()
    {
        return (real *)&sh_mem[blockDim.x * (5 * sizeof(real))];
    }

    __device__ inline volatile real* getQexps()
    {
        return (real *)&sh_mem[blockDim.x * (5 * sizeof(real)) + values.maxValuationsBlock * sizeof(real)];
    }

    __device__ inline volatile real* getDts()
    {
        return (real *)&sh_mem[blockDim.x * (5 * sizeof(real)) + values.maxValuationsBlock * 2 * sizeof(real)];
    }

    __device__ inline volatile real* getMdrdts()
    {
        return (real *)&sh_mem[blockDim.x * (5 * sizeof(real)) + values.maxValuationsBlock * 3 * sizeof(real)];
    }

    __device__ inline volatile real* getMs()
    {
        return (real *)&sh_mem[blockDim.x * (5 * sizeof(real)) + values.maxValuationsBlock * 4 * sizeof(real)];
    }

    __device__ inline volatile real* getExpmOasdts()
    {
        return (real *)&sh_mem[blockDim.x * (5 * sizeof(real)) + values.maxValuationsBlock * 5 * sizeof(real)];
    }

    __device__ inline volatile real* getLastCashflows()
    {
        return (real *)&sh_mem[blockDim.x * (5 * sizeof(real)) + values.maxValuationsBlock * 6 * sizeof(real)];
    }

    __device__ inline volatile real* getStrikes() // X
    {
        return (real *)&sh_mem[blockDim.x * (5 * sizeof(real)) + values.maxValuationsBlock * 7 * sizeof(real)];
    }

    __device__ inline volatile real** getFirstYieldCurveRates()
    {
        return (real **)&sh_mem[blockDim.x * (5 * sizeof(real)) + values.maxValuationsBlock * 8 * sizeof(real)];
    }

    __device__ inline volatile uint16_t** getFirstYieldCurveTimeSteps()
    {
        return (uint16_t **)&sh_mem[blockDim.x * (5 * sizeof(real)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *))];
    }

    // 4-byte size variables, aligned to 4 byte (32-bit) border

    __device__ inline volatile int32_t* getLastUsedCIndices()
    {
        return (int32_t *)&sh_mem[blockDim.x * (5 * sizeof(real)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *))];
    }
       
    // 2-byte size variables, aligned to 2-byte (16-bit) border

    __device__ inline volatile uint16_t* getValuationFlags()
    {
        return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *) + sizeof(int32_t))];
    }

    __device__ inline volatile uint16_t* getJmaxs()
    {
        return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *) + sizeof(int32_t))];
    }

    __device__ inline volatile uint16_t* getNs()
    {
        return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *) + sizeof(int32_t) + sizeof(uint16_t))];
    }

    __device__ inline volatile uint16_t* getTermUnits()
    {
        return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *) + sizeof(int32_t) + 2 * sizeof(uint16_t))];
    }

    __device__ inline volatile uint16_t* getLastUsedYCTermIndices()
    {
        return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *) + sizeof(int32_t) + 3 * sizeof(uint16_t))];
    }

    __device__ inline volatile uint16_t* getLastCSteps()
    {
        return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *) + sizeof(int32_t) + 4 * sizeof(uint16_t))];
    }

    __device__ inline volatile uint16_t* getRemainingCashflows()
    {
        return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *) + sizeof(int32_t) + 5 * sizeof(uint16_t))];
    }

    __device__ inline volatile uint16_t* getYieldCurveTermCounts()
    {
        return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *) + sizeof(int32_t) + 6 * sizeof(uint16_t))];
    }

    __device__ inline volatile uint16_t* getLastExerciseSteps()
    {
        return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *) + sizeof(int32_t) + 7 * sizeof(uint16_t))];
    }

    __device__ inline volatile uint16_t* getFirstExerciseSteps()
    {
        return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *) + sizeof(int32_t) + 8 * sizeof(uint16_t))];
    }

    __device__ inline volatile uint16_t* getExerciseStepFrequencies()
    {
        return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *) + sizeof(int32_t) + 9 * sizeof(uint16_t))];
    }

    // 1-byte size variables, aligned to 1 byte (8-bit) border

    __device__ inline volatile OptionTypeE* getOptionTypes() // type
    {
        return (OptionTypeE *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *) + sizeof(int32_t) + 10 * sizeof(uint16_t))];
    }

    __device__ inline real getPs(const int index, const int branch)
    {
        switch (branch)
        {
        case 1:
            return getPus()[index]; // up
        case 2:
            return getPms()[index]; // mid
        case 3:
            return getPds()[index]; // down
        }
        return zero;
    }

    __device__ inline void setPs(const int index, const int branch, const real value)
    {
        switch (branch)
        {
        case 1:
            getPus()[index] = value; // up
        case 2:
            getPms()[index] = value; // mid
        case 3:
            getPds()[index] = value; // down
        }
    }
};

/**
 * global: absolute, relative to all valuations
 * local: relative to valuations in the thread-block this thread belongs to
 *
 * idxBlock: global index of the first valuation in the block for this thread
 * idxBlockNext: global index of the first valuation in the next block for this thread
 *
 * idx:(init) global index of the valuation
 *
 * valGIdx: global (absolute) index of the valuation in this thread
 * valLIdx: local (relative) index of the valuation in this thread
 * firstThreadIdx: index of the first thread handling this valuation
 *
 * width: (init) number of threads handling this valuation
 */
template<class KernelArgsT>
__global__ void kernelMultipleValuationsPerThreadBlock(const KernelValuations valuations, KernelArgsT args)
{
    // Compute valuation indices
    const auto firstValGIdxBlock = blockIdx.x == 0 ? 0 : args.values.inds[blockIdx.x - 1];
    const auto firstValGIdxBlockNext = args.values.inds[blockIdx.x];
    const auto idx = firstValGIdxBlock + threadIdx.x; // ?
    auto valGIdx = -1; // defined later
    auto valLIdx = -1; // defined later
    auto firstThreadIdx = -1; // defined later
    auto middleThreadIdx = -1; // defined later

    // Get the valuation and compute its constants
    if (idx < firstValGIdxBlockNext) // Do not fetch valuations from next block
    {
        args.getValuationInds()[threadIdx.x] = valuations.Widths[idx];

        const auto termUnits = valuations.TermUnits[idx];
        args.getTermUnits()[threadIdx.x] = termUnits;
        const auto termUnitsInYearCount = (int)ceil((real)year / termUnits);
        const auto termStepCount = valuations.TermSteps[idx];
        const auto dt = termUnitsInYearCount / (real)termStepCount;
        args.getDts()[threadIdx.x] = dt;
        args.getNs()[threadIdx.x] = termStepCount * termUnitsInYearCount * valuations.Maturities[idx];
        args.getStrikes()[threadIdx.x] = valuations.StrikePrices[idx];
        args.getOptionTypes()[threadIdx.x] = valuations.OptionTypes[idx];
        const auto a = valuations.MeanReversionRates[idx];
        const auto sigma = valuations.Volatilities[idx];
        const auto V = sigma * sigma * (one - exp(-two * a * dt)) / (two * a);
        const auto dr = sqrt(three * V);
        args.getMs()[threadIdx.x] = exp(-a * dt) - one;

        // simplified computations
        // dr = sigma * sqrt(three * dt);
        // args.getMs()[threadIdx.x] = -a * dt;

        args.getMdrdts()[threadIdx.x] = -dr * dt;
        args.getJmaxs()[threadIdx.x] = (int)(minus184 / args.getMs()[threadIdx.x]) + 1;
        args.getExpmOasdts()[threadIdx.x] = exp(-(valuations.Spreads[idx] / hundred)*dt);

        args.getLastExerciseSteps()[threadIdx.x] = valuations.LastExerciseSteps[idx];
        args.getFirstExerciseSteps()[threadIdx.x] = valuations.FirstExerciseSteps[idx];
        args.getExerciseStepFrequencies()[threadIdx.x] = valuations.ExerciseStepFrequencies[idx];

        const auto ycIndex = valuations.YieldCurveIndices[idx];
        const auto firstYCTermIndex = valuations.YieldCurveTermIndices[ycIndex];
        args.getYieldCurveTermCounts()[threadIdx.x] = valuations.YieldCurveTerms[ycIndex];
        args.getFirstYieldCurveRates()[threadIdx.x] = &valuations.YieldCurveRates[firstYCTermIndex];
        args.getFirstYieldCurveTimeSteps()[threadIdx.x] = &valuations.YieldCurveTimeSteps[firstYCTermIndex];
        args.getLastUsedYCTermIndices()[threadIdx.x] = 0;

        const auto lastUsedCIdx = valuations.CashflowIndices[idx] + (int)valuations.Cashflows[idx] - 1;
        args.getLastUsedCIndices()[threadIdx.x] = lastUsedCIdx;
        args.getLastCashflows()[threadIdx.x] = valuations.Repayments[lastUsedCIdx] + valuations.Coupons[lastUsedCIdx];
        args.getLastCSteps()[threadIdx.x] = valuations.CashflowSteps[lastUsedCIdx];
        args.getRemainingCashflows()[threadIdx.x] = valuations.Cashflows[idx];
#ifdef DEV
        if (idx == PRINT_IDX) printf("%d %d %d: Input %d %.18f %d %.18f %d %.18f %.18f %d %.18f %d %d %d %d %.18f %d %d %d %.18f %d %d %d %d %.18f %d %d %d\n", idx, threadIdx.x, blockIdx.x,
            args.getTermUnits()[threadIdx.x], args.getDts()[threadIdx.x], args.getNs()[threadIdx.x], args.getStrikes()[threadIdx.x], args.getOptionTypes()[threadIdx.x],
            args.getMs()[threadIdx.x], args.getMdrdts()[threadIdx.x], args.getJmaxs()[threadIdx.x], args.getExpmOasdts()[threadIdx.x],
            args.getLastExerciseSteps()[threadIdx.x], args.getFirstExerciseSteps()[threadIdx.x], args.getExerciseStepFrequencies()[threadIdx.x],
            args.getYieldCurveTermCounts()[threadIdx.x], *args.getFirstYieldCurveRates()[threadIdx.x], *args.getFirstYieldCurveTimeSteps()[threadIdx.x], args.getLastUsedYCTermIndices()[threadIdx.x],
            args.getLastUsedCIndices()[threadIdx.x], args.getLastCashflows()[threadIdx.x], args.getLastCSteps()[threadIdx.x], args.getRemainingCashflows()[threadIdx.x],
            ycIndex, firstYCTermIndex, valuations.YieldCurveRates[firstYCTermIndex], valuations.YieldCurveTimeSteps[firstYCTermIndex],
            valuations.CashflowIndices[idx], valuations.Cashflows[idx]);
#endif
    }
    else
    {
        args.getValuationInds()[threadIdx.x] = 0;
    }
    args.getValuationFlags()[threadIdx.x] = threadIdx.x == 0 ? blockDim.x : 0;
    __syncthreads();

    // Scan widths
    sgmScanIncBlock<Add<int32_t>>(args.getValuationInds(), args.getValuationFlags());

    if (idx <= firstValGIdxBlockNext)
    {
        firstThreadIdx = threadIdx.x == 0 ? 0 : args.getValuationInds()[threadIdx.x - 1];
    }
    __syncthreads();

    // Send valuation indices to all threads
    args.getValuationInds()[threadIdx.x] = 0;
    args.getValuationFlags()[threadIdx.x] = 0;
    __syncthreads();

    if (idx < firstValGIdxBlockNext)
    {
        args.getValuationInds()[firstThreadIdx] = threadIdx.x;
        args.getValuationFlags()[firstThreadIdx] = 2 * args.getJmaxs()[threadIdx.x] + 1;
    }
    else if (idx == firstValGIdxBlockNext && firstThreadIdx < blockDim.x) // fake valuation to fill block
    {
        args.getValuationInds()[firstThreadIdx] = threadIdx.x;
        args.getValuationFlags()[firstThreadIdx] = blockDim.x - firstThreadIdx;
    }
    __syncthreads();

    sgmScanIncBlock<Add<int32_t>>(args.getValuationInds(), args.getValuationFlags());

    // Let all threads know about their Q start
    if (idx <= firstValGIdxBlockNext)
    {
        args.getValuationFlags()[threadIdx.x] = firstThreadIdx;
    }
    __syncthreads();

    valLIdx = args.getValuationInds()[threadIdx.x];
    firstThreadIdx = args.getValuationFlags()[valLIdx];
    middleThreadIdx = (valGIdx < firstValGIdxBlockNext) ? firstThreadIdx + args.getJmaxs()[valLIdx] : -1;
    args.init(valLIdx, firstValGIdxBlock, firstValGIdxBlockNext, valuations.ValuationCount);
    valGIdx = args.getValuationIdx();
#ifdef DEV_LIMIT
    //if (valGIdx != 0 && valGIdx != 1) return;
    if (valGIdx != PRINT_IDX) return;
#endif
    __syncthreads();

    // --------------------------------------------------

    // Precompute exponent of rates for each node on the width on the tree (constant over forward propagation)
    const int j = (valGIdx < firstValGIdxBlockNext) ? threadIdx.x - args.getJmaxs()[valLIdx] - firstThreadIdx : 0;
    if (valGIdx < firstValGIdxBlockNext)
    {
        args.getRates()[threadIdx.x] = exp((real)j*args.getMdrdts()[valLIdx]);
        if (j == args.getJmaxs()[valLIdx])
        {
            args.getPus()[threadIdx.x] = PU_C(args.getJmaxs()[valLIdx], args.getMs()[valLIdx]); args.getPms()[threadIdx.x] = PM_C(args.getJmaxs()[valLIdx], args.getMs()[valLIdx]); args.getPds()[threadIdx.x] = PD_C(args.getJmaxs()[valLIdx], args.getMs()[valLIdx]);
        }
        else if (j == -args.getJmaxs()[valLIdx])
        {
            args.getPus()[threadIdx.x] = PU_B(-args.getJmaxs()[valLIdx], args.getMs()[valLIdx]); args.getPms()[threadIdx.x] = PM_B(-args.getJmaxs()[valLIdx], args.getMs()[valLIdx]); args.getPds()[threadIdx.x] = PD_B(-args.getJmaxs()[valLIdx], args.getMs()[valLIdx]);
        }
        else
        {
            args.getPus()[threadIdx.x] = PU_A(j, args.getMs()[valLIdx]); args.getPms()[threadIdx.x] = PM_A(j, args.getMs()[valLIdx]); args.getPds()[threadIdx.x] = PD_A(j, args.getMs()[valLIdx]);
        }
#ifdef DEV
        if (valGIdx == PRINT_IDX) printf("%d: %d: %.18f %.18f %.18f %.18f\n", valGIdx, threadIdx.x, /*blockIdx.x, threadIdx.x, j, args.getJmaxs()[valLIdx],*/ args.getRates()[threadIdx.x], args.getPs(threadIdx.x, 1), args.getPs(threadIdx.x, 2), args.getPs(threadIdx.x, 3));
#endif
    }
    __syncthreads();

    // Forward propagation
    if (idx < firstValGIdxBlockNext)
    {
        const auto alpha = interpolateRateAtTimeStep(args.getDts()[threadIdx.x], args.getTermUnits()[threadIdx.x], (const real *)args.getFirstYieldCurveRates()[threadIdx.x], (const uint16_t *)args.getFirstYieldCurveTimeSteps()[threadIdx.x], args.getYieldCurveTermCounts()[threadIdx.x], &args.getLastUsedYCTermIndices()[threadIdx.x]);
        args.getAlphas()[threadIdx.x] = exp(-alpha * args.getDts()[threadIdx.x]);
    }
    __syncthreads();
    if (valGIdx < firstValGIdxBlockNext && threadIdx.x == middleThreadIdx)
    {
        args.getQs()[threadIdx.x] = one; // Initialize the root of the tree
        args.setAlphaAt(0, args.getAlphas()[valLIdx], valLIdx);
#ifdef DEV1
        if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx)
            printf("%d %d: %.18f %.18f %.18f %d\n", valGIdx, 0, 1.0, args.getAlphas()[valLIdx], args.getAlphaAt(0, valLIdx), args.getLastUsedYCTermIndices()[valLIdx]);
#endif
    }
    __syncthreads();

    for (int i = 1; i <= args.getMaxHeight(); ++i)
    {
        const int jhigh = min(i, args.getJmaxs()[valLIdx]);

        // Precompute Qexp
        if (i <= args.getNs()[valLIdx] && j >= -jhigh && j <= jhigh)
        {
            const real expmAlphadt = args.getAlphas()[valLIdx];
            if (valGIdx < firstValGIdxBlockNext && threadIdx.x == middleThreadIdx) args.setAlphaAt(i - 1, expmAlphadt, valLIdx);
            args.getQs()[threadIdx.x] *= expmAlphadt * args.getRates()[threadIdx.x];
        }
        __syncthreads();

        // Forward iteration step, compute Qs in the next time step
        real Q = 0;
        if (i <= args.getNs()[valLIdx] && j >= -jhigh && j <= jhigh)
        {
            const auto expu = j == jhigh ? zero : args.getQs()[threadIdx.x + 1];
            const auto expm = args.getQs()[threadIdx.x];
            const auto expd = j == -jhigh ? zero : args.getQs()[threadIdx.x - 1];

            if (i == 1)
            {
                if (j == -jhigh) {
                    Q = args.getPs(threadIdx.x + 1, 3) * expu;
                }
                else if (j == jhigh) {
                    Q = args.getPs(threadIdx.x - 1, 1) * expd;
                }
                else {
                    Q = args.getPs(threadIdx.x, 2) * expm;
                }
            }
            else if (i <= args.getJmaxs()[valLIdx])
            {
                if (j == -jhigh) {
                    Q = args.getPs(threadIdx.x + 1, 3) * expu;
                }
                else if (j == -jhigh + 1) {
                    Q = args.getPs(threadIdx.x, 2) * expm +
                        args.getPs(threadIdx.x + 1, 3) * expu;
                }
                else if (j == jhigh) {
                    Q = args.getPs(threadIdx.x - 1, 1) * expd;
                }
                else if (j == jhigh - 1) {
                    Q = args.getPs(threadIdx.x - 1, 1) * expd +
                        args.getPs(threadIdx.x, 2) * expm;
                }
                else {
                    Q = args.getPs(threadIdx.x - 1, 1) * expd +
                        args.getPs(threadIdx.x, 2) * expm +
                        args.getPs(threadIdx.x + 1, 3) * expu;
                }
            }
            else
            {
                if (j == -jhigh) {
                    Q = args.getPs(threadIdx.x, 3) * expm +
                        args.getPs(threadIdx.x + 1, 3) * expu;
                }
                else if (j == -jhigh + 1) {
                    Q = args.getPs(threadIdx.x - 1, 2) * expd +
                        args.getPs(threadIdx.x, 2) * expm +
                        args.getPs(threadIdx.x + 1, 3) * expu;

                }
                else if (j == jhigh) {
                    Q = args.getPs(threadIdx.x - 1, 1) * expd +
                        args.getPs(threadIdx.x, 1) * expm;
                }
                else if (j == jhigh - 1) {
                    Q = args.getPs(threadIdx.x - 1, 1) * expd +
                        args.getPs(threadIdx.x, 2) * expm +
                        args.getPs(threadIdx.x + 1, 2) * expu;

                }
                else {
                    Q = (j == -jhigh + 2 ? args.getPs(threadIdx.x - 2, 1) * args.getQs()[threadIdx.x - 2] : zero) +
                        args.getPs(threadIdx.x - 1, 1) * expd +
                        args.getPs(threadIdx.x, 2) * expm +
                        args.getPs(threadIdx.x + 1, 3) * expu +
                        (j == jhigh - 2 ? args.getPs(threadIdx.x + 2, 3) * args.getQs()[threadIdx.x + 2] : zero);
                }
            }
        }
        __syncthreads();

        args.getQs()[threadIdx.x] = Q > zero ? Q * args.getRates()[threadIdx.x] : zero;
        __syncthreads();

        // Repopulate flags
        args.getValuationFlags()[threadIdx.x] = threadIdx.x == firstThreadIdx ? 2 * args.getJmaxs()[valLIdx] + 1 : 0;
        __syncthreads();

        // Determine the new alpha using equation 30.22
        // by summing up Qs from the next time step
        real Qexp = sgmScanIncBlock<Add<real>>(args.getQs(), args.getValuationFlags());
        if (valGIdx < firstValGIdxBlockNext && i <= args.getNs()[valLIdx] && threadIdx.x == firstThreadIdx + (2 * args.getJmaxs()[valLIdx] + 1) - 1)
        {
            args.getQexps()[valLIdx] = Qexp;
        }
        __syncthreads();

        if (idx < firstValGIdxBlockNext && i <= args.getNs()[threadIdx.x])
        {
            const auto alpha = computeAlpha(args.getQexps()[threadIdx.x], i - 1, args.getDts()[threadIdx.x], args.getTermUnits()[threadIdx.x], (const real *)args.getFirstYieldCurveRates()[threadIdx.x], (const uint16_t *)args.getFirstYieldCurveTimeSteps()[threadIdx.x], args.getYieldCurveTermCounts()[threadIdx.x], &args.getLastUsedYCTermIndices()[threadIdx.x]);
            args.getAlphas()[threadIdx.x] = exp(-alpha * args.getDts()[threadIdx.x]);
#ifdef DEV1
            if (idx == PRINT_IDX) printf("%d %d: %.18f %.18f %.18f %d\n", idx, i, args.getQexps()[threadIdx.x], alpha, args.getAlphas()[threadIdx.x], args.getLastUsedYCTermIndices()[threadIdx.x]);
#endif
        }
        __syncthreads();

        args.getQs()[threadIdx.x] = Q;
        __syncthreads();
#ifdef DEV1
        if (valGIdx == PRINT_IDX && threadIdx.x == 0 && i <= args.getNs()[threadIdx.x] && i > PRINT_FIRST_ITER && i < PRINT_LAST_ITER)
        {
            printf("%d %d: ", valGIdx, i);
            for (auto k = 0; k < blockDim.x; ++k)
            {
                printf("%d: %.18f ", k, args.getQs()[k]);
            }
            printf("\n");
        }
        __syncthreads();
#endif
    }

    // Backward propagation
    if (idx < firstValGIdxBlockNext)
    {
        const auto lastUsedCIdx = args.getLastUsedCIndices()[threadIdx.x];
#ifdef DEV2
        if (idx == PRINT_IDX)
            printf("%d %d: %d %d %.18f %.18f %d %.18f\n", idx, args.getNs()[threadIdx.x],
                lastUsedCIdx, args.getRemainingCashflows()[threadIdx.x], valuations.Repayments[lastUsedCIdx], valuations.Coupons[lastUsedCIdx],
                valuations.CashflowSteps[lastUsedCIdx], args.getLastCashflows()[threadIdx.x]);
#endif
        args.getRemainingCashflows()[threadIdx.x]--;
        if (valuations.CashflowSteps[lastUsedCIdx] <= args.getNs()[threadIdx.x] && args.getRemainingCashflows()[threadIdx.x] > 0)
        {
            args.getLastCSteps()[threadIdx.x] = valuations.CashflowSteps[lastUsedCIdx - 1];
            args.getLastUsedCIndices()[threadIdx.x]--;
        }
        else
        {
            args.getLastCSteps()[threadIdx.x] = valuations.CashflowSteps[lastUsedCIdx];

        }
    }
    __syncthreads();

    if (valGIdx < firstValGIdxBlockNext)
    {
        args.getQs()[threadIdx.x] = args.getLastCashflows()[valLIdx];
    }

    __syncthreads();

    for (auto i = args.getMaxHeight() - 1; i >= 0; --i)
    {
        const int jhigh = min(i, args.getJmaxs()[valLIdx]);

        real price = args.getQs()[threadIdx.x];

        auto isExerciseStep = false;
        auto ai = zero;

        // check if there is an option exercise at this step
        // add coupon and repayment if crossed a time step with a cashflow
        if (i < args.getNs()[valLIdx] && j >= -jhigh && j <= jhigh)
        {
            isExerciseStep = (i <= args.getLastExerciseSteps()[valLIdx] && i >= args.getFirstExerciseSteps()[valLIdx] && (args.getLastCSteps()[valLIdx] - i) % args.getExerciseStepFrequencies()[valLIdx] == 0);
#ifdef DEV2
            if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx)
                printf("%d %d: %.18f %d %d %d %d\n",
                    valGIdx, i, args.getAlphaAt(i, valLIdx), isExerciseStep, args.getLastCSteps()[valLIdx], (args.getLastCSteps()[valLIdx] - i) % args.getExerciseStepFrequencies()[threadIdx.x],
                    (args.getLastCSteps()[valLIdx] - i) % args.getExerciseStepFrequencies()[threadIdx.x] == 0);
#endif
            if (i == args.getLastCSteps()[valLIdx] - 1 && args.getRemainingCashflows()[valLIdx] > 0)
            {
#ifdef DEV2
                if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx)
                    printf("%d %d: %d %d coupon: %.18f %.18f\n", valGIdx, i, args.getLastUsedCIndices()[valLIdx], args.getRemainingCashflows()[valLIdx],
                        args.getQs()[threadIdx.x], args.getLastCashflows()[valLIdx]);
#endif
                args.getQs()[threadIdx.x] += args.getLastCashflows()[valLIdx];
#ifdef DEV2
                if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx)
                    printf("%d %d: %d %d coupon: %.18f\n", valGIdx, i, args.getLastUsedCIndices()[valLIdx], args.getRemainingCashflows()[valLIdx], args.getQs()[threadIdx.x]);
#endif
            }
        }
        __syncthreads();

        if (idx < firstValGIdxBlockNext && i < args.getNs()[threadIdx.x] && i == args.getLastCSteps()[threadIdx.x] - 1 && args.getRemainingCashflows()[threadIdx.x] > 0)
        {
            args.getLastUsedCIndices()[threadIdx.x] = (args.getLastUsedCIndices()[threadIdx.x] - 1 >= 0) ? args.getLastUsedCIndices()[threadIdx.x] - 1 : 0;
            auto lastUsedCIdx = args.getLastUsedCIndices()[threadIdx.x];
            args.getLastCSteps()[threadIdx.x] = valuations.CashflowSteps[lastUsedCIdx];
            args.getRemainingCashflows()[threadIdx.x]--;
        }
        __syncthreads();

        if (i < args.getNs()[valLIdx] && j >= -jhigh && j <= jhigh)
        {
            const auto expmAlphadt = (valGIdx < firstValGIdxBlockNext) ? args.getAlphaAt(i, valLIdx) : 0.0;
            const auto discFactor = expmAlphadt * args.getRates()[threadIdx.x] * args.getExpmOasdts()[valLIdx];

            real res;
            if (j == args.getJmaxs()[valLIdx])
            {
                // Top edge branching
                res = (args.getPs(threadIdx.x, 1) * args.getQs()[threadIdx.x] +
                    args.getPs(threadIdx.x, 2) * args.getQs()[threadIdx.x - 1] +
                    args.getPs(threadIdx.x, 3) * args.getQs()[threadIdx.x - 2]) *
                    discFactor;
            }
            else if (j == -args.getJmaxs()[valLIdx])
            {
                // Bottom edge branching
                res = (args.getPs(threadIdx.x, 1) * args.getQs()[threadIdx.x + 2] +
                    args.getPs(threadIdx.x, 2) * args.getQs()[threadIdx.x + 1] +
                    args.getPs(threadIdx.x, 3) * args.getQs()[threadIdx.x]) *
                    discFactor;
            }
            else
            {
                // Standard branching
                res = (args.getPs(threadIdx.x, 1) * args.getQs()[threadIdx.x + 1] +
                    args.getPs(threadIdx.x, 2) * args.getQs()[threadIdx.x] +
                    args.getPs(threadIdx.x, 3) * args.getQs()[threadIdx.x - 1]) *
                    discFactor;
            }

            // calculate accrued interest from cashflow
            ai = isExerciseStep && args.getLastCSteps()[valLIdx] != 0 && args.getRemainingCashflows()[valLIdx] > 0
                ? computeAccruedInterest(i, args.getLastCSteps()[valLIdx], valuations.CashflowSteps[args.getLastUsedCIndices()[valLIdx] + 1], valuations.Coupons[args.getLastUsedCIndices()[valLIdx]])
                : zero;

            // after obtaining the result from (i+1) nodes, set the call for ith node
            price = getOptionPayoff(isExerciseStep, args.getStrikes()[valLIdx], args.getOptionTypes()[valLIdx], res, ai);
        }
        __syncthreads();

        if (idx < firstValGIdxBlockNext && i < args.getNs()[threadIdx.x] && i == args.getLastCSteps()[threadIdx.x])
        {
            auto lastUsedCIdx = args.getLastUsedCIndices()[threadIdx.x];
            args.getLastCashflows()[threadIdx.x] = valuations.Repayments[lastUsedCIdx] + valuations.Coupons[lastUsedCIdx];
        }

        args.getQs()[threadIdx.x] = price;
#ifdef DEV2
        if (idx < firstValGIdxBlockNext && i < args.getNs()[threadIdx.x] && idx == PRINT_IDX && isExerciseStep && args.getLastCSteps()[threadIdx.x] != 0 && args.getRemainingCashflows()[threadIdx.x] > 0)
        {
            auto lastUsedCIdx = args.getLastUsedCIndices()[threadIdx.x];
            printf("%d %d: ai %f %d %d %f %d %d %f\n", idx, i, ai, args.getLastCSteps()[threadIdx.x],
                valuations.CashflowSteps[lastUsedCIdx + 1], valuations.Coupons[lastUsedCIdx],
                valuations.CashflowSteps[lastUsedCIdx + 1] - args.getLastCSteps()[threadIdx.x], valuations.CashflowSteps[lastUsedCIdx + 1] - i,
                (real)(valuations.CashflowSteps[lastUsedCIdx + 1] - args.getLastCSteps()[threadIdx.x] - valuations.CashflowSteps[lastUsedCIdx + 1] - i) / (valuations.CashflowSteps[lastUsedCIdx + 1] - args.getLastCSteps()[threadIdx.x]));
        }

        if (i < args.getNs()[valLIdx] && valGIdx == PRINT_IDX && valGIdx < firstValGIdxBlockNext && threadIdx.x == middleThreadIdx)
            printf("%d %d: %.18f \n", valGIdx, i, args.getQs()[threadIdx.x]);
#endif
        __syncthreads();
    }

    if (valGIdx < firstValGIdxBlockNext && threadIdx.x == middleThreadIdx)
    {
        args.values.res[valGIdx] = args.getQs()[threadIdx.x];
#ifdef DEV
        if (valGIdx == PRINT_IDX) printf("%d: res %.18f\n", valGIdx, args.getQs()[threadIdx.x]);
#endif
    }
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
    void runKernel(CudaValuations &valuations, std::vector<real> &results, thrust::device_vector<int32_t> &inds,
        KernelArgsValuesT &values, const int totalAlphasCount, const int maxValuationsBlock)
    {
        const int sharedMemorySize = 
            BlockSize * (5 * sizeof(real) + sizeof(uint16_t))
            + maxValuationsBlock * (8 * sizeof(real) + sizeof(real *) + sizeof(uint16_t *) + sizeof(int32_t) + 10 * sizeof(uint16_t) + sizeof(OptionTypeE));
        thrust::device_vector<real> alphas(totalAlphasCount);
        thrust::device_vector<real> result(valuations.ValuationCount);

        valuations.DeviceMemory += vectorsizeof(inds);
        valuations.DeviceMemory += vectorsizeof(alphas);
        valuations.DeviceMemory += vectorsizeof(result);
        runtime.DeviceMemory = valuations.DeviceMemory;

        if (IsTest)
        {
            std::cout << "Running pricing for " << valuations.ValuationCount <<
#ifdef USE_DOUBLE
                " double"
#else
                " float"
#endif
                << " valuations with block size " << BlockSize << std::endl;
            std::cout << "Shared memory size " << sharedMemorySize << ", alphas count " << totalAlphasCount << std::endl;
            std::cout << "Global memory size " << valuations.DeviceMemory / (1024.0 * 1024.0) << " MB" << std::endl;

            cudaDeviceSynchronize();
            size_t memoryFree, memoryTotal;
            cudaMemGetInfo(&memoryFree, &memoryTotal);
            std::cout << "Current GPU memory usage " << (memoryTotal - memoryFree) / (1024.0 * 1024.0) << " MB out of " << memoryTotal / (1024.0 * 1024.0) << " MB " << std::endl;
        }

        values.res = thrust::raw_pointer_cast(result.data());
        values.alphas = thrust::raw_pointer_cast(alphas.data());
        values.inds = thrust::raw_pointer_cast(inds.data());
        values.maxValuationsBlock = maxValuationsBlock;
        KernelArgsT kernelArgs(values);

        auto time_begin_kernel = std::chrono::steady_clock::now();
        kernelMultipleValuationsPerThreadBlock << <inds.size(), BlockSize, sharedMemorySize >> > (valuations.KernelValuations, kernelArgs);
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

    void run(const Valuations &valuations, std::vector<real> &results,
        const int blockSize = -1, const SortType sortType = SortType::NONE, bool isTest = false)
    {
        CudaValuations cudaValuations(valuations);

        // Start timing when input is copied to device
        cudaDeviceSynchronize();
        auto timeBegin = std::chrono::steady_clock::now();

        cudaValuations.initialize();

        // Get the max width
        auto maxWidth = *(thrust::max_element(cudaValuations.Widths.begin(), cudaValuations.Widths.end()));

        BlockSize = blockSize;
        if (BlockSize <= 0)
        {
            // Compute the smallest block size for the max width
            BlockSize = ((maxWidth + 32 - 1) / 32) * 32;
        }

        if (maxWidth > BlockSize)
        {
            std::ostringstream oss;
            oss << "Block size (" << BlockSize << ") cannot be smaller than max valuation width (" << maxWidth << ").";
            throw std::invalid_argument(oss.str());
        }

        run(cudaValuations, results, timeBegin, blockSize, sortType, isTest);
    }

    void run(CudaValuations &cudaValuations, std::vector<real> &results, const std::chrono::time_point<std::chrono::steady_clock> timeBegin,
        const int blockSize, const SortType sortType = SortType::NONE, const bool isTest = false)
    {
        TimeBegin = timeBegin;
        IsTest = isTest;

        cudaValuations.sortValuations(sortType, isTest);
        runPreprocessing(cudaValuations, results);
    }

};

}

}

#endif
