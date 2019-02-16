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

            __device__ inline volatile real* getQs()
            {
                return (real *)&sh_mem;
            }

            __device__ inline volatile int32_t* getValuationInds()
            {
                return (int32_t *)&sh_mem;  // Sharing the same array with Qs!
            }

            __device__ inline volatile uint16_t* getValuationFlags()
            {
                return (uint16_t *)&sh_mem[blockDim.x * sizeof(real)];
            }

            __device__ inline volatile real* getRates()
            {
                return (real *)&sh_mem[blockDim.x * (sizeof(real) + sizeof(uint16_t))];
            }

            __device__ inline volatile real* getPus()
            {
                return (real *)&sh_mem[blockDim.x * (2 * sizeof(real) + sizeof(uint16_t))];
            }

            __device__ inline volatile real* getPms()
            {
                return (real *)&sh_mem[blockDim.x * (3 * sizeof(real) + sizeof(uint16_t))];
            }

            __device__ inline volatile real* getPds()
            {
                return (real *)&sh_mem[blockDim.x * (4 * sizeof(real) + sizeof(uint16_t))];
            }

            __device__ inline volatile real* getAlphas()
            {
                return (real *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t))];
            }

            __device__ inline volatile real* getDts()
            {
                return (real *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * sizeof(real)];
            }

            __device__ inline volatile real* getQexps()
            {
                return (real *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * 2 * sizeof(real)];
            }

            __device__ inline volatile uint16_t* getNs()
            {
                return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * 3 * sizeof(real)];
            }

            __device__ inline volatile uint16_t* getTermUnits()
            {
                return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (3 * sizeof(real) + sizeof(uint16_t))];
            }

            __device__ inline volatile uint16_t* getLastUsedYCTermIdx()
            {
                return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (3 * sizeof(real) + 2 * sizeof(uint16_t))];
            }

            __device__ inline volatile uint16_t* getLastUsedCIdx()
            {
                return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (3 * sizeof(real) + 3 * sizeof(uint16_t))];
            }

            __device__ inline volatile uint16_t* getLastCStep()
            {
                return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (3 * sizeof(real) + 4 * sizeof(uint16_t))];
            }

            __device__ inline volatile uint16_t* getLastCashflow()
            {
                return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (3 * sizeof(real) + 5 * sizeof(uint16_t))];
            }

            __device__ inline volatile uint16_t* getCashflowsRemaining()
            {
                return (uint16_t *)&sh_mem[blockDim.x * (5 * sizeof(real) + sizeof(uint16_t)) + values.maxValuationsBlock * (3 * sizeof(real) + 6 * sizeof(uint16_t))];
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
            int32_t width;

            if (idx < firstValGIdxBlockNext) // Do not fetch valuations from next block
            {
#ifdef DEV
                printf("idx %d threadIdx.x %d blockIdx.x %d\n", idx, threadIdx.x, blockIdx.x);
#endif
                width = valuations.Widths[idx];
                args.getValuationInds()[threadIdx.x] = width;

                auto termUnits = valuations.TermUnits[idx];
                args.getTermUnits()[threadIdx.x] = termUnits;
                auto termUnitsInYearCount = (int)lround((real)year / termUnits);
                auto termStepCount = valuations.TermSteps[idx];
                args.getDts()[threadIdx.x] = termUnitsInYearCount / (real)termStepCount;
                args.getNs()[threadIdx.x] = termStepCount * termUnitsInYearCount * valuations.Maturities[idx];
                args.getLastUsedYCTermIdx()[threadIdx.x] = 0;
                args.getLastUsedCIdx()[threadIdx.x] = valuations.CashflowIndices[idx] + valuations.Cashflows[idx] - 1;
                args.getLastCStep()[threadIdx.x] = valuations.CashflowSteps[args.getLastUsedCIdx()[threadIdx.x]];
                args.getCashflowsRemaining()[threadIdx.x] = valuations.Cashflows[idx];
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
                args.getValuationFlags()[firstThreadIdx] = width;
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

            // --------------------------------------------------

            valLIdx = args.getValuationInds()[threadIdx.x];
            firstThreadIdx = args.getValuationFlags()[valLIdx];

            // Get the valuation and compute its constants
            ValuationConstants c;
            args.init(valLIdx, firstValGIdxBlock, firstValGIdxBlockNext, valuations.ValuationCount);
            valGIdx = args.getValuationIdx();
#ifdef DEV_LIMIT
            //if (valGIdx != 0 && valGIdx != 1) return;
            if (valGIdx != PRINT_IDX) return;
#endif

            if (valGIdx < firstValGIdxBlockNext)
            {
                computeConstants(c, valuations, args.getValuationIdx());
#ifdef DEV
                if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx)
                {
                    printf("threadIdx.x %d blockIdx.x %d firstValGIdxBlock %d firstValGIdxBlockNext %d idx %d firstThreadIdx %d valLIdx %d valGIdx %d c.jmax %d\n",
                        threadIdx.x, blockIdx.x, firstValGIdxBlock, firstValGIdxBlockNext, idx, firstThreadIdx, valLIdx, valGIdx, c.jmax);
                    printf("%d: %d %d %d %d %f %d %d\n", valGIdx, c.firstYCTermIdx, c.LastExerciseStep, c.FirstExerciseStep, c.ExerciseStepFrequency, valuations.YieldCurveRates[c.firstYCTermIdx], valuations.YieldCurveTimeSteps[c.firstYCTermIdx], valuations.YieldCurveTerms[valuations.YieldCurveIndices[idx]]);
                }
#endif
            }
            else
            {
                c.n = 0;
                c.width = blockDim.x - firstThreadIdx;
            }
            middleThreadIdx = firstThreadIdx + c.jmax;
            __syncthreads();

            // --------------------------------------------------

            // Precompute exponent of rates for each node on the width on the tree (constant over forward propagation)
            const int j = threadIdx.x - c.jmax - firstThreadIdx;
            if (valGIdx < firstValGIdxBlockNext)
            {
                args.getRates()[threadIdx.x] = exp((real)j*c.mdrdt);
                if (j == c.jmax)
                {
                    args.getPus()[threadIdx.x] = PU_C(c.jmax, c.M); args.getPms()[threadIdx.x] = PM_C(c.jmax, c.M); args.getPds()[threadIdx.x] = PD_C(c.jmax, c.M);
                }
                else if (j == -c.jmax)
                {
                    args.getPus()[threadIdx.x] = PU_B(-c.jmax, c.M); args.getPms()[threadIdx.x] = PM_B(-c.jmax, c.M); args.getPds()[threadIdx.x] = PD_B(-c.jmax, c.M);
                }
                else
                {
                    args.getPus()[threadIdx.x] = PU_A(j, c.M); args.getPms()[threadIdx.x] = PM_A(j, c.M); args.getPds()[threadIdx.x] = PD_A(j, c.M);
                }
#ifdef DEV
                if (valGIdx == PRINT_IDX) printf("%d: %d: %d %d %d %d %f %f %f %f\n", valGIdx, 0, blockIdx.x, threadIdx.x, j, c.jmax, args.getRates()[threadIdx.x], args.getPs(threadIdx.x, 1), args.getPs(threadIdx.x, 2), args.getPs(threadIdx.x, 3));
#endif
            }
            __syncthreads();

            // Forward propagation
            if (idx < firstValGIdxBlockNext)
            {
                const auto alpha = interpolateRateAtTimeStep(args.getDts()[threadIdx.x], args.getTermUnits()[threadIdx.x], c.firstYieldCurveRate, c.firstYieldCurveTimeStep, c.yieldCurveTermCount, &args.getLastUsedYCTermIdx()[threadIdx.x]);
                args.getAlphas()[threadIdx.x] = exp(-alpha * c.dt);
//#ifdef DEV1
//                if (idx == PRINT_IDX)
//                    printf("%d %d: %.18f %.18f threadIdx %d dt %f termUnit %d n %d\n",
//                        idx, 0, alpha, args.getAlphas()[threadIdx.x], threadIdx.x, args.getDts()[threadIdx.x], args.getTermUnits()[threadIdx.x], args.getNs()[threadIdx.x]);
//#endif
            }
            __syncthreads();
            if (valGIdx < firstValGIdxBlockNext && threadIdx.x == middleThreadIdx)
            {
                args.getQs()[threadIdx.x] = one; // Initialize the root of the tree
                args.setAlphaAt(0, args.getAlphas()[valLIdx], valLIdx);
#ifdef DEV1
                if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx)
                    printf("%d %d: %.18f %.18f %.18f %d\n", valGIdx, 0, 1.0, args.getAlphas()[valLIdx], args.getAlphaAt(0, valLIdx), args.getLastUsedYCTermIdx()[valLIdx]);
#endif
            }
            __syncthreads();

            for (int i = 1; i <= args.getMaxHeight(); ++i)
            {
                const int jhigh = min(i, c.jmax);

                // Precompute Qexp
                if (i <= c.n && j >= -jhigh && j <= jhigh)
                {
                    const real expmAlphadt = args.getAlphas()[valLIdx];
                    if (threadIdx.x == middleThreadIdx) args.setAlphaAt(i - 1, expmAlphadt, valLIdx);
//#ifdef DEV1
//                    if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx)
//                        printf("%d %d: %.18f %.18f\n", valGIdx, i, expmAlphadt, args.getAlphaAt(i - 1, valLIdx));
//#endif

                    args.getQs()[threadIdx.x] *= expmAlphadt * args.getRates()[threadIdx.x];
                }
                __syncthreads();

                // Forward iteration step, compute Qs in the next time step
                real Q = 0;
                if (i <= c.n && j >= -jhigh && j <= jhigh)
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
                    else if (i <= c.jmax)
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
                            Q = ((j == -jhigh + 2) ? args.getPs(threadIdx.x - 2, 1) * args.getQs()[threadIdx.x - 2] : zero) +
                                args.getPs(threadIdx.x - 1, 1) * expd +
                                args.getPs(threadIdx.x, 2) * expm +
                                args.getPs(threadIdx.x + 1, 3) * expu +
                                ((j == jhigh - 2) ? args.getPs(threadIdx.x + 2, 3) * args.getQs()[threadIdx.x + 2] : zero);
                        }
                    }
                }
                __syncthreads();

                args.getQs()[threadIdx.x] = Q > zero ? Q * args.getRates()[threadIdx.x] : zero;
                __syncthreads();

                // Repopulate flags
                args.getValuationFlags()[threadIdx.x] = threadIdx.x == firstThreadIdx ? c.width : 0;
                __syncthreads();

                // Determine the new alpha using equation 30.22
                // by summing up Qs from the next time step
                real Qexp = sgmScanIncBlock<Add<real>>(args.getQs(), args.getValuationFlags());
                if (i <= c.n && threadIdx.x == firstThreadIdx + c.width - 1)
                {
                    args.getQexps()[valLIdx] = Qexp;
                }
                __syncthreads();

                if (idx < firstValGIdxBlockNext && i <= args.getNs()[threadIdx.x])
                {
                    const auto alpha = computeAlpha(args.getQexps()[threadIdx.x], i - 1, args.getDts()[threadIdx.x], args.getTermUnits()[threadIdx.x], c.firstYieldCurveRate, c.firstYieldCurveTimeStep, c.yieldCurveTermCount, &args.getLastUsedYCTermIdx()[threadIdx.x]);
                    args.getAlphas()[threadIdx.x] = exp(-alpha * c.dt);
#ifdef DEV1
                    if (idx == PRINT_IDX) printf("%d %d: %.18f %.18f %.18f %d\n", idx, i, args.getQexps()[threadIdx.x], alpha, args.getAlphas()[threadIdx.x], args.getLastUsedYCTermIdx()[threadIdx.x]);
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
                const auto lastUsedCIdx = args.getLastUsedCIdx()[threadIdx.x];
                args.getLastCashflow()[threadIdx.x] = valuations.Repayments[lastUsedCIdx] + valuations.Coupons[lastUsedCIdx];
#ifdef DEV2
                if (idx == PRINT_IDX) printf("%d %d: %d %d %f %f %d %f\n", idx, c.n, 
                    lastUsedCIdx, args.getCashflowsRemaining()[threadIdx.x], valuations.Repayments[lastUsedCIdx], valuations.Coupons[lastUsedCIdx], valuations.CashflowSteps[lastUsedCIdx], args.getLastCashflow()[threadIdx.x]);
#endif
                args.getCashflowsRemaining()[threadIdx.x] -= 1;
                if (valuations.CashflowSteps[lastUsedCIdx] <= c.n && args.getCashflowsRemaining()[threadIdx.x] > 0)
                {
                    args.getLastCStep()[threadIdx.x] = valuations.CashflowSteps[lastUsedCIdx - 1];
                    args.getLastUsedCIdx()[threadIdx.x] -= 1;
                }
                else
                {
                    valuations.CashflowSteps[lastUsedCIdx];
                }
            }
            __syncthreads();

            if (valGIdx < firstValGIdxBlockNext)
            {
                args.getQs()[threadIdx.x] = args.getLastCashflow()[valLIdx];
            }
            __syncthreads();

            for (auto i = args.getMaxHeight() - 1; i >= 0; --i)
            {
                const int jhigh = min(i, c.jmax);

                real price = args.getQs()[threadIdx.x];

                auto isExerciseStep = false;
                auto ai = zero;

                if (i < c.n && j >= -jhigh && j <= jhigh)
                {
                    isExerciseStep = (i <= c.LastExerciseStep && i >= c.FirstExerciseStep && (args.getLastCStep()[valLIdx] - i) % c.ExerciseStepFrequency == 0);
#ifdef DEV2
                    if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx)
                        printf("%d %d: %d %d %d %d\n", 
                            valGIdx, i, isExerciseStep, args.getLastCStep()[valLIdx], (args.getLastCStep()[valLIdx] - i) % c.ExerciseStepFrequency,
                            (args.getLastCStep()[valLIdx] - i) % c.ExerciseStepFrequency == 0);

                    //if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx)
                    //    printf("%d %d: ERROR %d %d %d %d %d\n",
                    //        valGIdx, i, 
                    //        args.getCashflowsRemaining()[valLIdx], (i == args.getLastCStep()[valLIdx] - 1 && args.getCashflowsRemaining()[valLIdx] > 0),
                    //        args.getLastCStep()[valLIdx], args.getLastCStep()[valLIdx] - 1, args.getCashflowsRemaining()[valLIdx] > 0);
#endif
                    // add coupon and repayment if crossed a time step with a cashflow
                    if (i == args.getLastCStep()[valLIdx] - 1 && args.getCashflowsRemaining()[valLIdx] > 0)
                    {
#ifdef DEV2
                        if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx)
                            printf("%d %d: %d %d coupon: %.18f %.18f\n", valGIdx, i, args.getLastUsedCIdx()[valLIdx], args.getCashflowsRemaining()[valLIdx], args.getQs()[threadIdx.x], args.getLastCashflow()[valLIdx]);
#endif
                        args.getQs()[threadIdx.x] += args.getLastCashflow()[valLIdx];
#ifdef DEV2
                        if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx)
                            printf("%d %d: %d %d coupon: %.18f\n", valGIdx, i, args.getLastUsedCIdx()[valLIdx], args.getCashflowsRemaining()[valLIdx], args.getQs()[threadIdx.x]);
#endif
                        
                    }
                }
                __syncthreads();

                if (idx < firstValGIdxBlockNext && i == args.getLastCStep()[valLIdx] - 1 && args.getCashflowsRemaining()[valLIdx] > 0)
                {
                    auto lastUsedCIdx = args.getLastUsedCIdx()[threadIdx.x];
                    args.getLastCStep()[threadIdx.x] = valuations.CashflowSteps[lastUsedCIdx];
                    args.getCashflowsRemaining()[threadIdx.x]--;
                }
                __syncthreads();

#ifdef DEV2
                if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx && i == args.getLastCStep()[valLIdx])
                    {
                        auto lastUsedCIdx = args.getLastUsedCIdx()[valLIdx];
                        printf("%d %d: ai %f %d %d %d %f %d %d %f\n", valGIdx, i, ai, c.termStepCount, args.getLastCStep()[valLIdx],
                            valuations.CashflowSteps[lastUsedCIdx + 1], valuations.Coupons[lastUsedCIdx],
                            valuations.CashflowSteps[lastUsedCIdx + 1] - args.getLastCStep()[valLIdx], valuations.CashflowSteps[lastUsedCIdx + 1] - i,
                            (real)(valuations.CashflowSteps[lastUsedCIdx + 1] - args.getLastCStep()[valLIdx] - valuations.CashflowSteps[lastUsedCIdx + 1] - i) / (valuations.CashflowSteps[lastUsedCIdx + 1] - args.getLastCStep()[valLIdx]));
                    }
#endif
                if (i < c.n && j >= -jhigh && j <= jhigh)
                {
                    const auto expmAlphadt = args.getAlphaAt(i, valLIdx);
//#ifdef DEV2
//                    if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx) printf("%d %d: %.18f\n", valGIdx, i, expmAlphadt);
//#endif
                    const auto discFactor = expmAlphadt * args.getRates()[threadIdx.x] * c.expmOasdt;

                    real res;
                    if (j == c.jmax)
                    {
                        // Top edge branching
                        res = (args.getPs(threadIdx.x, 1) * args.getQs()[threadIdx.x] +
                            args.getPs(threadIdx.x, 2) * args.getQs()[threadIdx.x - 1] +
                            args.getPs(threadIdx.x, 3) * args.getQs()[threadIdx.x - 2]) *
                            discFactor;
                    }
                    else if (j == -c.jmax)
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
                    ai = isExerciseStep && args.getLastCStep()[valLIdx] != 0 && args.getCashflowsRemaining()[valLIdx] > 0
                        ? computeAccruedInterest(0, i, args.getLastCStep()[valLIdx], valuations.CashflowSteps[args.getLastUsedCIdx()[valLIdx] + 1], valuations.Coupons[args.getLastUsedCIdx()[valLIdx]])
                        : zero;
#ifdef DEV
                    if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx && i == args.getLastCStep()[valLIdx])
                    {
                        printf("%d %d: ai %f lastCStep %d nextCStep %d coupon %f lastUsedCIdx %d\n", valGIdx, i, ai, args.getLastCStep()[valLIdx], valuations.CashflowSteps[args.getLastUsedCIdx()[valLIdx] + 1], valuations.Coupons[args.getLastUsedCIdx()[valLIdx]], args.getLastUsedCIdx()[valLIdx]);
                    }
#endif
                    // after obtaining the result from (i+1) nodes, set the call for ith node
                    price = getOptionPayoff(isExerciseStep, c.X, c.type, res, ai);
#ifdef DEV
                    if (valGIdx == PRINT_IDX && threadIdx.x == middleThreadIdx && isExerciseStep)
                    {
                        printf("%d %d: ai %f lastCStep %d isExerciseStep %d res %f price %f\n", valGIdx, i, ai, args.getLastCStep()[valLIdx], isExerciseStep, res, price);
                    }
#endif
                }
                __syncthreads();

                if (idx < firstValGIdxBlockNext && i == args.getLastCStep()[valLIdx])
                {
                    auto lastUsedCIdx = args.getLastUsedCIdx()[threadIdx.x];
                    args.getLastCashflow()[threadIdx.x] = valuations.Repayments[lastUsedCIdx] + valuations.Coupons[lastUsedCIdx];
                    args.getLastUsedCIdx()[threadIdx.x] = (lastUsedCIdx - 1 >= 0) ? args.getLastUsedCIdx()[threadIdx.x] - 1 : 0;
#ifdef DEV
                    if (idx == PRINT_IDX)
                        printf("%d %d: %d ai %f %d %d %d %f %f %d %d\n", idx, i, lastUsedCIdx, ai, c.termStepCount, args.getLastCStep()[threadIdx.x],
                            valuations.CashflowSteps[lastUsedCIdx + 1], valuations.Repayments[lastUsedCIdx], valuations.Coupons[lastUsedCIdx],
                            valuations.CashflowSteps[lastUsedCIdx + 1] - args.getLastCStep()[threadIdx.x], valuations.CashflowSteps[lastUsedCIdx + 1] - i);
#endif
                }

                args.getQs()[threadIdx.x] = price;
#ifdef DEV2
                if (i < c.n  && valGIdx == PRINT_IDX && valGIdx < firstValGIdxBlockNext && threadIdx.x == middleThreadIdx) printf("%d %d: %.18f \n", valGIdx, i, args.getQs()[threadIdx.x]);
#endif
                __syncthreads();
            }

            if (valGIdx < firstValGIdxBlockNext && threadIdx.x == middleThreadIdx)
            {
                args.values.res[valGIdx] = args.getQs()[threadIdx.x];
#ifdef DEV
                printf("%d: res %.18f\n", valGIdx, args.getQs()[threadIdx.x]);
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
                const int sharedMemorySize = (5 * sizeof(real) + sizeof(uint16_t)) * BlockSize + (3 * sizeof(real) + 7 * sizeof(uint16_t)) * maxValuationsBlock;
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
