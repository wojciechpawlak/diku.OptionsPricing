// basic file operations
#include <iostream>

// math
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

//#define USE_GETOPT_ARGS

#ifdef USE_GETOPT_ARGS
#include "../common/getoptpp/getopt_pp_standalone.h"
#else
#include "../common/cxxopts/cxxopts.hpp"
#endif

#include "../common/Valuations.hpp"
#include "../common/Real.hpp"

using namespace std;
using namespace trinom;

real exampleYieldCurveHW1FData_Ps[] = { 0.0501772, 0.0498284, 0.0497234, 0.0496157, 0.0499058, 0.0509389, 0.0579733, 0.0630595, 0.0673464, 0.0694816, 0.0708807, 0.0727527, 0.0730852, 0.0739790, 0.0749015 };
uint16_t exampleYieldCurveHW1FData_ts[] = { 3, 31, 62, 94, 185, 367, 731, 1096, 1461, 1826, 2194, 2558, 2922, 3287, 3653 };

int randIntInRange(int a, int b)
{
    return rand() % (b - a + 1) + a;
}

real randRealInRange(real low, real high)
{
    real t = (real)rand() / (real)RAND_MAX;
    return ((real)1.0 - t) * low + t * high;
}

struct RandValuation
{
    // Model parameters
    uint16_t TermUnit;
    uint16_t TermStep;
    real MeanReversionRate;
    real Volatility;
    // Bond parameters
    real Maturity;
    uint16_t Cashflows;
    std::vector<uint16_t> CashflowSteps;
    std::vector<real> Repayments;
    std::vector<real> Coupons;
    real Spread;
    // Option parameters
    OptionTypeE OptionType;
    real StrikePrice;
    uint16_t FirstExerciseStep;
    uint16_t LastExerciseStep;
    uint16_t ExerciseStepFrequency;

    uint16_t YieldCurveIndex;

    int Width;
    int Height;
    bool Skewed;

    RandValuation(const real maturity, const int termStepCount, const real meanReversionRate, const bool skewed, const int yieldCurveCount)
    {
        TermUnit = 360;
        TermStep = termStepCount;
        MeanReversionRate = meanReversionRate;
        Volatility = randRealInRange(0.0, 0.9);

        Maturity = maturity;
        computeWidth();
        computeHeight();

        computeCashflows();
        Spread = (real)randIntInRange(1, 1000) / 100;
        OptionType = OptionTypeE(randIntInRange(0, 1));
        StrikePrice = (real)randIntInRange(50, 150);

        computeOptionality();

        YieldCurveIndex = randIntInRange(0, yieldCurveCount - 1);

        Skewed = skewed;
    }

    void computeWidth()
    {
        Width = 2 * ((int)(minus184 / (exp(-MeanReversionRate * (lround((real)year / TermUnit) / (real)TermStep)) - one)) + 1) + 1;
    }

    void computeHeight()
    {
        Height = TermStep * lround((real)year / TermUnit) * Maturity;
    }

    void computeCashflows()
    {
        int currentTimeStep = TermStep;
        int cashflowFrequency = TermStep;
        while (currentTimeStep <= Height)
        {
            CashflowSteps.push_back(currentTimeStep);
            real coupon = (real)randIntInRange(1, 10);
            Coupons.push_back(coupon);
            real repayment = (currentTimeStep == Height) ? hundred : zero;
            Repayments.push_back(repayment);
            currentTimeStep += cashflowFrequency;
        }
        Cashflows = Coupons.size();
    }

    void computeOptionality()
    {
        ExerciseStepFrequency = randIntInRange(1, 12);
        int currentTimeStep = (Height <= Maturity) ? ExerciseStepFrequency : TermStep;
        FirstExerciseStep = currentTimeStep;
        while (currentTimeStep < Height - ExerciseStepFrequency)
        {
            currentTimeStep += ExerciseStepFrequency;
        }
        LastExerciseStep = currentTimeStep;
    }
};

struct RandYieldCurve
{
    // Yield Curve parameters
    uint16_t YieldCurveTermCount;
    std::vector<real> YieldCurveRates;
    std::vector<uint16_t> YieldCurveTimeSteps;

    RandYieldCurve()
    {
        YieldCurveTermCount = 15;
        YieldCurveRates.insert(YieldCurveRates.end(), std::begin(exampleYieldCurveHW1FData_Ps), std::end(exampleYieldCurveHW1FData_Ps));
        YieldCurveTimeSteps.insert(YieldCurveTimeSteps.end(), std::begin(exampleYieldCurveHW1FData_ts), std::end(exampleYieldCurveHW1FData_ts));
    }
};

const vector<real> reversionRates = { 0.00433, 0.00434, 0.00435, 0.00436, 0.00437, 0.00438, 0.00439, 0.00440, 0.00441, 0.00442, 0.00443, 0.00444, 0.00445, 0.00446, 0.00447, 0.00448, 0.00449, 0.00450, 0.00451, 0.00452, 0.00453, 0.00454, 0.00455, 0.00456, 0.00457, 0.00458, 0.00459, 0.00460, 0.00461, 0.00462, 0.00463, 0.00464, 0.00465, 0.00466, 0.00467, 0.00468, 0.00469, 0.00470, 0.00471, 0.00472, 0.00473, 0.00474, 0.00475, 0.00476, 0.00477, 0.00479, 0.00480, 0.00481, 0.00482, 0.00483, 0.00484, 0.00485, 0.00486, 0.00487, 0.00488, 0.00489, 0.00490, 0.00491, 0.00492, 0.00493, 0.00495, 0.00496, 0.00497, 0.00498, 0.00499, 0.00500, 0.00501, 0.00502, 0.00504, 0.00505, 0.00506, 0.00507, 0.00508, 0.00509, 0.00511, 0.00512, 0.00513, 0.00514, 0.00515, 0.00516, 0.00518, 0.00519, 0.00520, 0.00521, 0.00523, 0.00524, 0.00525, 0.00526, 0.00528, 0.00529, 0.00530, 0.00531, 0.00533, 0.00534, 0.00535, 0.00537, 0.00538, 0.00539, 0.00540, 0.00542, 0.00543, 0.00544, 0.00546, 0.00547, 0.00549, 0.00550, 0.00551, 0.00553, 0.00554, 0.00555, 0.00557, 0.00558, 0.00560, 0.00561, 0.00562, 0.00564, 0.00565, 0.00567, 0.00568, 0.00570, 0.00571, 0.00573, 0.00574, 0.00576, 0.00577, 0.00579, 0.00580, 0.00582, 0.00583, 0.00585, 0.00586, 0.00588, 0.00589, 0.00591, 0.00593, 0.00594, 0.00596, 0.00597, 0.00599, 0.00601, 0.00602, 0.00604, 0.00606, 0.00607, 0.00609, 0.00611, 0.00612, 0.00614, 0.00616, 0.00617, 0.00619, 0.00621, 0.00623, 0.00624, 0.00626, 0.00628, 0.00630, 0.00632, 0.00633, 0.00635, 0.00637, 0.00639, 0.00641, 0.00643, 0.00644, 0.00646, 0.00648, 0.00650, 0.00652, 0.00654, 0.00656, 0.00658, 0.00660, 0.00662, 0.00664, 0.00666, 0.00668, 0.00670, 0.00672, 0.00674, 0.00676, 0.00678, 0.00680, 0.00682, 0.00684, 0.00686, 0.00689, 0.00691, 0.00693, 0.00695, 0.00697, 0.00699, 0.00702, 0.00704, 0.00706, 0.00708, 0.00711, 0.00713, 0.00715, 0.00718, 0.00720, 0.00722, 0.00725, 0.00727, 0.00729, 0.00732, 0.00734, 0.00737, 0.00739, 0.00742, 0.00744, 0.00747, 0.00749, 0.00752, 0.00754, 0.00757, 0.00760, 0.00762, 0.00765, 0.00767, 0.00770, 0.00773, 0.00775, 0.00778, 0.00781, 0.00784, 0.00787, 0.00789, 0.00792, 0.00795, 0.00798, 0.00801, 0.00804, 0.00807, 0.00810, 0.00813, 0.00816, 0.00819, 0.00822, 0.00825, 0.00828, 0.00831, 0.00834, 0.00837, 0.00840, 0.00844, 0.00847, 0.00850, 0.00853, 0.00857, 0.00860, 0.00863, 0.00867, 0.00870, 0.00874, 0.00877, 0.00881, 0.00884, 0.00888, 0.00891, 0.00895, 0.00898, 0.00902, 0.00906, 0.00909, 0.00913, 0.00917, 0.00921, 0.00925, 0.00929, 0.00933, 0.00936, 0.00940, 0.00944, 0.00949, 0.00953, 0.00957, 0.00961, 0.00965, 0.00969, 0.00974, 0.00978, 0.00982, 0.00987, 0.00991, 0.00996, 0.01000, 0.01005, 0.01009, 0.01014, 0.01018, 0.01023, 0.01028, 0.01033, 0.01038, 0.01042, 0.01047, 0.01052, 0.01057, 0.01063, 0.01068, 0.01073, 0.01078, 0.01083, 0.01089, 0.01094, 0.01100, 0.01105, 0.01111, 0.01116, 0.01122, 0.01128, 0.01133, 0.01139, 0.01145, 0.01151, 0.01157, 0.01163, 0.01169, 0.01176, 0.01182, 0.01188, 0.01195, 0.01201, 0.01208, 0.01214, 0.01221, 0.01228, 0.01235, 0.01242, 0.01249, 0.01256, 0.01263, 0.01270, 0.01277, 0.01285, 0.01292, 0.01300, 0.01308, 0.01316, 0.01323, 0.01331, 0.01339, 0.01348, 0.01356, 0.01364, 0.01373, 0.01381, 0.01390, 0.01399, 0.01408, 0.01417, 0.01426, 0.01435, 0.01445, 0.01454, 0.01464, 0.01473, 0.01483, 0.01493, 0.01503, 0.01514, 0.01524, 0.01535, 0.01546, 0.01556, 0.01567, 0.01579, 0.01590, 0.01602, 0.01613, 0.01625, 0.01637, 0.01649, 0.01662, 0.01674, 0.01687, 0.01700, 0.01713, 0.01727, 0.01740, 0.01754, 0.01768, 0.01782, 0.01797, 0.01812, 0.01827, 0.01842, 0.01857, 0.01873, 0.01889, 0.01905, 0.01922, 0.01939, 0.01956, 0.01974, 0.01991, 0.02009, 0.02028, 0.02047, 0.02066, 0.02085, 0.02105, 0.02125, 0.02146, 0.02167, 0.02189, 0.02211, 0.02233, 0.02256, 0.02279, 0.02303, 0.02327, 0.02352, 0.02377, 0.02403, 0.02429, 0.02456, 0.02484, 0.02512, 0.02541, 0.02571, 0.02601, 0.02632, 0.02664, 0.02696, 0.02730, 0.02764, 0.02799, 0.02835, 0.02871, 0.02909, 0.02948, 0.02988, 0.03029, 0.03071, 0.03114, 0.03159, 0.03205, 0.03252, 0.03301, 0.03351, 0.03402, 0.03455, 0.03510, 0.03567, 0.03626, 0.03686, 0.03749, 0.03813, 0.03880, 0.03950, 0.04022, 0.04096, 0.04174, 0.04254, 0.04338, 0.04425, 0.04515, 0.04609, 0.04708, 0.04810, 0.04917, 0.05029, 0.05146, 0.05269, 0.05398, 0.05533, 0.05675, 0.05825, 0.05983, 0.06150, 0.06326, 0.06512, 0.06710, 0.06920, 0.07144, 0.07383, 0.07639, 0.07912, 0.08206, 0.08523, 0.08865, 0.09236, 0.09639, 0.10079, 0.10561, 0.11092, 0.11678, 0.12330, 0.13060, 0.13880, 0.14812, 0.15876, 0.17106, 0.18543, 0.20243, 0.22286, 0.24788, 0.27923, 0.31965, 0.37377, 0.44994, 0.56510, 0.75954 };
const vector<int> widths = { 1023, 1019, 1017, 1015, 1013, 1011, 1009, 1005, 1003, 1001, 999, 997, 995, 993, 991, 987, 985, 983, 981, 979, 977, 975, 973, 971, 969, 967, 965, 963, 961, 959, 955, 953, 951, 949, 947, 945, 943, 941, 939, 937, 935, 933, 931, 929, 927, 925, 923, 921, 919, 917, 915, 913, 911, 909, 907, 905, 903, 901, 899, 897, 895, 893, 891, 889, 887, 885, 883, 881, 879, 877, 875, 873, 871, 869, 867, 865, 863, 861, 859, 857, 855, 853, 851, 849, 847, 845, 843, 841, 839, 837, 835, 833, 831, 829, 827, 825, 823, 821, 819, 817, 815, 813, 811, 809, 807, 805, 803, 801, 799, 797, 795, 793, 791, 789, 787, 785, 783, 781, 779, 777, 775, 773, 771, 769, 767, 765, 763, 761, 759, 757, 755, 753, 751, 749, 747, 745, 743, 741, 739, 737, 735, 733, 731, 729, 727, 725, 723, 721, 719, 717, 715, 713, 711, 709, 707, 705, 703, 701, 699, 697, 695, 693, 691, 689, 687, 685, 683, 681, 679, 677, 675, 673, 671, 669, 667, 665, 663, 661, 659, 657, 655, 653, 651, 649, 647, 645, 643, 641, 639, 637, 635, 633, 631, 629, 627, 625, 623, 621, 619, 617, 615, 613, 611, 609, 607, 605, 603, 601, 599, 597, 595, 593, 591, 589, 587, 585, 583, 581, 579, 577, 575, 573, 571, 569, 567, 565, 563, 561, 559, 557, 555, 553, 551, 549, 547, 545, 543, 541, 539, 537, 535, 533, 531, 529, 527, 525, 523, 521, 519, 517, 515, 513, 511, 509, 507, 505, 503, 501, 499, 497, 495, 493, 491, 489, 487, 485, 483, 481, 479, 477, 475, 473, 471, 469, 467, 465, 463, 461, 459, 457, 455, 453, 451, 449, 447, 445, 443, 441, 439, 437, 435, 433, 431, 429, 427, 425, 423, 421, 419, 417, 415, 413, 411, 409, 407, 405, 403, 401, 399, 397, 395, 393, 391, 389, 387, 385, 383, 381, 379, 377, 375, 373, 371, 369, 367, 365, 363, 361, 359, 357, 355, 353, 351, 349, 347, 345, 343, 341, 339, 337, 335, 333, 331, 329, 327, 325, 323, 321, 319, 317, 315, 313, 311, 309, 307, 305, 303, 301, 299, 297, 295, 293, 291, 289, 287, 285, 283, 281, 279, 277, 275, 273, 271, 269, 267, 265, 263, 261, 259, 257, 255, 253, 251, 249, 247, 245, 243, 241, 239, 237, 235, 233, 231, 229, 227, 225, 223, 221, 219, 217, 215, 213, 211, 209, 207, 205, 203, 201, 199, 197, 195, 193, 191, 189, 187, 185, 183, 181, 179, 177, 175, 173, 171, 169, 167, 165, 163, 161, 159, 157, 155, 153, 151, 149, 147, 145, 143, 141, 139, 137, 135, 133, 131, 129, 127, 125, 123, 121, 119, 117, 115, 113, 111, 109, 107, 105, 103, 101, 99, 97, 95, 93, 91, 89, 87, 85, 83, 81, 79, 77, 75, 73, 71, 69, 67, 65, 63, 61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7 };

class RandomDataGenerator
{
    vector<RandValuation> randValuations;
    vector<RandYieldCurve> randYieldCurves;
    long totalValuations;
    long totalYieldCurves;
    bool noCashflows;
    bool onlyEuropean;
    int skewPercent;
    unsigned int randomSeed;
    string outputPath;

public:
    RandomDataGenerator(long totalValuations, long totalYieldCurves, string outputPath, int skewPercent, bool noCashflows, bool onlyEuropean)
        : totalValuations(totalValuations), totalYieldCurves(totalYieldCurves), outputPath(outputPath), skewPercent(skewPercent), noCashflows(noCashflows), onlyEuropean(onlyEuropean)
    {
        distributeYieldCurves();
    }

    void distribute_0()
    {
        long currentNumOptions = 0;

        while (currentNumOptions < totalValuations)
        {
            RandValuation val(9, 12, 0.1, false, randYieldCurves.size());
            addOption(val, currentNumOptions);
        }

        writeDataToFile("0_UNIFORM_");
    }

    void distribute_1()
    {
        long currentNumOptions = 0;

        while (currentNumOptions < totalValuations)
        {
            real maturity = (real)randIntInRange(1, 100);
            real reversionRate = getRRByWidthRange(7, 511);
            RandValuation val(maturity, 12, reversionRate, false, randYieldCurves.size());
            addOption(val, currentNumOptions);
        }

        writeDataToFile("1_RAND_");
    }

    void distribute_2()
    {
        long currentNumOptions = 0;

        while (currentNumOptions < totalValuations)
        {
            real reversionRate = getRRByWidthRange(7, 511);
            RandValuation o(9, 12, reversionRate, false, randYieldCurves.size());
            addOption(o, currentNumOptions);
        }

        writeDataToFile("2_RANDCONSTHEIGHT_");
    }

    void distribute_3()
    {
        long currentNumOptions = 0;

        while (currentNumOptions < totalValuations)
        {
            int maturity = randIntInRange(1, 100);
            RandValuation o(maturity, 12, 0.1, false, randYieldCurves.size());
            addOption(o, currentNumOptions);
        }

        writeDataToFile("3_RANDCONSTWIDTH_");
    }

    void distribute_4()
    {
        long currentNumOptions = 0;

        while (currentNumOptions < (skewPercent / (real)100) * totalValuations)
        {
            int maturity = randIntInRange(70, 100);
            real reversionRate = getRRByWidthRange(411, 511);
            RandValuation o(maturity, 12, reversionRate, true, randYieldCurves.size());
            addOption(o, currentNumOptions);
        }

        const long currentNumSkewOptions = currentNumOptions;

        while (currentNumOptions < totalValuations)
        {
            int maturity = randIntInRange(1, 30);
            real reversionRate = getRRByWidthRange(7, 107);
            RandValuation o(maturity, 12, reversionRate, false, randYieldCurves.size());
            addOption(o, currentNumOptions);
        }

        writeDataToFile("4_SKEWED_" + to_string(skewPercent) + "_");
    }

    void distribute_5()
    {
        long currentNumOptions = 0;

        while (currentNumOptions < (skewPercent / (real)100) * totalValuations)
        {
            // int maturity = randIntInRange(1, 100);
            real reversionRate_skew = getRRByWidthRange(7, 107);
            RandValuation o(100, 12, reversionRate_skew, true, randYieldCurves.size());
            addOption(o, currentNumOptions);
        }

        const long currentNumSkewOptions = currentNumOptions;

        while (currentNumOptions < totalValuations)
        {
            int maturity = randIntInRange(1, 30);
            real reversionRate = getRRByWidthRange(7, 107);
            RandValuation o(maturity, 12, reversionRate, false, randYieldCurves.size());
            addOption(o, currentNumOptions);
        }

        writeDataToFile("5_SKEWEDCONSTHEIGHT_" + to_string(skewPercent) + "_");
    }

    void distribute_6()
    {
        long currentNumOptions = 0;

        real reversionRate_skew = getRRByWidthRange(509, 511);
        while (currentNumOptions < (skewPercent / (real)100) * totalValuations)
        {
            int maturity = randIntInRange(1, 30);

            RandValuation o(maturity, 12, reversionRate_skew, true, randYieldCurves.size());
            addOption(o, currentNumOptions);
        }

        const long currentNumSkewOptions = currentNumOptions;

        while (currentNumOptions < totalValuations)
        {
            int maturity = randIntInRange(1, 30);
            real reversionRate = getRRByWidthRange(7, 107);
            RandValuation o(maturity, 12, reversionRate, false, randYieldCurves.size());
            addOption(o, currentNumOptions);
        }

        writeDataToFile("6_SKEWEDCONSTWIDTH_" + to_string(skewPercent) + "_");
    }

private:
    real getRRByWidthRange(int a, int b)
    {
        int randWidth = randIntInRange(a, b);
        auto it = find(widths.begin(), widths.end(), randWidth);

        while (it == widths.end())
        {
            randWidth = randIntInRange(a, b);
            it = find(widths.begin(), widths.end(), randWidth);
        }

        auto index = std::distance(widths.begin(), it);
        return reversionRates.at(index);
    }

    void addOption(const RandValuation v, long &currentNumOptions)
    {
        randValuations.push_back(v);
        currentNumOptions++;
    }

    int getNumSkewed(const int x, const RandValuation y) { return y.Skewed ? x + 1 : x; }

    void writeDataToFile(string filename)
    {
        auto rng = default_random_engine{};
        shuffle(begin(randValuations), end(randValuations), rng);

        Valuations valuations(randValuations.size());
        for (int i = 0; i < randValuations.size(); i++)
        {
            valuations.TermUnits.push_back(randValuations.at(i).TermUnit);
            valuations.TermSteps.push_back(randValuations.at(i).TermStep);
            valuations.MeanReversionRates.push_back(randValuations.at(i).MeanReversionRate);
            valuations.Volatilities.push_back(randValuations.at(i).Volatility);

            valuations.Maturities.push_back(randValuations.at(i).Maturity);
            valuations.Cashflows.push_back(randValuations.at(i).Cashflows);
            valuations.CashflowSteps.insert(valuations.CashflowSteps.end(), randValuations.at(i).CashflowSteps.begin(), randValuations.at(i).CashflowSteps.end());
            valuations.Repayments.insert(valuations.Repayments.end(), randValuations.at(i).Repayments.begin(), randValuations.at(i).Repayments.end());
            valuations.Coupons.insert(valuations.Coupons.end(), randValuations.at(i).Coupons.begin(), randValuations.at(i).Coupons.end());
            valuations.Spreads.push_back(randValuations.at(i).Spread);

            valuations.OptionTypes.push_back(randValuations.at(i).OptionType);
            valuations.StrikePrices.push_back(randValuations.at(i).StrikePrice);
            valuations.FirstExerciseSteps.push_back(randValuations.at(i).FirstExerciseStep);
            valuations.LastExerciseSteps.push_back(randValuations.at(i).LastExerciseStep);
            valuations.ExerciseStepFrequencies.push_back(randValuations.at(i).ExerciseStepFrequency);

            valuations.YieldCurveIndices.push_back(randValuations.at(i).YieldCurveIndex);
        }

        for (int i = 0; i < randYieldCurves.size(); i++)
        {
            valuations.YieldCurveTerms.push_back(randYieldCurves.at(i).YieldCurveTermCount);
            valuations.YieldCurveRates.insert(valuations.YieldCurveRates.end(), randYieldCurves.at(i).YieldCurveRates.begin(), randYieldCurves.at(i).YieldCurveRates.end());
            valuations.YieldCurveTimeSteps.insert(valuations.YieldCurveTimeSteps.end(), randYieldCurves.at(i).YieldCurveTimeSteps.begin(), randYieldCurves.at(i).YieldCurveTimeSteps.end());
        }

        filename.append(to_string(randValuations.size()));
        if (noCashflows) filename.append("_zero");
        if (onlyEuropean) filename.append("_EU");
        string dataFile = outputPath + filename + ".in";
        valuations.writeToFile(dataFile);
        cout << "finished writing to " << dataFile << endl;
    }

    void distributeYieldCurves()
    {
        long currentNumYieldCurves = 0;
        while (currentNumYieldCurves < totalYieldCurves)
        {
            RandYieldCurve yc{};
            randYieldCurves.push_back(yc);
            currentNumYieldCurves++;
        }
    }
};

int main(int argc, char *argv[])
{
    int dataType;
    long totalValuations;
    long totalYieldCurves;
    bool noCashflows;
    bool onlyEuropean;
    int skewPercent;
    unsigned int randomSeed;
    string outputPath;

#ifdef USE_GETOPT_ARGS
    dataType = -1;
    totalValuations = 1;
    totalYieldCurves = 1;
    skewPercent = 1;
    randomSeed = 12345;
    outputPath = "../data/";

    GetOpt::GetOpt_pp cmd(argc, argv);
    cmd >> GetOpt::Option('t', "dataType", dataType);
    cmd >> GetOpt::Option('n', "totalValuations", totalValuations);
    cmd >> GetOpt::Option('y', "totalYieldCurves", totalYieldCurves);
    cmd >> GetOpt::Option('c', "noCashflows", noCashflows);
    cmd >> GetOpt::Option('e', "onlyEuropean", onlyEuropean);
    cmd >> GetOpt::Option('s', "skewPercent", skewPercent);
    cmd >> GetOpt::Option('r', "randomSeed", randomSeed);
    cmd >> GetOpt::Option('p', "outputPath", outputPath);

#else
    cxxopts::Options options(argv[0], " - Random Valuation Dataset Generator");
    options
        .positional_help("[optional args]")
        .show_positional_help();

    options
        .add_options()
        ("h,help", "Print help")
        ("t,dataType", "Dataset type, options: 0: uniform, 1: random, 2: random fixed height, 3: random fixed width, 4: skewed, 5: skewed fixed height, 6: skewed fixed width", cxxopts::value<int>(dataType)->default_value("-1"))
        ("n,totalValuations", "Number of valuation", cxxopts::value<long>(totalValuations)->default_value("1"))
        ("y,totalYieldCurves", "Number of yield curves", cxxopts::value<long>(totalYieldCurves)->default_value("1"))
        ("c,noCashflows", "Dataset will only comprise of zero-coupon bonds", cxxopts::value<bool>(noCashflows)->default_value("false"))
        ("e,onlyEuropean", "Only European optionality on bonds", cxxopts::value<bool>(onlyEuropean)->default_value("false"))
        ("s,skewPercent", "Percentage of skewed valuations", cxxopts::value<int>(skewPercent)->default_value("1"))
        ("r,randomSeed", "Seed for Random Number Generator", cxxopts::value<unsigned int>(randomSeed)->default_value("12345"))
        ("p,outputPath", "Output directory", cxxopts::value<string>(outputPath)->default_value("../data/"))
        ;

    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help({ "", "Group" }) << std::endl;
        exit(0);
    }
#endif
    srand(randomSeed);

    RandomDataGenerator rdg(totalValuations, totalYieldCurves, outputPath, skewPercent, noCashflows, onlyEuropean);

    switch (dataType)
    {
    case 0:
        rdg.distribute_0();
        break;
    case 1:
        rdg.distribute_1();
        break;
    case 2:
        rdg.distribute_2();
        break;
    case 3:
        rdg.distribute_3();
        break;
    case 4:
        rdg.distribute_4();
        break;
    case 5:
        rdg.distribute_5();
        break;
    case 6:
        rdg.distribute_6();
        break;
    default:
        // if out of range - just print them all
        rdg.distribute_0();
        rdg.distribute_1();
        rdg.distribute_2();
        rdg.distribute_3();
        rdg.distribute_4();
        rdg.distribute_5();
        rdg.distribute_6();
    }

    return 0;
}