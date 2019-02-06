#include <chrono>
#include <limits>

#include "../common/Args.hpp"
#include "../common/Arrays.hpp"
#include "Seq.hpp"

using namespace std;
using namespace chrono;
using namespace trinom;

void computeAllOptions(const Args &args)
{
    // Read options from filename
    Valuations options(args.valuations);
    YieldCurves yield(args.yield);

    if (args.test)
    {
        cout << "Sequential implementation" << endl;
    }

    if (args.runs > 0)
    {
        if (args.test)
        {
            cout << "Performing " << args.runs << " runs..." << endl;
        }
        long long best = std::numeric_limits<long long>::max();
        for (auto i = 0; i < args.runs; ++i)
        {
            vector<real> results;
            results.resize(valuations.ValuationCount);

            auto time_begin = steady_clock::now();
            seq::computeOptions(valuations, results);
            auto time_end = steady_clock::now();
            auto runtime = duration_cast<microseconds>(time_end - time_begin).count();
            if (runtime < best)
            {
                best = runtime;
            }
        }
        if (args.test)
        {
            cout << "Best times: total " << best << " microsec." << endl;
        }
        else
        {
            cout << "-,-,-,-," << best << ",-" << endl;
        }
    }
    else
    {
        vector<real> results;
        results.resize(valuations.ValuationCount);

        auto time_begin = steady_clock::now();
        seq::computeOptions(valuations, results);
        auto time_end = steady_clock::now();
        auto runtime = duration_cast<microseconds>(time_end - time_begin).count();

        if (!args.test)
        {
            Arrays::write_array(cout, results);
        }
        else
        {
            cout << "Total time " << runtime << " microsec." << endl;
        }
    }
}

int main(int argc, char *argv[])
{
    Args args(argc, argv);

    computeAllOptions(args);

    return 0;
}
