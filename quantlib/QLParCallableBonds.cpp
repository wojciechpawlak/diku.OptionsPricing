/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*!
 Copyright (C) 2008 Allen Kuo

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <http://quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
 */

 /* This example sets up a callable fixed rate bond with a Hull White pricing
    engine and compares to Bloomberg's Hull White price/yield calculations.
 */

#include <ql/qldefines.hpp>
#ifdef BOOST_MSVC
#  include <ql/auto_link.hpp>
#endif
#include <ql/experimental/callablebonds/callablebond.hpp>
#include <ql/experimental/callablebonds/treecallablebondengine.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/daycounters/actualactual.hpp>

#include <chrono>
#include <limits>
#include <vector>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "../common/Args.hpp"
#include "../common/Arrays.hpp"
#include "QlParCallableBonds.h"

using namespace std;
using namespace chrono;
using namespace QuantLib;
using namespace trinom;

#if defined(QL_ENABLE_SESSIONS)
namespace QuantLib {
    Integer sessionId() { return 0; }
}
#endif

namespace qlpar
{

    void computeAllValuations(const Args &args)
    {
        try {
            //boost::timer timer;
            //cout << endl;
            //cout << "Pricing a callable fixed rate bond using" << endl;
            //cout << "Hull White model w/ reversion parameter = 0.03" << endl;
            //cout << "BAC4.65 09/15/12  ISIN: US06060WBJ36" << endl;
            //cout << "roughly five year tenor, ";
            //cout << "quarterly coupon and call dates" << endl;
            //cout << "reference date is : " << today << endl << endl;

            // Read options from filename
            Valuations valuations(args.valuations);

            if (args.test)
            {
                cout << "QuantLib Parallel implementation" << endl;
                cout << args.valuations << endl;
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
                    vector<trinom::real> results;
                    results.resize(valuations.ValuationCount);

                    auto time_begin = steady_clock::now();
                    qlpar::computeValuations(valuations, results);
                    auto time_end = steady_clock::now();
                    auto runtime = duration_cast<microseconds>(time_end - time_begin).count();
                    if (runtime < best)
                    {
                        best = runtime;
                    }
                }


                const trinom::real elapsedTotalSec = best / 1000000.0;
                const size_t valsPerSec = (int)round(valuations.ValuationCount / elapsedTotalSec);
                const size_t timeFor100000 = (int)round(100000 / valsPerSec);

                if (args.test)
                {
                    cout << "Best times: total " << best << " microsec." << endl;
                }
                else
                {
                    cout << "-,-,-,-," << best << ",-," << valsPerSec << ',' << timeFor100000 << endl;
                }
            }
            else
            {
                vector<trinom::real> results;
                results.resize(valuations.ValuationCount);

                auto time_begin = steady_clock::now();
                qlpar::computeValuations(valuations, results);
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

            //double seconds = timer.elapsed();
            //Integer hours = int(seconds / 3600);
            //seconds -= hours * 3600;
            //Integer minutes = int(seconds / 60);
            //seconds -= minutes * 60;
            //cout << " \nRun completed in ";
            //if (hours > 0)
            //    cout << hours << " h ";
            //if (hours > 0 || minutes > 0)
            //    cout << minutes << " m ";
            //cout << fixed << setprecision(0)
            //    << seconds << " s\n" << endl;
        }
        catch (std::exception& e) {
            std::cerr << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "unknown error" << std::endl;
        }
    }

} // namespace qlpar

int main(int argc, char *argv[])
{
    trinom::Args args(argc, argv);

    qlpar::computeAllValuations(args);

    return 0;
}

