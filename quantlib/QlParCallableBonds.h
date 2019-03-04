#pragma once

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
//#include <ql/time/daycounters/thirty360.hpp>

#include <chrono>
#include <limits>
#include <vector>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "../common/ValuationConstants.hpp"
#include "../common/Domain.hpp"
#include "../common/progress-cpp/ProgressBar.hpp"

using namespace std;
using namespace QuantLib;
using namespace trinom;

namespace qlpar
{
    ext::shared_ptr<YieldTermStructure>
        flatRate(const Date& today,
            const ext::shared_ptr<Quote>& forward,
            const DayCounter& dc,
            const Compounding& compounding,
            const Frequency& frequency) {
        return ext::shared_ptr<YieldTermStructure>(
            new FlatForward(today,
                Handle<Quote>(forward),
                dc,
                compounding,
                frequency));
    }

    ext::shared_ptr<YieldTermStructure>
        flatRate(const Date& today,
            Rate forward,
            const DayCounter& dc,
            const Compounding &compounding,
            const Frequency &frequency) {
        return flatRate(today,
            ext::shared_ptr<Quote>(new SimpleQuote(forward)),
            dc,
            compounding,
            frequency);
    }

    /* Bloomberg OAS1: "N" model (Hull White)
    varying volatility parameter

    The curve entered into Bloomberg OAS1 is a flat curve,
    at constant yield = 5.5%, semiannual compounding.
    Assume here OAS1 curve uses an ACT/ACT day counter,
    as documented in PFC1 as a "default" in the latter case.
    */
    Real computeValuation(Integer maturityDays, Real meanReversionRate, Real volatility, Real strike, Integer gridIntervals)
    {
        Date today = Date(16, October, 2007);
        Settings::instance().evaluationDate() = today;

        // set up a flat curve corresponding to Bloomberg flat curve
        Rate bbCurveRate = 0.0501772;
        DayCounter bbDayCounter = ActualActual(ActualActual::Bond);
        //DayCounter bbDayCounter = Thirty360(Thirty360::BondBasis);
        InterestRate bbIR(bbCurveRate, bbDayCounter, Compounded, Semiannual);

        Handle<YieldTermStructure> termStructure(flatRate(today,
            bbIR.rate(),
            bbIR.dayCounter(),
            bbIR.compounding(),
            bbIR.frequency()));

        // set up the call schedule
        CallabilitySchedule callSchedule;
        Real callPrice = strike;
        //Size numberOfCallDates = 24;
        Date callDate = today + 180;

        Date maturity = today + maturityDays;

        //for (Size i = 0; i < numberOfCallDates; i++) {
        while (callDate < maturity) {
            Calendar nullCalendar = NullCalendar();

            Callability::Price myPrice(callPrice,
                Callability::Price::Clean);
            callSchedule.push_back(
                ext::make_shared<Callability>(
                    myPrice,
                    Callability::Call,
                    callDate));
            callDate = nullCalendar.advance(callDate, 3, Months);
        }

        // set up the callable bond
        Date dated = Date(16, September, 2004);
        Date issue = dated;
        Natural settlementDays = 3;  // Bloomberg OAS1 settle is Oct 19, 2007
        Calendar bondCalendar = UnitedStates(UnitedStates::GovernmentBond);
        Real coupon = .0465;
        Frequency frequency = Quarterly;
        Real redemption = 100.0;
        Real faceAmount = 100.0;

        /* The 30/360 day counter Bloomberg uses for this bond cannot
           reproduce the US Bond/ISMA (constant) cashflows used in PFC1.
           Therefore use ActAct(Bond)
        */
        DayCounter bondDayCounter = ActualActual(ActualActual::Bond);

        // PFC1 shows no indication dates are being adjusted
        // for weekends/holidays for vanilla bonds
        BusinessDayConvention accrualConvention = Unadjusted;
        BusinessDayConvention paymentConvention = Unadjusted;

        Schedule sch(dated, maturity, Period(frequency), bondCalendar,
            accrualConvention, accrualConvention,
            DateGeneration::Backward, false);

        Size maxIterations = 1000;
        Real accuracy = 1e-8;
        //Integer gridIntervals = 40;
        Real reversionParameter = meanReversionRate;

        Real sigma = volatility; // core dumps if zero on Cygwin

        ext::shared_ptr<ShortRateModel> hw0(
            new HullWhite(termStructure, reversionParameter, sigma));

        ext::shared_ptr<PricingEngine> engine0(
            new TreeCallableFixedRateBondEngine(hw0, gridIntervals));

        CallableFixedRateBond callableBond(settlementDays, faceAmount, sch,
            vector<Rate>(1, coupon),
            bondDayCounter, paymentConvention,
            redemption, issue, callSchedule);
        callableBond.setPricingEngine(engine0);

        //cout << setprecision(2)
        //    << showpoint
        //    << fixed
        //    << "sigma/vol (%) = "
        //    << 100.*sigma
        //    << endl;

        auto price = callableBond.cleanPrice();
        //auto yield = 100. * callableBond.yield(bondDayCounter,
        //    Compounded,
        //    frequency,
        //    accuracy,
        //    maxIterations);

        //cout << ".";
        //cout << "QuantLib price/yld (%)  ";
        //cout << price << " / " << yield << endl;

        return price;
    }

    void computeValuations(const Valuations &valuations, std::vector<trinom::real> &results)
    {
        //ProgressBar progressBar(valuations.ValuationCount, 70, '#', '-');
#pragma omp parallel for
        for (auto i = 0; i < valuations.ValuationCount; ++i)
        {
            //++progressBar; // record the tick

            auto maturityDays = (int)ceil(valuations.Maturities[i] * 12 * 30); // days
            auto meanReversionRate = valuations.MeanReversionRates[i];
            auto volatility = valuations.Volatilities[i];
            auto strike = valuations.StrikePrices[i];
            auto gridIntervals = (int)(valuations.Maturities[i] * valuations.TermSteps[i]);

            auto result = qlpar::computeValuation(maturityDays, meanReversionRate, volatility, strike, gridIntervals);
            results[i] = result;

            //cout << ".";

            // display the bar only at certain steps
            //if (i % 10 == 0)
            //    progressBar.display();
        }

        cout << endl;

        // tell the bar to finish
        //progressBar.done();
    }

} // namespace qlpar
