-- ==
-- compiled input @ ../data/fut/1_RAND.in
-- compiled input @ ../data/fut/4_SKEWED.in

import "header64"

-------------------------------------------------------------
--- Follows code independent of the instantiation of real ---
-------------------------------------------------------------
let zero = i2r 0
let one = i2r 1
let two = i2r 2
let half = one / two
let three = i2r 3
let six = i2r 6
let seven = i2r 7
let year = i2r 365
let hundred = i2r 100

-----------------------
--- Data Structures ---
-----------------------

type YieldCurveData = {
    P : real    -- Discount Factor function, P(t) gives the yield (interest rate) for a specific period of time t
  , t : i32     -- Time [days]
}

type TOptionData = {
    StrikePrice                 : real   -- X
  , Maturity                    : real   -- T, [years]
  , Length                      : real   -- t, [years]
  , ReversionRateParameter      : real   -- a, parameter in HW-1F model
  , VolatilityParameter         : real   -- sigma, volatility parameter in HW-1F model
  , TermUnit                    : u16
  , TermStepCount               : u16
  , OptionType                  : i8     -- option type, [0 - Put | 1 - Call]
}

let OptionType_CALL = i8.i32 0
let OptionType_PUT = i8.i32 1

-- | As `map5`@term, but with three more arrays.
let map8 'a 'b 'c 'd 'e 'f 'g 'h [n] 'x (i: a -> b -> c -> d -> e -> f -> g -> h -> x) (as: [n]a) (bs: [n]b) (cs: [n]c) (ds: [n]d) (es: [n]e) (fs: [n]f) (gs: [n]g) (hs: [n]h): *[n]x =
        map (\((a, b, c, d), (e, f, g, h)) -> i a b c d e f g h) (zip (zip4 as bs cs ds) (zip4 es fs gs hs))

let computeWH (optionData : TOptionData) : (i32,i32) =
     let T  = optionData.Maturity
     let termUnit = ui2r optionData.TermUnit
     let termUnitsInYearCount = r2i (r_ceil(year / termUnit))
     let dt = (i2r termUnitsInYearCount) / (ui2r optionData.TermStepCount)
     let n = r2i ((ui2r optionData.TermStepCount) * (i2r termUnitsInYearCount) * T)
     let a = optionData.ReversionRateParameter
     let M  = (r_exp (zero - a*dt)) - one
     let jmax = r2i (- 0.184 / M) + 1
     let Qlen = 2 * jmax + 1
     in  (Qlen, n)

let simpleFops (single: bool) (hwd: i64) : i64 = 1
--  if single then 1
--  else if hwd == 1
--       then 2 -- P100   Compute 6
--       else 3 -- GTX780 Compute 3.5

let specialFops (single: bool) (hwd: i64) : i64 =
  (simpleFops single hwd) *
  if hwd == 1
  then 1 -- P100   Compute 6
  else 1 -- GTX690 Compute 3.5

let mopsPerOption (single: bool) (w: i64) (h: i64) : i64 =
  let header = 8
  let getYield = 6
  let fwdHelper= 3
  let bwdHelper= 3
  let computeQ = w + 1 + h + 1 +
                 h*(1 + w + w*fwdHelper + w + getYield + 1)
  let computeCall = w + h * (1 + w*bwdHelper) + 1
  let total = header + computeQ + computeCall
  in  if single then total else 2*total






--let fopsPerOption (single: bool) (hwd: i64) (w: i64) (h: i64) : i64 =
--  let simple = simpleFops  single hwd
--  let special= specialFops single hwd
--  let header      = 28 * simple + 2*special
--  let getYield    = 6  * simple
--  let fwdHelper   = 52 * simple
--  let bwdHelper   = 47 * simple + special
--  let computeQ    = h * ( w*(12*simple + 2*special + fwdHelper) + getYield + 7*simple +2*special )
--  let computeCall = h * ( 2*simple + w*bwdHelper )
--  in  header + computeQ + computeCall

let fopsPerOption (single: bool) (hwd: i64) (w: i64) (h: i64) : i64 =
  let simple = simpleFops  single hwd
  let special= specialFops single hwd
  let header      = 20 * simple + 2*special -- 2 adds, 4 subs, 4 divs, 10 muls, 2 exps
  let getYield    = 4  * simple -- 1 add, 1 sub, 1 mul, 1 div
  let fwdHelper   = 52 * simple
  let bwdHelper   = 47 * simple + special
  let computeQ    = h * ( w*(12*simple + 2*special + fwdHelper) + getYield + 7*simple +2*special )
  let computeCall = h * ( 2*simple + w*bwdHelper )
  in  header + computeQ + computeCall

let main [q] [y] (strikes           : [q]real)
                 (maturities        : [q]real) 
                 (lenghts           : [q]real)
                 (termunits         : [q]u16 ) 
                 (termstepcounts    : [q]u16 ) 
                 (rrps              : [q]real) 
                 (vols              : [q]real) 
                 (types             : [q]i8)
                 (yield_p           : [y]real)
                 (yield_t           : [y]i32)
               : (i64, i64, i64, i64, i64, i64) =
    let options = map8 (\s m l u c r v t -> {StrikePrice=s, Maturity=m, Length=l, TermUnit=u, TermStepCount=c,
                                        ReversionRateParameter=r, VolatilityParameter=v, OptionType=t }
                ) strikes maturities lenghts termunits termstepcounts rrps vols types

    let (ws0, hs0) = unzip (map computeWH options)
    let ws = map (i64.i32) ws0
    let hs = map (i64.i32) hs0
    
    -- hwd == 1 => P100
    let fops_single_p100 = map2 (fopsPerOption true  1) ws hs |> reduce (+) 0i64
    let fops_double_p100 = map2 (fopsPerOption false 1) ws hs |> reduce (+) 0i64

    -- hwd == 2 => GTX690 (or anything different than 1)
    let fops_single_gtx690 = map2 (fopsPerOption true  2) ws hs |> reduce (+) 0i64
    let fops_double_gtx690 = map2 (fopsPerOption false 2) ws hs |> reduce (+) 0i64

    let mops_single = map2 (mopsPerOption true ) ws hs |> reduce (+) 0i64
    let mops_double = map2 (mopsPerOption false) ws hs |> reduce (+) 0i64

    in  ( fops_single_p100, fops_double_p100
        , fops_single_gtx690, fops_double_gtx690
        , mops_single, mops_double
        )
