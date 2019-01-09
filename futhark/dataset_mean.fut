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
        map (\(a, b, c, d, e, f, g, h) -> i a b c d e f g h) (zip8 as bs cs ds es fs gs hs)

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

let decideSkewness (nf : f32) (ws : []i32) : (f32, f32, f32) =
    let dfs = map (\w -> (r32 w)/nf) ws
    let m_w = reduce (+) 0f32 dfs
    let (mr_ws, ones) = unzip <| map (\w -> if r32 w < 2*m_w then (r32 w, 1.0f32) else (0.0f32, 0.0f32)) ws
    let n_r  = reduce (+) 0.0f32 ones
    let mr_w1 = reduce (+) 0.0f32 mr_ws

    let (ms_ws, ones) = unzip <| map (\w -> if r32 w >= 2*m_w then (r32 w, 1.0f32) else (0.0f32, 0.0f32)) ws
    let n_s  = reduce (+) 0.0f32 ones
    let ms_w1 = reduce (+) 0.0f32 ms_ws
    
    let mr_w = if n_r > 0 then mr_w1 / n_r else 0
    let ms_w = if n_s > 0 then ms_w1 / n_s else mr_w

    in (mr_w, ms_w, n_r / nf)


let main [q] [y] (strikes           : [q]real)
                 (maturities        : [q]real) 
                 (lenghts           : [q]real)
                 (termunits         : [q]u16 ) 
                 (termstepcounts    : [q]u16 ) 
                 (rrps              : [q]real) 
                 (vols              : [q]real) 
                 (types             : [q]i8)
                 (yield_p           : [y]real)
                 (yield_t           : [y]i32) : ( f32, f32, f32, f32, f32, f32 ) =
        
    let yield = map2 (\p d -> {P = p, t = d}) yield_p yield_t

    let options = map8 (\s m l u c r v t -> {StrikePrice=s, Maturity=m, Length=l, TermUnit=u, TermStepCount=c,
                                        ReversionRateParameter=r, VolatilityParameter=v, OptionType=t }
                ) strikes maturities lenghts termunits termstepcounts rrps vols types
    let nf = r32 q
    let (ws, hs) = unzip (map computeWH options)
    let (w_m_r, w_m_s, w_per) = decideSkewness nf ws
    let (h_m_r, h_m_s, h_per) = decideSkewness nf hs
    in  ( w_m_r, w_m_s, w_per, h_m_r, h_m_s, h_per )
            