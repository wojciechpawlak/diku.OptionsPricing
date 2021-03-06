import "/futlib/math"

type real = f32

let i2r     (i : i32 ) : real = r32 i
let ui2r    (i : u16 ) : real = f32.u16 i
let r2i     (a : real) : i32  = t32 a
let r_exp   (a : real) : real = f32.exp  a
let r_sqrt  (a : real) : real = f32.sqrt a
let r_abs   (a : real) : real = f32.abs  a
let r_log   (a : real) : real = f32.log  a
let r_ceil  (a : real) : real = f32.ceil a
let r_round (a : real) : real = f32.round a
let r_isinf (a : real) : bool = f32.isinf a
let r_max   (a : real, b : real) : real = f32.max a b
let r_convert_inf (a : real) : real =
    if (a == f32.inf) then 3.40282e+38
    else if (a == -f32.inf) then 1.17549e-38
    else a
