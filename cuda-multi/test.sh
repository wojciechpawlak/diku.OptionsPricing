#!/bin/bash

# Usage:
# $ sh test.sh compile - to compile all 4 executables once
# $ sh test.sh - to run benchmarking with the specified parameters

# program options
rep=3
device=0
sorts="-s - -s h -s H"
# sorts="w W h H"
# block_sizes="32 64 128 256 512 1024"
block_sizes="-b 512"
versions="-v 1 -v 2 -v 3"
# versions="3"

sorts_gtx780="- h H"
block_sizes_gtx780="512"
versions_gtx780="1 2 3"

# data
data_path="../data"
results_path="../results"
# files=("book" "options-1000" "options-60000")
# files=("0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
# files=("0_UNIFORM_100000" "1_RAND_100000" "4_SKEWED_100000")
# files=("0_UNIFORM_1_CALL_PUT_EU_Berm_US_oas" "0_UNIFORM_10000" "0_UNIFORM_100000" "1_RAND_100000" "2_RANDCONSTHEIGHT_100000" "3_RANDCONSTWIDTH_100000" "4_SKEWED_100000" "5_SKEWEDCONSTHEIGHT_100000" "6_SKEWEDCONSTWIDTH_100000")
# files=("0_UNIFORM_1_CALL_PUT_EU_Berm_US_oas" "0_UNIFORM_10000" "0_UNIFORM_100000" "0_UNIFORM_100000_zero_EU" "0_UNIFORM_65536" "1_RAND_100000" "1_RAND_100000_zero_EU" "1_RAND_65536" "2_RANDCONSTHEIGHT_100000" "4_SKEWED_1_100000" "4_SKEWED_1_100000_zero_EU" "4_SKEWED_1_65536")
files=("0_UNIFORM_1000" "0_UNIFORM_3000" "0_UNIFORM_5000" "1_RAND_100000" "1_RAND_NORMH_100000" "1_RAND_NORMW_100000" "4_SKEWED_1_100000" "4_SKEWED_INV_1_100000")

# executables
exe="../build/CudaMulti"
exefloat=$exe"-float"
exefloatreg=$exefloat"-reg32"
exedouble=$exe"-double"
exedoublereg=$exedouble"-reg32"
exes=($exefloat $exefloatreg $exedouble $exedoublereg)
exes_names=("float,-" "float,32" "double,-" "double,32")
# exes_to_run=(0 1 2 3)
exes_to_run=(0 1 2 3)

test() {
    echo "file,precision,registers,version,block,sort,kernel time,total time,memory"
    for file in ${files[*]}
    do
        for index in ${exes_to_run[*]}
        do 
            ./${exes[$index]} -o $data_path/$file.in $sorts $versions $block_sizes -r $rep -d $device | awk -v prefix="$file,${exes_names[$index]}," '{print prefix $0}'
        done
    done
}

test_gtx780() {
    echo "file,precision,registers,version,block,sort,kernel time,total time,memory"
    for file in ${files[*]}
    do
        for index in ${exes_to_run[*]}
        do 
            ./${exes[$index]} -o $data_path/$file.in -s $sorts_gtx780 -v  $versions_gtx780 -b $block_sizes_gtx780 -r $rep -d $device | awk -v prefix="$file,${exes_names[$index]}," '{print prefix $0}'
        done
    done
}

compile() {
    echo "Compiling float version..."
    make -B compile REAL=32
    mv $exe $exefloat
    echo "Compiling float version with 32 registers..."
    make -B compile REAL=32 REG=32
    mv $exe $exefloatreg
    echo "Compiling double version..."
    make -B compile REAL=64
    mv $exe $exedouble
    echo "Compiling double version with 32 registers..."
    make -B compile REAL=64 REG=32
    mv $exe $exedoublereg
}

compile_gtx780() {
    echo "Compiling float version..."
    make -B compile_gtx780 REAL=32
    mv $exe $exefloat
    echo "Compiling float version with 32 registers..."
    make -B compile_gtx780 REAL=32 REG=32
    mv $exe $exefloatreg
    echo "Compiling double version..."
    make -B compile_gtx780 REAL=64
    mv $exe $exedouble
    echo "Compiling double version with 32 registers..."
    make -B compile_gtx780 REAL=64 REG=32
    mv $exe $exedoublereg
}

if [ "$1" = "compile" ]; then
    compile
elif [ "$1" = "compile_gtx780" ]; then
    compile_gtx780
elif [ "$1" = "test_gtx780" ]; then
    test_gtx780
else
    test
fi
