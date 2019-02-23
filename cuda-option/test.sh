#!/bin/bash

# Usage:
# $ sh test.sh compile - to compile all 4 executables once
# $ sh test.sh - to run benchmarking with the specified parameters

# program options
device=$2
echo "Running on device $device."
rep=3
sorts="-s - -s w -s W"
# sorts="w h"
# block_sizes="32 64 128 256 512 1024"
block_sizes="-b 128"
versions="-v 1 -v 2 -v 3 -v 4"
# versions="3 4"

sorts_gtx780="- w W"
block_sizes_gtx780="64 128"
versions_gtx780="1 2 3 4"

# data
data_path="../data"
files=("0_UNIFORM_1000" "0_UNIFORM_3000" "0_UNIFORM_5000" "0_UNIFORM_100000")
# full case
# files=("0_UNIFORM_5000" "1_RAND_100000" "1_RAND_NORMH_100000" "1_RAND_NORMW_100000" "2_RANDCONSTHEIGHT_100000" "3_RANDCONSTWIDTH_100000" "4_SKEWED_1_100000" "4_SKEWED_INV_1_100000" "4_SKEWED_5_100000" "4_SKEWED_INV_5_100000" "1_RAND_100000_zero_EU" "0_UNIFORM_1_CALL_PUT_EU_Berm_US_oas")

# executables
exe="../build/CudaOption"
exefloat=$exe"-float"
exefloatreg=$exefloat"-reg32"
exedouble=$exe"-double"
exedoublereg=$exedouble"-reg32"
exes=($exefloat $exefloatreg $exedouble $exedoublereg)
exes_names=("float,-" "float,32" "double,-" "double,32")
# exes_to_run=(0 1 2 3)
exes_to_run=(0 2)

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
            ./${exes[$index]} -o $data_path/$file.in -s $sorts_gtx780 -v $versions_gtx780 -b $block_sizes_gtx780 -r $rep -d $device | awk -v prefix="$file,${exes_names[$index]}," '{print prefix $0}'
        done
    done
}

compile() {
    echo "Compiling float version..."
    make -B compile REAL=32
    mv $exe $exefloat
    # echo "Compiling float version with 32 registers..."
    # make -B compile REAL=32 REG=32
    # mv $exe $exefloatreg
    echo "Compiling double version..."
    make -B compile REAL=64
    mv $exe $exedouble
    # echo "Compiling double version with 32 registers..."
    # make -B compile REAL=64 REG=32
    # mv $exe $exedoublereg
}

compile_gtx780() {
    echo "Compiling float version..."
    make -B compile_gtx780 REAL=32
    mv $exe $exefloat
    # echo "Compiling float version with 32 registers..."
    # make -B compile REAL=32 REG=32
    # mv $exe $exefloatreg
    echo "Compiling double version..."
    make -B compile_gtx780 REAL=64
    mv $exe $exedouble
    # echo "Compiling double version with 32 registers..."
    # make -B compile REAL=64 REG=32
    # mv $exe $exedoublereg
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
