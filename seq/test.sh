#!/bin/bash

# Usage:
# $ sh test.sh compile - to compile all 4 executables once
# $ sh test.sh - to run benchmarking with the specified parameters

# program options
rep=3

# data
data_path="../data"
files=("0_UNIFORM_1000" "0_UNIFORM_3000" "0_UNIFORM_5000" "0_UNIFORM_100000")
# full case
# files=("0_UNIFORM_5000" "1_RAND_100000" "1_RAND_NORMH_100000" "1_RAND_NORMW_100000" "2_RANDCONSTHEIGHT_100000" "3_RANDCONSTWIDTH_100000" "4_SKEWED_1_100000" "4_SKEWED_INV_1_100000" "4_SKEWED_5_100000" "4_SKEWED_INV_5_100000" "1_RAND_100000_zero_EU" "0_UNIFORM_1_CALL_PUT_EU_Berm_US_oas")

# executables
exe="../build/Seq"
exefloat=$exe"-float"
exedouble=$exe"-double"
exes=($exefloat $exedouble)
exes_names=("float,-" "double,-")
exes_to_run=(0 1)

compile() {
    echo "Compiling float version..."
    make -B compile REAL=32
    mv $exe $exefloat
    echo "Compiling double version..."
    make -B compile REAL=64
    mv $exe $exedouble
}

compile_gtx780() {
    echo "Compiling float version..."
    make -B compile_gtx780 REAL=32
    mv $exe $exefloat
    echo "Compiling double version..."
    make -B compile_gtx780 REAL=64
    mv $exe $exedouble
}

test() {
    echo "file,precision,registers,version,block,sort,kernel time,total time,memory"
    for file in ${files[*]}
    do
        for index in ${exes_to_run[*]}
        do 
            ./${exes[$index]} -o $data_path/$file.in -r $rep | awk -v prefix="$file,${exes_names[$index]}," '{print prefix $0}'
        done
    done
}

validate() {
    mkdir -p $data_path/out32/
    mkdir -p $data_path/out64/
    for file in ${files[*]}
    do
        for index in ${exes_to_run[*]}
        do
            echo "Validating ${exes[$index]} on $file"
            if [ $index = 0 ]; then
                ./${exes[$index]} -o $data_path/$file.in > $data_path/out32/$file.out 
            else
                ./${exes[$index]} -o $data_path/$file.in > $data_path/out64/$file.out
            fi
        done
    done
}

if [ "$1" = "compile" ]; then
    compile
elif [ "$1" = "compile_gtx780" ]; then
    compile_gtx780
elif [ "$1" = "validate" ]; then
    validate
else
    test
fi
