#!/bin/bash

# Usage:
# $ sh test.sh compile - to compile all 4 executables once
# $ sh test.sh - to run benchmarking with the sepcified parameters

# program options
rep=5
device=1
sorts="- h H w W"
block_sizes="256 512 1024"
versions="4"

# data
data_path="../data"
files=("0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
yield="yield"

# executables
exe="../build/CudaOption"
exefloat=$exe"-float"
exefloatreg=$exefloat"-reg32"
exedouble=$exe"-double"
exedoublereg=$exedouble"-reg32"
exes=($exefloat $exefloatreg $exedouble $exedoublereg)
exes_names=("float,-" "float,32" "double,-" "double,32")

compile() {
    echo "Compiling float version..."
    make -B REAL=32
    mv $exe $exefloat
    echo "Compiling float version with 32 registers..."
    make -B REAL=32 REG=32
    mv $exe $exefloatreg
    echo "Compiling double version..."
    make -B REAL=64
    mv $exe $exedouble
    echo "Compiling double version with 32 registers..."
    make -B REAL=64 REG=32
    mv $exe $exedoublereg
}

test() {
    echo "file,precision,registers,version,block,sort,kernel time,total time"
    for file in ${files[*]}
    do
        for index in ${!exes[*]}
        do 
            ./${exes[$index]} -o $data_path/$file.in -y $data_path/$yield.in -s $sorts -v $versions -b $block_sizes -r $rep -d $device | awk -v prefix="$file,${exes_names[$index]}," '{print prefix $0}'
        done
    done
}

if [ "$1" = "compile" ]; then
    compile
else
    test
fi
