#!/bin/bash

# Usage:
# $ sh test.sh compile - to compile all executables once
# $ sh test.sh - to compare outputs with the sepcified parameters

# compile options
real=64
reg=""

# program options
device=0
# sorts="- w W h H"
sorts="-"
# block_sizes="64 128"
block_sizes="128"
versions="1 2 3 4"

# data
data_path="../data"
# files=("book" "options-1000" "options-60000")
# files=("0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
files=("0_UNIFORM_1_CALL_PUT_EU_Berm_US_oas" "0_UNIFORM_10000" "0_UNIFORM_100000" "0_UNIFORM_100000_zero_EU" "0_UNIFORM_65536" "1_RAND_100000" "1_RAND_100000_zero_EU" "1_RAND_65536" "2_RANDCONSTHEIGHT_100000" "4_SKEWED_1_100000" "4_SKEWED_1_100000_zero_EU" "4_SKEWED_1_65536")

# executables
compare="../build/Compare"
test_dir="../test"
exe="../build/CudaOption"
exefloat=$exe"-float"
exefloatreg=$exefloat"-reg32"
exedouble=$exe"-double"
exedoublereg=$exedouble"-reg32"
exes=($exefloat $exefloatreg $exedouble $exedoublereg)
exes_names=("float,-" "float,32" "double,-" "double,32")
# exes_to_run=(0 1 2 3)
exes_to_run=(2)

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
    echo "Compiling compare..."
    make --no-print-directory -C $test_dir -B compile-compare REAL=$real
}

compare() {
    for file in ${files[*]}
    do
        for index in ${exes_to_run[*]}
        do 
            for version in ${versions[*]}
            do
                for block_size in ${block_sizes[*]}
                do
                    for sort in ${sorts[*]}
                    do
                        echo "Comparing ${exes[$index]} on $file (version $version, block size $block_size, sort $sort)"
                        {
                            ./${exes[$index]} -o $data_path/$file.in -s $sort -v $version -b $block_size -d $device
                                if [ $index = 0 ] || [ $index = 1 ]; then
                                    cat $data_path/out32/$file.out
                                else
                                    cat $data_path/out64/$file.out
                                fi
                        } | ./$compare
                    done
                done
            done
        done
    done
}

if [ "$1" = "compile" ]; then
    compile
else
    compare
fi
