#!/bin/bash

# Usage:
# $ sh test.sh compile - to compile all 4 executables once
# $ sh test.sh - to run benchmarking with the specified parameters

# program options
rep=3
device=0
sorts="-s - -s w -s W -s h -s H"
# sorts="w W h H"
# block_sizes="32 64 128 256 512 1024"
# block_sizes="512"
block_sizes="-b 512 -b 1024"
versions="-v 1 -v 2 -v 3"
# versions="3"

# data
data_path="../data"
results_path="../results"
# files=("book" "options-1000" "options-60000")
# files=("0_UNIFORM" "1_RAND" "2_RANDCONSTHEIGHT" "3_RANDCONSTWIDTH" "4_SKEWED" "5_SKEWEDCONSTHEIGHT" "6_SKEWEDCONSTWIDTH")
# files=("0_UNIFORM_100000" "1_RAND_100000" "4_SKEWED_100000")
files=("0_UNIFORM_100000" "1_RAND_100000" "2_RANDCONSTHEIGHT_100000" "3_RANDCONSTWIDTH_100000" "4_SKEWED_100000" "5_SKEWEDCONSTHEIGHT_100000" "6_SKEWEDCONSTWIDTH_100000")

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

validate() {
    for file in ${files[*]}
    do
        for index in ${exes_to_run[*]}
        do 
            ./${exes[$index]} -o $data_path/$file.in $sorts $versions $block_sizes -d $device > $results_path/test.out  | awk -v prefix="$file,${exes_names[$index]}," '{print prefix $0}'
            if [ index = 0 || index = 1 ]; then
                cat $data_path/out32/$file.out $results_path/test.out | ../build/Compare
            else
                cat $data_path/out/$file.out $results_path/test.out | ../build/Compare
            fi
        done
    done
}

if [ "$1" = "compile" ]; then
    compile
elif [ "$1" = "validate" ]; then
    validate
else
    test
fi
