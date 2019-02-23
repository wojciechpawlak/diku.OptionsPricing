#!/bin/bash

# Absolute path
output_path=$1
device=$2

mkdir -p $output_path

cd ./seq/
./test.sh test_gtx780 $device 2>&1 | tee $output_path/32core_cpu-seq.csv
cd ..
cd ./cuda-option/
./test.sh test_gtx780 $device 2>&1 | tee $output_path/gtx780_cuda-option.csv
cd ..
cd ./cuda-multi/
./test.sh test_gtx780 $device 2>&1 | tee $output_path/gtx780_cuda-multi.csv
cd ..