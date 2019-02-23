#!/bin/bash

output_path=$1

cd ./seq/
./test.sh 2>&1 | tee $output_path/52core_cpu-seq.csv
cd ..
cd ./cuda-option/
./test.sh 2>&1 | tee $output_path/v100_cuda-option.csv
cd ..
cd ./cuda-multi/
./test.sh 2>&1 | tee $output_path/v100_cuda-multi.csv
cd ..