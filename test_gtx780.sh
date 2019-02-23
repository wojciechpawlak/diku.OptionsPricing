#!/bin/bash

output_path=$1

cd ./seq/
./test.sh test_gtx780 2>&1 | tee $output_path/32core_cpu-seq.csv
cd ..
cd ./cuda-option/
./test.sh test_gtx780 2>&1 | tee $output_path/gtx780_cuda-option.csv
cd ..
cd ./cuda-multi/
./test.sh test_gtx780 2>&1 | tee $output_path/gtx780_cuda-multi.csv
cd ..