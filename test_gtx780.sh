#!/bin/bash

cd ./seq/
./test.sh 0 2>&1 | tee ../results/32core_cpu-seq.csv
cd ..
cd ./cuda-option/
./test.sh 0 2>&1 | tee ../results/gtx780_cuda-option.csv
cd ..
cd ./cuda-multi/
./test.sh 0 2>&1 | tee ../results/gtx780_cuda-multi.csv
cd ..