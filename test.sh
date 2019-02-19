#!/bin/bash

cd ./seq/
./test.sh 2>&1 | tee ../results/52core_cpu-seq.csv
cd ..
cd ./cuda-option/
./test.sh 2>&1 | tee ../results/v100_cuda-option.csv
cd ..
cd ./cuda-multi/
./test.sh 2>&1 | tee ../results/v100_cuda-multi.csv
cd ..