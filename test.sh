#!/bin/bash

cd ./seq/
./test.sh
cd ..
cd ./cuda-option/
./test.sh
cd ..
cd ./cuda-multi/
./test.sh
cd ..