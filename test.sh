#!/bin/bash

cd ./seq/
# ./test.sh
./compute.sh
cd ../cuda-option/
./test.sh
cd ../cuda-multi/
./test.sh
cd ..