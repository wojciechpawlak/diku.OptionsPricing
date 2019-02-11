#!/bin/bash

cd ./seq/
./test.sh compile
# cd ./cuda-option/
cd ../cuda-option/
./test.sh compile
cd ../cuda-multi/
# cd ./cuda-multi/
./test.sh compile
cd ..