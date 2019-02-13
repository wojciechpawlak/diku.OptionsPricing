#!/bin/bash

# cd ./seq/
# ./test.sh validate
# cd ..
cd ./cuda-option/
./compare.sh
cd ..
cd ./cuda-multi/
./compare.sh
cd ..