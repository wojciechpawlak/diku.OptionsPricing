#!/bin/bash

cd ./seq/
./test.sh compile
cd ..
cd ./cuda-option/
./test.sh compile
cd ..
cd ./cuda-multi/
./test.sh compile
cd ..
cd ./test/
make compile
make compare
cd ..
cd ./data-generator/
make compile
cd ..
