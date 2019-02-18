#!/bin/bash

cd ./seq/
./test.sh compile_GTX780
cd ..
cd ./cuda-option/
./test.sh compile_GTX780
cd ..
cd ./cuda-multi/
./test.sh compile_GTX780
cd ..
cd ./test/
make compile_GTX780
make compare
cd ..
cd ./data-generator/
make compile_GTX780
cd ..
