#!/bin/bash

# American/Bermudan fixed-rate (coupon) bonds
# ./build/RandomDataGenerator -n 10000 -p ./data/
# ./build/RandomDataGenerator -n 65536 -p ./data/
# ./build/RandomDataGenerator -n 100000 -p ./data/
# European zero-coupon bonds
# ./build/RandomDataGenerator -n 10000 -e -c -p ./data/
# ./build/RandomDataGenerator -n 65536 -e -c -p ./data/
# ./build/RandomDataGenerator -n 100000 -e -c -p ./data/

./build/RandomDataGenerator -t 0 -n 1000 -p ./data/
./build/RandomDataGenerator -t 0 -n 3000 -p ./data/
./build/RandomDataGenerator -t 0 -n 5000 -p ./data/

./build/RandomDataGenerator -t 1 -n 100000 -p ./data/
./build/RandomDataGenerator -t 1 -n 100000 -p ./data/ --isHeightNormalDist
./build/RandomDataGenerator -t 1 -n 100000 -p ./data/ --isWidthNormalDist

./build/RandomDataGenerator -t 4 -n 100000 -p ./data/ -s 1
./build/RandomDataGenerator -t 4 -n 100000 -p ./data/ -s 1 --inverseSkewed