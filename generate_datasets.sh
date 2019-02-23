#!/bin/bash

# American/Bermudan fixed-rate (coupon) bonds
# ./build/RandomDataGenerator -n 10000 -p ./data/
# ./build/RandomDataGenerator -n 65536 -p ./data/
# ./build/RandomDataGenerator -n 100000 -p ./data/
# European zero-coupon bonds
# ./build/RandomDataGenerator -n 10000 -e -c -p ./data/
# ./build/RandomDataGenerator -n 65536 -e -c -p ./data/
# ./build/RandomDataGenerator -n 100000 -e -c -p ./data/

# full case
# files=("0_UNIFORM_5000" "1_RAND_100000" "1_RAND_NORMH_100000" "1_RAND_NORMW_100000" "2_RANDCONSTHEIGHT_100000" "3_RANDCONSTWIDTH_100000" "4_SKEWED_1_100000" "4_SKEWED_INV_1_100000" "4_SKEWED_5_100000" "4_SKEWED_INV_5_100000" "1_RAND_100000_zero_EU" "0_UNIFORM_1_CALL_PUT_EU_Berm_US_oas")

./build/RandomDataGenerator -t 0 -n 1000 -p ./data/
./build/RandomDataGenerator -t 0 -n 3000 -p ./data/
./build/RandomDataGenerator -t 0 -n 5000 -p ./data/
./build/RandomDataGenerator -t 0 -n 100000 -p ./data/

./build/RandomDataGenerator -t 1 -n 100000 -p ./data/
./build/RandomDataGenerator -t 1 -n 100000 -p ./data/ --isHeightNormalDist
./build/RandomDataGenerator -t 1 -n 100000 -p ./data/ --isWidthNormalDist
./build/RandomDataGenerator -t 1 -n 100000 -p ./data/ -c -e
./build/RandomDataGenerator -t 2 -n 100000 -p ./data/
./build/RandomDataGenerator -t 3 -n 100000 -p ./data/

./build/RandomDataGenerator -t 4 -n 100000 -p ./data/ -s 1
./build/RandomDataGenerator -t 4 -n 100000 -p ./data/ -s 1 --inverseSkewed
./build/RandomDataGenerator -t 4 -n 100000 -p ./data/ -s 5
./build/RandomDataGenerator -t 4 -n 100000 -p ./data/ -s 5 --inverseSkewed
