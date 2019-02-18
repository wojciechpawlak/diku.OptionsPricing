#!/bin/bash

# American/Bermudan fixed-rate (coupon) bonds
./build/RandomDataGenerator -n 10000 -p ./data/
./build/RandomDataGenerator -n 100000 -p ./data/
# European zero-coupon bonds
./build/RandomDataGenerator -n 10000 -e -c -p ./data/
./build/RandomDataGenerator -n 100000 -e -c -p ./data/
