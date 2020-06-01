#!/bin/bash
rm ret.txt
rm execution_results.txt
R_size=128000000
S_size=128000000
MAX_RADIX=12

for i in $(seq 1 $MAX_RADIX); do

  sed -i "/#define NUM_RADIX_BITS /c\#define NUM_RADIX_BITS $i" ./host/src/prj_params.h
  make clean
  make
  echo  RADIX $i >> ret.txt
  ~/bin/harp_run bin/host >> ret.txt

done
