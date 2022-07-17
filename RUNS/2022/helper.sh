#!/bin/bash
#Created on Sun Jul 17 13:34:17 2022

#@author: Gerardo A. Rivera Tello
#@handle: grivera@igp.gob.pe

for dir in $(ls -d */); do
    echo "Running in $dir"
    mkdir ${dir}logs && mv ${dir}*.txt ${dir}logs
done
