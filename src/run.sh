#!/bin/bash

# shell file for running python script

file=${1--}
set -e
source "../../.myenv/bin/activate" # insert path to your virtual envirnment
python -u $file
