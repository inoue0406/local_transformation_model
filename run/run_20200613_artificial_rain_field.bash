#!/bin/bash

python ../src_prep/artificial_rain_field.py 1 0 2000 >& log1 &
python ../src_prep/artificial_rain_field.py 2 2000 4000 >& log2 &
python ../src_prep/artificial_rain_field.py 3 4000 6000 >& logex3 &
python ../src_prep/artificial_rain_field.py 4 6000 8000 >& log4 &
python ../src_prep/artificial_rain_field.py 5 8000 10000 >& log5 &


