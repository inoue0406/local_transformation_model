#!/bin/bash

python ../src_prep/artificial_rain_field_UVC.py 1 0 2000 >& log1 &
python ../src_prep/artificial_rain_field_UVC.py 2 2000 4000 >& log2 &
python ../src_prep/artificial_rain_field_UVC.py 3 4000 6000 >& log3 &
python ../src_prep/artificial_rain_field_UVC.py 4 6000 8000 >& log4 &
python ../src_prep/artificial_rain_field_UVC.py 5 8000 10000 >& log5 &
