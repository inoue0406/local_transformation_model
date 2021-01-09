#!/usr/bin/bash

python ../src_prep/prep_radarJMA_multiscale-avg_alljapan.py 0 3 >& log1 &
python ../src_prep/prep_radarJMA_multiscale-avg_alljapan.py 3 9 >& log2 &
python ../src_prep/prep_radarJMA_multiscale-avg_alljapan.py 9 12 >& log3 &
python ../src_prep/prep_radarJMA_multiscale-avg_alljapan.py 12 15 >& log4 &
python ../src_prep/prep_radarJMA_multiscale-avg_alljapan.py 15 18 >& log5 &
python ../src_prep/prep_radarJMA_multiscale-avg_alljapan.py 18 21 >& log6 &
python ../src_prep/prep_radarJMA_multiscale-avg_alljapan.py 21 24 >& log7 &
python ../src_prep/prep_radarJMA_multiscale-avg_alljapan.py 24 27 >& log8 &
python ../src_prep/prep_radarJMA_multiscale-avg_alljapan.py 27 30 >& log9 &
python ../src_prep/prep_radarJMA_multiscale-avg_alljapan.py 30 33 >& log10 &
python ../src_prep/prep_radarJMA_multiscale-avg_alljapan.py 33 35 >& log11 &
