#!/bin/bash

sl_polarity.py -i particles.star -c class_averages.mrcs -o polarity.mrcs

sl_assemble.py --mrcs polarity.mrcs --output comp.png --mask-radius 0.9  --angles 0 0  1.0  --yshifts 0 0 1  --prealign --lowpass 3  --   metric ssim --cc-window 100 60 --flips none

sl_sart_search.py -i comp.png -o comp-out --angles 0 0 1 --shifts 0 0 1 --radius 80 --prealign --resize 1.5

