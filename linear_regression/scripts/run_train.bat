@echo off
python train.py ^
    --epochs=5 ^
    --num_data=1600 ^
    --batch_size=16 ^
    --repeat_size=1
pause