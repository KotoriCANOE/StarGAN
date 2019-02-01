chcp 65001
cd /d "%~dp0"

python train.py "E:\Dataset.Image\CelebA" --processes 2 --max-steps 64000 --random-seed 0 --device /gpu:0 --batch-size 2 --gan-type dragan --postfix 10

pause
