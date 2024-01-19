chcp 65001
cd /d "%~dp0"

python train.py "E:\Dataset.Image\CelebA" --processes 2 --max-steps 128000 --random-seed 0 --device /gpu:0 --batch-size 12 --gan-type wgan-div --postfix 14 --restore

pause

python train.py "E:\Dataset.Image\CelebA" --processes 2 --max-steps 256000 --random-seed 0 --device /gpu:0 --batch-size 12 --gan-type wgan --postfix 13
