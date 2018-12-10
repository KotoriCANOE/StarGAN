export CUDA_VISIBLE_DEVICES=1
python train.py "/my/Datasets/CelebA" --processes 2 --max-steps 256000 --random-seed 0 --device /gpu:1 --batch-size 16 --postfix 10

exit
