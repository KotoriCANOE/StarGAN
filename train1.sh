export CUDA_VISIBLE_DEVICES=0
python train.py "/my/Datasets/CelebA" --processes 2 --max-steps 256000 --random-seed 0 --device /gpu:0 --batch-size 16 --postfix 9

exit
