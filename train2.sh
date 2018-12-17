export CUDA_VISIBLE_DEVICES=1
python train.py "/my/Datasets/CelebA" --processes 2 --max-steps 512000 --random-seed 0 --device /gpu:1 --batch-size 16 --gan-type wgan --postfix 12

exit
