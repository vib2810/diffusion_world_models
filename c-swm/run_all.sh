# python train.py --dataset data/balls_train.h5 --encoder medium --embedding-dim 4 --num-objects 3 --ignore-action --name balls
# python eval.py --dataset data/balls_eval.h5 --save-folder checkpoints/balls --num-steps 1

python train.py --dataset data/spaceinvaders_train.h5 --encoder medium --embedding-dim 4 --action-dim 6 --num-objects 3 --copy-action --epochs 200 --name spaceinvaders --decoder
python train.py --dataset data/shapes_train.h5 --encoder small --name shapes --decoder
# python eval.py --dataset data/spaceinvaders_eval.h5 --save-folder checkpoints/spaceinvaders --num-steps 1