python evaluate_imagenet.py --cnn materials/resnet50-base.pth --pred save/random_unfix_shufl2/epoch-5000.pred --test-set 2-hops --keep-ratio 0.1 --gpu 1 > ./save/random_unfix_shufl2/eval_log
python train_gcn_basic.py --save-path save/random_unfix_shufl2_gt --gpu 1 --max-epoch 5000 --save-epoch 5000 > ./save/random_unfix_shufl2_gt/train_log
python evaluate_imagenet.py --cnn materials/resnet50-base.pth --pred save/random_unfix_shufl2_gt/epoch-5000.pred --test-set 2-hops --keep-ratio 0.1 --gpu 1 > ./save/random_unfix_shufl2_gt/eval_log
