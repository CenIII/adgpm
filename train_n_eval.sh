#!/bin/sh
# python evaluate_imagenet.py --cnn materials/resnet50-base.pth --pred save/random_unfix_shufl2/epoch-5000.pred --test-set 2-hops --keep-ratio 0.1 --gpu 1 > ./save/random_unfix_shufl2/eval_log
# python train_gcn_basic.py --save-path save/random_unfix_shufl2_gt --gpu 1 --max-epoch 5000 --save-epoch 5000 > ./save/random_unfix_shufl2_gt/train_log
# python evaluate_imagenet.py --cnn materials/resnet50-base.pth --pred save/random_unfix_shufl2_gt/epoch-5000.pred --test-set 2-hops --keep-ratio 0.1 --gpu 1 > ./save/random_unfix_shufl2_gt/eval_log

num_list=(0 1 2 3 4)
## shallow
# original
git checkout master
save_prefix='./save/original'
for i in "${num_list[@]}"; do
	mkdir $save_prefix$i
	python train_gcn_basic.py --save-path $save_prefix$i --max-epoch 5000 --save-epoch 3000 > $save_prefix$i'/train_log'
done

# sym
save_prefix='./save/sym'
for i in "${num_list[@]}"; do
	mkdir $save_prefix$i
	python train_gcn_basic.py --save-path $save_prefix$i --max-epoch 5000 --save-epoch 3000 --norm-method sym > $save_prefix$i'/train_log'
done

# random_fix
git checkout random_fix
save_prefix='./save/random_fix'
for i in "${num_list[@]}"; do
	mkdir $save_prefix$i
	python train_gcn_basic.py --save-path $save_prefix$i --max-epoch 5000 --save-epoch 3000 > $save_prefix$i'/train_log'
done

# random_unfix
git checkout random_unfix
save_prefix='./save/random_unfix'
for i in "${num_list[@]}"; do
	mkdir $save_prefix$i
	python train_gcn_basic.py --save-path $save_prefix$i --max-epoch 5000 --save-epoch 3000 > $save_prefix$i'/train_log'
done

# EMFSaA
git checkout EMFSaA
save_prefix='./save/EMFSaA'
for i in "${num_list[@]}"; do
	mkdir $save_prefix$i
	python train_gcn_basic.py --save-path $save_prefix$i --max-epoch 5000 --save-epoch 3000 > $save_prefix$i'/train_log'
done

## Deep
# original
git checkout master
save_prefix='./save/original_deep'
for i in "${num_list[@]}"; do
	mkdir $save_prefix$i
	python train_gcn_basic.py --save-path $save_prefix$i --max-epoch 5000 --save-epoch 3000 --layers 2048,2048,1024,1024,d512,d > $save_prefix$i'/train_log'
done

# sym
save_prefix='./save/sym_deep'
for i in "${num_list[@]}"; do
	mkdir $save_prefix$i
	python train_gcn_basic.py --save-path $save_prefix$i --max-epoch 5000 --save-epoch 3000 --layers 2048,2048,1024,1024,d512,d > $save_prefix$i'/train_log'
done

# random_fix
git checkout random_fix
save_prefix='./save/random_fix_deep'
for i in "${num_list[@]}"; do
	mkdir $save_prefix$i
	python train_gcn_basic.py --save-path $save_prefix$i --max-epoch 5000 --save-epoch 3000 --layers 2048,2048,1024,1024,d512,d > $save_prefix$i'/train_log'
done

# random_unfix
git checkout random_unfix
save_prefix='./save/random_unfix_deep'
for i in "${num_list[@]}"; do
	mkdir $save_prefix$i
	python train_gcn_basic.py --save-path $save_prefix$i --max-epoch 5000 --save-epoch 3000 --layers 2048,2048,1024,1024,d512,d > $save_prefix$i'/train_log'
done

# EMFSaA
git checkout EMFSaA
save_prefix='./save/EMFSaA_deep'
for i in "${num_list[@]}"; do
	mkdir $save_prefix$i
	python train_gcn_basic.py --save-path $save_prefix$i --max-epoch 5000 --save-epoch 3000 --layers 2048,2048,1024,1024,d512,d > $save_prefix$i'/train_log'
done