num_list=(0 1 2 3 4)
save_prefix='./save/EMFSaA_deep'
for i in "${num_list[@]}"; do
        git reset --hard HEAD
        git checkout EMFSaA
        mkdir $save_prefix$i
        python train_gcn_basic.py --save-path $save_prefix$i --max-epoch 5000 --save-epoch 3000 --layers 2048,2048,1024,1024,d512,d --gpu 3 > $save_prefix$i'/train_log'
done
