num_list=(0 1 2 3 4)
save_prefix='./save/'
for i in "${num_list[@]}"; do
        echo $i
        python evaluate_imagenet.py --cnn materials/resnet50-base.pth --pred $save_prefix$1$i/epoch-*.pred --test-set 2-hops --gpu 0 --keep-ratio 0.1 > $save_prefix$1$i'/result'
        python evaluate_imagenet.py --cnn materials/resnet50-base.pth --pred $save_prefix$1$i/epoch-*.pred --test-set 2-hops --gpu 0 --keep-ratio 0.1 --consider-train >> $save_prefix$1$i'/result'
done
