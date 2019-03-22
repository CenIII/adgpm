python evaluate_imagenet.py --cnn materials/resnet50-base.pth --pred save/EMFSaA/epoch--.pred --test-set 2-hops --keep-ratio 0.1 --gpu 0 > ./save/EMFSaA/eval_log
python evaluate_imagenet.py --cnn materials/resnet50-base.pth --pred save/EMFSaA/epoch--.Expt.pred --test-set 2-hops --keep-ratio 0.1 --gpu 0 > ./save/EMFSaA/eval_log_Expt
