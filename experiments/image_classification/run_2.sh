for eps in 0.5 1 2 4
do
    # CUDA_VISIBLE_DEVICES=6,7 python run_experiment.py --config=configs/cifar10_wrn_28_10_eps${eps}_fintune_top10.py --jaxline_mode=train_eval_multithreaded
    # CUDA_VISIBLE_DEVICES=0,1 python run_experiment.py --config=configs/cifar100_wrn_28_10_eps1_top10.py --jaxline_mode=train_eval_multithreaded
    CUDA_VISIBLE_DEVICES=6,7 python run_experiment.py --config=configs/svhn_wrn_16_4_eps${eps}_top10.py --jaxline_mode=train_eval_multithreaded
done

# CUDA_VISIBLE_DEVICES=4,5,6,7 python run_experiment.py --config=configs/imagenet_nf_resnet_50_eps8_top50.py --jaxline_mode=train_eval_multithreaded
# CUDA_VISIBLE_DEVICES=3 python run_experiment.py --config=configs/project/alexnet_eps1_top50.py --jaxline_mode=train_eval_multithreaded
# CUDA_VISIBLE_DEVICES=3 python run_experiment.py --config=configs/project/mnist_lenet_eps1_top10.py --jaxline_mode=train_eval_multithreaded
# CUDA_VISIBLE_DEVICES=3 python run_experiment.py --config=configs/project/mnist_lenet_eps1_top50.py --jaxline_mode=train_eval_multithreaded