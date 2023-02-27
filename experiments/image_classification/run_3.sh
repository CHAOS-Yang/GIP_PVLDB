# CUDA_VISIBLE_DEVICES=4,5 python run_experiment.py --config=configs/resnet18/cifar10_resnet18_eps1_lr4_random50.py --jaxline_mode=train_eval_multithreaded
for eps in 0.5 1 2 4
do
    # CUDA_VISIBLE_DEVICES=6,7 python run_experiment.py --config=configs/cifar10_wrn_28_10_eps${eps}_fintune_top10.py --jaxline_mode=train_eval_multithreaded
    CUDA_VISIBLE_DEVICES=4,5 python run_experiment.py --config=configs/cifar10_wrn_16_4_eps${eps}_top10.py --jaxline_mode=train_eval_multithreaded
    # CUDA_VISIBLE_DEVICES=6,7 python run_experiment.py --config=configs/svhn_wrn_16_4_eps${eps}_top10.py --jaxline_mode=train_eval_multithreaded
done