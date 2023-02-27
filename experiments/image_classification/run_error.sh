for eps in 0.5 1 2 4
do
    # CUDA_VISIBLE_DEVICES=6,7 python run_experiment.py --config=configs/resnet18/cifar10_resnet18_eps1_lr4_top10.py --jaxline_mode=train_eval_multithreaded
    # CUDA_VISIBLE_DEVICES=4,5 python run_experiment.py --config=configs/cifar10_wrn_16_4_eps${eps}_top10.py --jaxline_mode=train_eval_multithreaded
    CUDA_VISIBLE_DEVICES=6,7 python run_experiment.py --config=configs/cifar100_wrn_28_10_eps${eps}_top10.py --jaxline_mode=train_eval_multithreaded
    # CUDA_VISIBLE_DEVICES=4,5 python run_experiment.py --config=configs/cifar100_wrn_28_10_eps2_top10.py --jaxline_mode=train_eval_multithreaded
    # CUDA_VISIBLE_DEVICES=6,7 python run_experiment.py --config=configs/cifar10_wrn_16_4_eps1_random${amount}.py --jaxline_mode=train_eval_multithreaded
done
# for eps in 0.25 2
# do
#     CUDA_VISIBLE_DEVICES=2,3 python run_experiment.py --config=configs/cifar10_wrn_28_10_eps${eps}_fintune_top10.py --jaxline_mode=train_eval_multithreaded
# done
# CUDA_VISIBLE_DEVICES=2 python run_experiment.py --config=configs/project/mlp_eps1_baseline.py --jaxline_mode=train_eval_multithreaded
# CUDA_VISIBLE_DEVICES=0,1 python run_experiment.py --config=configs/resnet18/cifar10_resnet18_eps1_lr4_top10.py --jaxline_mode=train_eval_multithreaded