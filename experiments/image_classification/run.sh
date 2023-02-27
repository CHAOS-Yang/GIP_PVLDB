CUDA_VISIBLE_DEVICES=0 python run_experiment.py --config=configs/resnet18/cifar10_resnet20_eps2_lr4_top10.py --jaxline_mode=train_eval_multithreaded
CUDA_VISIBLE_DEVICES=0 python run_experiment.py --config=configs/resnet18/cifar10_resnet20_eps2_lr4_random10.py --jaxline_mode=train_eval_multithreaded
CUDA_VISIBLE_DEVICES=0 python run_experiment.py --config=configs/resnet18/cifar10_resnet20_eps2_lr4_baseline.py --jaxline_mode=train_eval_multithreaded
# CUDA_VISIBLE_DEVICES=0 python run_experiment.py --config=configs/resnet18/cifar10_resnet18_eps1_lr4_top10.py --jaxline_mode=train_eval_multithreaded
# for eps in 1 5 8
# do
#     # CUDA_VISIBLE_DEVICES=6,7 python run_experiment.py --config=configs/cifar10_wrn_28_10_eps${eps}_fintune_top10.py --jaxline_mode=train_eval_multithreaded
#     CUDA_VISIBLE_DEVICES=0 python run_experiment.py --config=configs/resnet18/cifar10_resnet20_eps${eps}_lr4_top10.py --jaxline_mode=train_eval_multithreaded
#     # CUDA_VISIBLE_DEVICES=2,3 python run_experiment.py --config=configs/resnet18/cifar10_resnet18_eps${eps}_lr4_top10.py --jaxline_mode=train_eval_multithreaded
# done
# for eps in 1 2 5 8
# do
#     # CUDA_VISIBLE_DEVICES=6,7 python run_experiment.py --config=configs/cifar10_wrn_28_10_eps${eps}_fintune_top10.py --jaxline_mode=train_eval_multithreaded
#     CUDA_VISIBLE_DEVICES=1 python run_experiment.py --config=configs/resnet18/cifar10_resnet20_eps${eps}_lr4_top10_1.py --jaxline_mode=train_eval_multithreaded
#     # CUDA_VISIBLE_DEVICES=2,3 python run_experiment.py --config=configs/resnet18/cifar10_resnet18_eps${eps}_lr4_top10.py --jaxline_mode=train_eval_multithreaded
# done
# for eps in 1 2 5 8
# do
#     CUDA_VISIBLE_DEVICES=1 python run_experiment.py --config=configs/resnet18/cifar10_resnet20_eps${eps}_lr4_top10_2.py --jaxline_mode=train_eval_multithreaded
# done
# CUDA_VISIBLE_DEVICES=2 python run_experiment.py --config=configs/project/mlp_eps1_baseline.py --jaxline_mode=train_eval_multithreaded
# CUDA_VISIBLE_DEVICES=0,1 python run_experiment.py --config=configs/resnet18/cifar10_resnet18_eps1_lr4_top10.py --jaxline_mode=train_eval_multithreaded