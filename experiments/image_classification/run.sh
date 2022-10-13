# for amount in 10
# do
#     CUDA_VISIBLE_DEVICES=0,1 python run_experiment.py --config=configs/cifar10_wrn_16_4_eps4_top${amount}.py --jaxline_mode=train_eval_multithreaded
# done
# CUDA_VISIBLE_DEVICES=2 python run_experiment.py --config=configs/project/mlp_eps1_baseline.py --jaxline_mode=train_eval_multithreaded
CUDA_VISIBLE_DEVICES=2 python run_experiment.py --config=configs/project/mlp_eps1_baseline.py --jaxline_mode=train_eval_multithreaded