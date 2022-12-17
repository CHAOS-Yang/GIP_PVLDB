# CUDA_VISIBLE_DEVICES=3 python run_experiment.py --config=configs/project/alexnet_eps1_baseline.py --jaxline_mode=train_eval_multithreaded
CUDA_VISIBLE_DEVICES=4,5,6,7 python run_experiment.py --config=configs/imagenet_nf_resnet_50_eps8_top50.py --jaxline_mode=train_eval_multithreaded
# CUDA_VISIBLE_DEVICES=3 python run_experiment.py --config=configs/project/alexnet_eps1_top50.py --jaxline_mode=train_eval_multithreaded
# CUDA_VISIBLE_DEVICES=3 python run_experiment.py --config=configs/project/mnist_lenet_eps1_top10.py --jaxline_mode=train_eval_multithreaded
# CUDA_VISIBLE_DEVICES=3 python run_experiment.py --config=configs/project/mnist_lenet_eps1_top50.py --jaxline_mode=train_eval_multithreaded