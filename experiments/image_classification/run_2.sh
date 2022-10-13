# CUDA_VISIBLE_DEVICES=3 python run_experiment.py --config=configs/project/alexnet_eps1_baseline.py --jaxline_mode=train_eval_multithreaded
CUDA_VISIBLE_DEVICES=2 python run_experiment_gcn.py --config=configs/project/gcn_eps1_baseline.py --jaxline_mode=train_eval_multithreaded --jaxline_disable_pmap_jit
# CUDA_VISIBLE_DEVICES=3 python run_experiment.py --config=configs/project/alexnet_eps1_top50.py --jaxline_mode=train_eval_multithreaded
# CUDA_VISIBLE_DEVICES=3 python run_experiment.py --config=configs/project/mnist_lenet_eps1_top10.py --jaxline_mode=train_eval_multithreaded
# CUDA_VISIBLE_DEVICES=3 python run_experiment.py --config=configs/project/mnist_lenet_eps1_top50.py --jaxline_mode=train_eval_multithreaded