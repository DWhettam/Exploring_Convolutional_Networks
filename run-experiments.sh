#!/bin/sh
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type dilated_convolution --dilation 1 --dropout True --experiment_name dilation_final_test1 --use_gpu True
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type dilated_convolution --dilation 1 --dropout True --experiment_name dilation_final_test2 --use_gpu True
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type dilated_convolution --dilation 1 --dropout True --experiment_name dilation_final_test3 --use_gpu True

python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type strided_convolution --stride 1 --dropout True --experiment_name stride_final_test1 --use_gpu True
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type strided_convolution --stride 1 --dropout True --experiment_name stride_final_test2 --use_gpu True
python mlp/pytorch_experiment_scripts/train_evaluate_emnist_classification_system.py --batch_size 100 --seed 0 --num_filters 32 --dim_reduction_type strided_convolution --stride 1 --dropout True --experiment_name stride_final_test3 --use_gpu True
