#!/bin/bash

# batch size 4
START=0
END=31

# CUB
# baseline
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cub_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/cub_resnet101.pth --model_name resnet101
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cub_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/cub_resnet101.pth --model_name resnet101 --vis_mask CAM
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cub_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/cub_resnet101.pth --model_name resnet101 --vis_mask GradCAM

# cal
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cub_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/cal_ckpts/cub_resnet101_cal.pth --model_name resnet101 --selector cal --vis_mask GradCAM

for i in $(seq $START $END); do
    python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cub_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/cal_ckpts/cub_resnet101_cal.pth --model_name resnet101 --selector cal --vis_mask bap_${i}
done


# cars
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cars_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/cars_resnet101.pth --model_name resnet101
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cars_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/cars_resnet101.pth --model_name resnet101 --vis_mask CAM
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cars_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/cars_resnet101.pth --model_name resnet101 --vis_mask GradCAM

# cal
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cars_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/cal_ckpts/cars_resnet101_cal.pth --model_name resnet101 --selector cal --vis_mask GradCAM

for i in $(seq $START $END); do
    python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cars_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/cal_ckpts/cars_resnet101_cal.pth --model_name resnet101 --selector cal --vis_mask bap_${i}
done

# aircraft
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/aircraft_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/aircraft_resnet101.pth --model_name resnet101
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/aircraft_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/aircraft_resnet101.pth --model_name resnet101 --vis_mask CAM
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/aircraft_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/aircraft_resnet101.pth --model_name resnet101 --vis_mask GradCAM

# cal
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/aircraft_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/cal_ckpts/aircraft_resnet101_cal.pth --model_name resnet101 --selector cal --vis_mask GradCAM

for i in $(seq $START $END); do
    python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/aircraft_weakaugs.yaml --serial 42 --batch_size 4 --vis_cols 4 --ckpt_path /hdd/edwin/results_backbones/cal_ckpts/aircraft_resnet101_cal.pth --model_name resnet101 --selector cal --vis_mask bap_${i}
done


# batch size 8

START=0
END=31

# CUB
# baseline
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cub_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/cub_resnet101.pth --model_name resnet101
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cub_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/cub_resnet101.pth --model_name resnet101 --vis_mask CAM
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cub_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/cub_resnet101.pth --model_name resnet101 --vis_mask GradCAM

# cal
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cub_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/cal_ckpts/cub_resnet101_cal.pth --model_name resnet101 --selector cal --vis_mask GradCAM

for i in $(seq $START $END); do
    python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cub_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/cal_ckpts/cub_resnet101_cal.pth --model_name resnet101 --selector cal --vis_mask bap_${i}
done


# cars
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cars_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/cars_resnet101.pth --model_name resnet101
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cars_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/cars_resnet101.pth --model_name resnet101 --vis_mask CAM
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cars_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/cars_resnet101.pth --model_name resnet101 --vis_mask GradCAM

# cal
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cars_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/cal_ckpts/cars_resnet101_cal.pth --model_name resnet101 --selector cal --vis_mask GradCAM

for i in $(seq $START $END); do
    python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/cars_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/cal_ckpts/cars_resnet101_cal.pth --model_name resnet101 --selector cal --vis_mask bap_${i}
done

# aircraft
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/aircraft_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/aircraft_resnet101.pth --model_name resnet101
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/aircraft_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/aircraft_resnet101.pth --model_name resnet101 --vis_mask CAM
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/aircraft_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/ft_ckpts/aircraft_resnet101.pth --model_name resnet101 --vis_mask GradCAM

# cal
python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/aircraft_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/cal_ckpts/aircraft_resnet101_cal.pth --model_name resnet101 --selector cal --vis_mask GradCAM

for i in $(seq $START $END); do
    python tools/vis_dfsm.py --square_resize_random_crop --test_square_resize_center_crop --seed 1 --fp16 --debugging --cfg configs/aircraft_weakaugs.yaml --serial 43 --batch_size 8 --vis_cols 8 --ckpt_path /hdd/edwin/results_backbones/cal_ckpts/aircraft_resnet101_cal.pth --model_name resnet101 --selector cal --vis_mask bap_${i}
done
