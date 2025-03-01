@echo off

@REM python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_resnet101_tgda --output_file tgda_resnet18-resnet101.csv
@REM python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_resnet101_tgda --output_file tgda_vit_t16-resnet101.csv
@REM python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_vit_b16_tgda --output_file tgda_vit_t16-vit_b16.csv

@REM python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_resnet101_cal --output_file cal_resnet18-resnet101.csv
@REM python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_resnet101_cal --output_file cal_vit_t16-resnet101.csv
@REM python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_vit_b16_cal --output_file cal_vit_t16-vit_b16.csv

@REM python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_resnet101_kd --output_file kd_resnet18-resnet101.csv
@REM python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_resnet101_kd --output_file kd_vit_t16-resnet101.csv
@REM python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_vit_b16_kd --output_file kd_vit_t16-vit_b16.csv

@REM python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_ce --output_file ce_resnet18.csv
@REM python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_ce --output_file ce_vit_t16.csv

@REM python plot.py  --input_file results_all\acc\tgda_resnet18-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file "TGDA (ResNet18-ResNet101)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using TGDA (ResNet18-ResNet101)"
@REM python plot.py  --input_file results_all\acc\tgda_vit_t16-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file "TGDA (ViT_t16-ResNet101)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using TGDA (ViT_t16-ResNet101)"
@REM python plot.py  --input_file results_all\acc\tgda_vit_t16-vit_b16.csv --x_var_name serial --hue_var_name dataset_name --output_file "TGDA (ViT_t16-ViT_b16)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using TGDA (ViT_t16-ViT_b16)"

@REM python plot.py  --input_file results_all\acc\cal_resnet18-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file "CAL (ResNet18-ResNet101)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using CAL (ResNet18-ResNet101)"
@REM python plot.py  --input_file results_all\acc\cal_vit_t16-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file "CAL (ViT_t16-ResNet101)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using CAL (ViT_t16-ResNet101)"
@REM python plot.py  --input_file results_all\acc\cal_vit_t16-vit_b16.csv --x_var_name serial --hue_var_name dataset_name --output_file "CAL (ViT_t16-ViT_b16)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using CAL (ViT_t16-ViT_b16)"

@REM python plot.py  --input_file results_all\acc\kd_resnet18-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file "KD (ResNet18-ResNet101)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using KD (ResNet18-ResNet101)"
@REM python plot.py  --input_file results_all\acc\kd_vit_t16-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file "KD (ViT_t16-ResNet101)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using KD (ViT_t16-ResNet101)"
@REM python plot.py  --input_file results_all\acc\kd_vit_t16-vit_b16.csv --x_var_name serial --hue_var_name dataset_name --output_file "KD (ViT_t16-ViT_b16)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using KD (ViT_t16-ViT_b16)"

python plot.py  --input_file results_all\acc\ce_resnet18.csv --x_var_name serial --hue_var_name dataset_name --output_file "CE (ResNet18)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using CE (ResNet18)"
python plot.py  --input_file results_all\acc\ce_vit_t16.csv --x_var_name serial --hue_var_name dataset_name --output_file "CE (ViT_t16)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using CE (ViT_t16)"