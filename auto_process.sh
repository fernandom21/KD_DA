# reassign serials from wandb
python update_wandb_project_serialx_to_serialy --project_name nycu_pcs/KD_TGDA --serial_x 33 --serial_y 53
python update_wandb_project_serialx_to_serialy --project_name nycu_pcs/KD_TGDA --serial_x 34 --serial_y 54

# download raw data from wandb
# stage2 focused on accuracy since stage1 usually focuses on hparam search (if exists)
python download_save_wandb_data.py --project_name nycu_pcs/KD_TGDA --serials 20 21 22 23 24 25 31 32 41 42 43 44 51 52 53 54 61 --output_file kd_tgda_stage2.csv
python download_save_wandb_data.py --project_name nycu_pcs/KD_DA --serials 0 1 2 3 4 5 --da_data --output_file kd_da_stage2.csv
python download_save_wandb_data.py --project_name nycu_pcs/Backbones --serials 1 15 --teachers_data --output_file backbones_stage2.csv

# combine into one single dataframe with all data
python stack_two_df.py --input_df_1 data\kd_da_stage2.csv --input_df_2 data\kd_tgda_stage2.csv --output_file kd_da_tgda_stage2.csv
python stack_two_df.py --input_df_1 data\kd_da_tgda_stage2.csv --input_df_2 data\backbones_stage2.csv --output_file kd_da_tgda_backbones_stage2.csv


# tgda improves with longer training: plot acc vs epochs
python download_plot_acc_vs_epoch.py --serials 0 --keep_datasets cub --keep_methods vit_t16_ce_noaug vit_t16_resnet101_kd_noaug vit_t16_resnet101_kdct_noaug vit_t16_resnet101_tgda_noaug --output_file acc_vs_epoch_serial0_cub_vit_rn
python download_plot_acc_vs_epoch.py --serials 0 --keep_datasets cub --keep_methods resnet18_ce_noaug resnet18_resnet101_kd_noaug resnet18_resnet101_kdct_noaug resnet18_resnet101_tgda_noaug --output_file acc_vs_epoch_serial0_cub_rn


# preprocess raw data
python preprocess_raw.py --keep_epochs 800 --keep_lr 0.0005











# python preprocess_acc.py  --input_file data\test12.csv  --keep_serials 1 --keep_epochs 800 --keep_lr 0.0005 --output_file acc1.csv --keep_datasets aircraft --cont_loss

# python preprocess_acc.py  --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_resnet101 --output_file halo2.csv

# python plot.py  --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_resnet101_tgda --x_var_name dataset_name --output_file testplot --type_plot line

# python plot.py  --input_file D:\my_main\KD_DA\results_all\acc\TGDA_Resnet18-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file testplot --type_plot bar


# ---------------------------------------------
python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_resnet101_tgda --output_file tgda_resnet18-resnet101.csv

python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_resnet101_tgda --output_file tgda_vit_t16-resnet101.csv

python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_vit_b16_tgda --output_file tgda_vit_t16-vit_b16.csv

python plot.py  --input_file results_all\acc\tgda_resnet18-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file testplot --type_plot bar --x_rotation 45 --title 'Different Data Augmentation on Different Dataset Using TGDA (ResNet18-ResNet101)' --fig_size 9 5


python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_resnet101_tgda --output_file tgda_resnet18-resnet101.csv
python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_resnet101_tgda --output_file tgda_vit_t16-resnet101.csv
python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_vit_b16_tgda --output_file tgda_vit_t16-vit_b16.csv

python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_resnet101_cal --output_file cal_resnet18-resnet101.csv
python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_resnet101_cal --output_file cal_vit_t16-resnet101.csv
python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_vit_b16_cal --output_file cal_vit_t16-vit_b16.csv

python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_resnet101_kd --output_file kd_resnet18-resnet101.csv
python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_resnet101_kd --output_file kd_vit_t16-resnet101.csv
python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_vit_b16_kd --output_file kd_vit_t16-vit_b16.csv

python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_ce --output_file ce_resnet18.csv
python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_ce --output_file ce_vit_t16.csv

python plot.py  --input_file results_all\acc\tgda_resnet18-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file "TGDA (ResNet18-ResNet101)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using TGDA (ResNet18-ResNet101)"
python plot.py  --input_file results_all\acc\tgda_vit_t16-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file "TGDA (ViT_t16-ResNet101)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using TGDA (ViT_t16-ResNet101)"
python plot.py  --input_file results_all\acc\tgda_vit_t16-vit_b16.csv --x_var_name serial --hue_var_name dataset_name --output_file "TGDA (ViT_t16-ViT_b16)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using TGDA (ViT_t16-ViT_b16)"

python plot.py  --input_file results_all\acc\cal_resnet18-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file "CAL (ResNet18-ResNet101)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using CAL (ResNet18-ResNet101)"
python plot.py  --input_file results_all\acc\cal_vit_t16-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file "CAL (ViT_t16-ResNet101)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using CAL (ViT_t16-ResNet101)"
python plot.py  --input_file results_all\acc\cal_vit_t16-vit_b16.csv --x_var_name serial --hue_var_name dataset_name --output_file "CAL (ViT_t16-ViT_b16)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using CAL (ViT_t16-ViT_b16)"

python plot.py  --input_file results_all\acc\kd_resnet18-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file "KD (ResNet18-ResNet101)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using KD (ResNet18-ResNet101)"
python plot.py  --input_file results_all\acc\kd_vit_t16-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file "KD (ViT_t16-ResNet101)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using KD (ViT_t16-ResNet101)"
python plot.py  --input_file results_all\acc\kd_vit_t16-vit_b16.csv --x_var_name serial --hue_var_name dataset_name --output_file "KD (ViT_t16-ViT_b16)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using KD (ViT_t16-ViT_b16)"

python plot.py  --input_file results_all\acc\ce_resnet18.csv --x_var_name serial --hue_var_name dataset_name --output_file "CE (ResNet18)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using CE (ResNet18)"
python plot.py  --input_file results_all\acc\ce_vit_t16.csv --x_var_name serial --hue_var_name dataset_name --output_file "CE (ViT_t16)" --type_plot bar --x_rotation 45 --fig_size 9 5 --title "Different Data Augmentation on Different Dataset Using CE (ViT_t16)"
