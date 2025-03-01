# python download_save_wandb_data.py --serials 0 1 2 3 4 5 --output_file DA_all.csv

# python preprocess_acc.py  --input_file data\test12.csv  --keep_serials 1 --keep_epochs 800 --keep_lr 0.0005 --output_file acc1.csv --keep_datasets aircraft --cont_loss

# python preprocess_acc.py  --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_resnet101 --output_file halo2.csv

# python plot.py  --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_resnet101_tgda --x_var_name dataset_name --output_file testplot --type_plot line

# python plot.py  --input_file D:\my_main\KD_DA\results_all\acc\TGDA_Resnet18-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file testplot --type_plot bar



# ---------------------------------------------
echo "Hello, world!"

python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods resnet18_resnet101_tgda --output_file tgda_resnet18-resnet101.csv

python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_resnet101_tgda --output_file tgda_vit_t16-resnet101.csv

python preprocess_acc.py --input_file data\DA_all.csv --keep_epochs 800 --keep_lr 0.0005 --keep_methods vit_t16_vit_b16_tgda --output_file tgda_vit_t16-vit_b16.csv


echo "DEATH, world!"

python plot.py  --input_file results_all\acc\tgda_resnet18-resnet101.csv --x_var_name serial --hue_var_name dataset_name --output_file testplot --type_plot bar --x_rotation 45 --title 'Different Data Augmentation on Different Dataset Using TGDA (ResNet18-ResNet101)' --fig_size 9 5