#!/bin/bash

#motivation: pt+ft train data (and cost) >> directly train on target datasets
# also allows to train custom architectures for task / resource constraints
python compute_seen_images.py



# visualization of attention
python vis_attention.py --subfolder cub --save_name attention_cub
python vis_attention.py --subfolder cub --vis_all_masks --save_name attention_all_cub

python vis_attention.py --subfolder aircraft --save_name attention_aircraft
python vis_attention.py --subfolder aircraft --vis_all_masks --save_name attention_all_aircraft

python vis_attention.py --subfolder cars --save_name attention_cars
python vis_attention.py --subfolder cars --vis_all_masks --save_name attention_all_cars

# 8 images per row
python vis_attention.py --number_images_per_ds 8 --subfolder cub_8 --save_name attention_cub_8
python vis_attention.py --number_images_per_ds 8 --subfolder cub_8 --vis_all_masks --save_name attention_all_cub_8

python vis_attention.py --number_images_per_ds 8 --subfolder aircraft_8 --save_name attention_aircraft_8
python vis_attention.py --number_images_per_ds 8 --subfolder aircraft_8 --vis_all_masks --save_name attention_all_aircraft_8

python vis_attention.py --number_images_per_ds 8 --subfolder cars_8 --save_name attention_cars_8
python vis_attention.py --number_images_per_ds 8 --subfolder cars_8 --vis_all_masks --save_name attention_all_cars_8

# test images
python vis_attention.py --vis_all_masks --test_images --save_name attention_all_cub_test



# reassign serials from wandb
# python update_wandb_project_serialx_to_serialy --project_name nycu_pcs/KD_TGDA --serial_x 33 --serial_y 53
# python update_wandb_project_serialx_to_serialy --project_name nycu_pcs/KD_TGDA --serial_x 34 --serial_y 54
# python update_wandb_project_serialx_to_serialy_based_on_substr_in_field.py


# download raw data from wandb
# stage2 focused on accuracy since stage1 usually focuses on hparam search (if exists)
python download_save_wandb_data.py --project_name nycu_pcs/Backbones --serials 1 15 --teachers_data --output_file backbones_stage2.csv
python download_save_wandb_data.py --project_name nycu_pcs/KD_DA --serials 0 1 2 3 4 5 --da_data --output_file kd_da_stage2.csv
python download_save_wandb_data.py --project_name nycu_pcs/KD_TGDA --serials 20 21 22 23 24 25 26 27 28 29 31 32 41 42 43 44 45 46 51 52 53 54 61 --output_file kd_tgda_stage2.csv



# extract tables from pdf
# python extract_table_from_pdf.py --input_pdf papers\s3mix.pdf --pdf_page_number 8
# python extract_table_from_pdf.py --input_pdf papers\idmm.pdf --pdf_page_number 9
# # fails for this as it not really a pdf but a printed page / image
# # python extract_table_from_pdf.py --input_pdf papers\dssd_liang_2024.pdf --pdf_page_number 5
# python extract_table_from_pdf.py --input_pdf papers\sit.pdf --pdf_page_number 7
# python extract_table_from_pdf.py --input_pdf papers\gmml.pdf --pdf_page_number 4


# make tables from previous papers into our format
python adapt_paper_tables_to_our_format.py --input_file data/tables_previous_mod/vits_is224_scratch_vs_ssl.csv --serial 25
python adapt_paper_tables_to_our_format.py --input_file data/tables_previous_mod/vits_is448_scratch_vs_pt.csv --serial 24

python adapt_paper_tables_to_our_format.py --input_file data/tables_previous_mod/fgir_is128_rn18.csv --serial 51 --suffix " (128, Res-18)" --project_name prev_is128_rn18
python adapt_paper_tables_to_our_format.py --input_file data/tables_previous_mod/fgir_is128_rn34.csv --serial 52 --suffix " (128, Res-34)" --project_name prev_is128_rn34
python adapt_paper_tables_to_our_format.py --input_file data/tables_previous_mod/fgir_is128_rn50.csv --serial 53 --suffix " (128, Res-50)" --project_name prev_is128_rn50
python adapt_paper_tables_to_our_format.py --input_file data/tables_previous_mod/fgir_is128_rn101.csv --serial 54 --suffix " (128, Res-101)" --project_name prev_is128_rn101

python adapt_paper_tables_to_our_format.py --input_file data/tables_previous_mod/fgir_is448_rn18.csv --serial 23 --suffix " (448, Res-18)" --project_name prev_is448_rn18
python adapt_paper_tables_to_our_format.py --input_file data/tables_previous_mod/fgir_is448_rn34.csv --serial 23 --suffix " (448, Res-34)" --project_name prev_is448_rn34
python adapt_paper_tables_to_our_format.py --input_file data/tables_previous_mod/fgir_is448_rn50.csv --serial 23 --suffix " (448, Res-50)" --project_name prev_is448_rn50
python adapt_paper_tables_to_our_format.py --input_file data/tables_previous_mod/fgir_is448_rn101.csv --serial 23 --suffix " (448, Res-101)" --project_name prev_is448_rn101

python adapt_paper_tables_to_our_format.py --input_file data/tables_previous_mod/sota_is128_rn101.csv --serial 54 --suffix " (IS=128 SotA)" --project_name prev_is128_sota

python adapt_paper_tables_to_our_format.py --input_file data/tables_previous_mod/sota_is448_rn50.csv --serial 23 --suffix " (IS=448 SotA)" --project_name prev_is448_sota_rn50
python adapt_paper_tables_to_our_format.py --input_file data/tables_previous_mod/sota_is448_rn101.csv --serial 23 --suffix " (IS=448 SotA)" --project_name prev_is448_sota_rn101



# stack results from previous papers into a single table
python stack_two_df.py --input_df_1 results_all/melted_tables/vits_is224_scratch_vs_ssl.csv --input_df_2 results_all/melted_tables/vits_is448_scratch_vs_pt.csv --output_file prev_papers.csv
python stack_two_df.py --input_df_1 data/prev_papers.csv --input_df_2 results_all/melted_tables/vits_is448_scratch_vs_pt.csv --output_file prev_papers.csv

python stack_two_df.py --input_df_1 data/prev_papers.csv --input_df_2 results_all/melted_tables/fgir_is128_rn18.csv --output_file prev_papers.csv
python stack_two_df.py --input_df_1 data/prev_papers.csv --input_df_2 results_all/melted_tables/fgir_is128_rn34.csv --output_file prev_papers.csv
python stack_two_df.py --input_df_1 data/prev_papers.csv --input_df_2 results_all/melted_tables/fgir_is128_rn50.csv --output_file prev_papers.csv
python stack_two_df.py --input_df_1 data/prev_papers.csv --input_df_2 results_all/melted_tables/fgir_is128_rn101.csv --output_file prev_papers.csv

python stack_two_df.py --input_df_1 data/prev_papers.csv --input_df_2 results_all/melted_tables/fgir_is448_rn18.csv --output_file prev_papers.csv
python stack_two_df.py --input_df_1 data/prev_papers.csv --input_df_2 results_all/melted_tables/fgir_is448_rn34.csv --output_file prev_papers.csv
python stack_two_df.py --input_df_1 data/prev_papers.csv --input_df_2 results_all/melted_tables/fgir_is448_rn50.csv --output_file prev_papers.csv
python stack_two_df.py --input_df_1 data/prev_papers.csv --input_df_2 results_all/melted_tables/fgir_is448_rn101.csv --output_file prev_papers.csv

python stack_two_df.py --input_df_1 data/prev_papers.csv --input_df_2 results_all/melted_tables/sota_is128_rn101.csv --output_file prev_papers.csv

python stack_two_df.py --input_df_1 data/prev_papers.csv --input_df_2 results_all/melted_tables/sota_is448_rn50.csv --output_file prev_papers.csv
python stack_two_df.py --input_df_1 data/prev_papers.csv --input_df_2 results_all/melted_tables/sota_is448_rn101.csv --output_file prev_papers.csv



# combine into one single dataframe with all data
python stack_two_df.py --input_df_1 data/kd_da_stage2.csv --input_df_2 data/kd_tgda_stage2.csv --output_file kd_da_tgda_stage2.csv
python stack_two_df.py --input_df_1 data/kd_da_tgda_stage2.csv --input_df_2 data/backbones_stage2.csv --output_file kd_da_tgda_backbones_stage2.csv
python stack_two_df.py --input_df_1 data/kd_da_tgda_backbones_stage2.csv --input_df_2 data/prev_papers.csv --output_file kd_da_tgda_backbones_prev.csv


# preprocess raw data
python preprocess_raw_wandb_data.py


# summarize acc
python summarize_acc.py --main_serials 1 15 --keep_settings teacher_ft_224 teacher_cal_224 teacher_cal_448 --output_file acc_teachers
python summarize_acc.py --main_serials 0 1 2 3 4 5 --keep_settings ce_noaug ce_re ce_ta ce_cm ce_mu ce_cmmu kd_noaug kd_re kd_ta kd_cm kd_mu kd_cmmu sod_noaug sod_re sod_ta sod_cm sod_mu sod_cmmu tgda_noaug tgda_re tgda_ta tgda_cm tgda_mu tgda_cmmu --output_file acc_dataaug
python summarize_acc.py --main_serials 20 21 22 23 24 25 26 27 28 29 31 32 51 52 53 54 61 --results_dir results_all/acc_unfiltered --output_file acc_tgda
python summarize_acc.py --main_serials 23 24 25 51 52 53 54 --filter_methods vitfs_tiny_patch16_gap_224_resnet101_hr_vits_tgda_ta_ls_sd_800 vit_t16_resnet101_hr_vits_tgda_ta_ls_sd_800 resnet18d_resnet101_lr_rn18like resnet34d_resnet101_lr_rn34like resnet50d_resnet101_lr_rn50like resnet101d_resnet101_lr_rn101like --output_file acc_tgda



# summarize cost
python summarize_cost.py --keep_methods "resnet18_resnet101_lr_rn18like" "resnet18d_resnet101_lr_rn18like" "lrresnet22_resnet101_lr_rn18like" "lrresnet22d_resnet101_lr_rn18like" "lrresnet22dwrk_resnet101_lr_rn18like" "lrresnet18t2d_resnet101_lr_rn18like" "lrresnet18t2pdwosrnk_resnet101_lr_rn18like" "lrresnet18t2pdwsrnk_resnet101_lr_rn18like" "lrresnet18t4pdwsrnk_resnet101_lr_rn18like" "resnet34_resnet101_lr_rn34like" "resnet34d_resnet101_lr_rn34like" "lrresnet38_resnet101_lr_rn34like" "lrresnet38d_resnet101_lr_rn34like" "lrresnet34t2d_resnet101_lr_rn34like" "lrresnet34t2pdwosrnk_resnet101_lr_rn34like" "lrresnet34t2pdwsrnk_resnet101_lr_rn34like" "lrresnet34t4pdwsrnk_resnet101_lr_rn34like" "resnet50_resnet101_lr_rn50like" "resnet50d_resnet101_lr_rn50like" "lrresnet56_resnet101_lr_rn50like" "lrresnet56d_resnet101_lr_rn50like" "lrresnet50t2d_resnet101_lr_rn50like" "lrresnet50t2pdwosrnk_resnet101_lr_rn50like" "lrresnet50t2pdwsrnk_resnet101_lr_rn50like" "lrresnet50t4pdwsrnk_resnet101_lr_rn50like" "resnet101_resnet101_lr_rn101like" "resnet101d_resnet101_lr_rn101like" "lrresnet107_resnet101_lr_rn101like" "lrresnet107d_resnet101_lr_rn101like" "lrresnet101t2d_resnet101_lr_rn101like" "lrresnet101t2pdwosrnk_resnet101_lr_rn101like" "lrresnet101t2pdwsrnk_resnet101_lr_rn101like" "lrresnet101t4pdwsrnk_resnet101_lr_rn101like" "DSSD (128, Res-18)" "DSSD (128, Res-34)" "DSSD (128, Res-50)" "DSSD (128, Res-101)"


# plots
# acc vs flops
python plot.py --keep_methods "DSSD (128, Res-18)" "DSSD (128, Res-34)" "DSSD (128, Res-50)" "DSSD (128, Res-101)" "lrresnet18t2d_resnet101_lr_rn18like" "lrresnet50t2d_resnet101_lr_rn50like" "lrresnet101t2d_resnet101_lr_rn101like"

# acc vs flops vs params
python plot.py --keep_methods "DSSD (128, Res-18)" "DSSD (128, Res-34)" "DSSD (128, Res-50)" "DSSD (128, Res-101)" "lrresnet18t2d_resnet101_lr_rn18like" "lrresnet50t2d_resnet101_lr_rn50like" "lrresnet101t2d_resnet101_lr_rn101like" --size_var_name no_params_trainable --type_plot scatter --output_file acc_vs_flops_vs_params --title "Accuracy vs FLOPs with Parameter Count as Marker Size" --sizes 40 400 --font_scale 0.8


# tgda improves with longer training: plot acc vs epochs
# can make many plots for this
dataset_array=("aircraft" "cars")

for ds in ${dataset_array[@]}; do
    python download_plot_acc_vs_epoch.py --project_name nycu_pcs/KD_DA --serials 0 --keep_datasets ${ds} --keep_methods resnet18_ce_noaug resnet18_resnet101_kd_noaug resnet18_resnet101_sod_noaug resnet18_resnet101_tgda_noaug --output_file acc_vs_epoch_serial0_${ds}_rn
    python download_plot_acc_vs_epoch.py --project_name nycu_pcs/KD_DA --serials 0 --keep_datasets ${ds} --keep_methods resnet18_ce_noaug resnet18_resnet101_kd_noaug resnet18_resnet101_sod_noaug resnet18_resnet101_tgda_noaug --output_file acc_vs_epoch_serial0_${ds}_rn
done

# python download_plot_acc_vs_epoch.py --project_name nycu_pcs/KD_DA --serials 0 --keep_datasets cub --keep_methods vit_t16_ce_noaug vit_t16_resnet101_kd_noaug vit_t16_resnet101_sod_noaug vit_t16_resnet101_tgda_noaug --output_file acc_vs_epoch_serial0_cub_vit_rn
# python download_plot_acc_vs_epoch.py --project_name nycu_pcs/KD_DA --serials 0 --keep_datasets cub --keep_methods resnet18_ce_noaug resnet18_resnet101_kd_noaug resnet18_resnet101_sod_noaug resnet18_resnet101_tgda_noaug --output_file acc_vs_epoch_serial0_cub_rn



# compute params and flops separately for certain models
# requires installing torchprofile from source: 
# https://github.com/zhijian-liu/torchprofile
# for t2t_vit_7. These numbers do not count einsum which may be a decent chunk
# FLOPs and Number of Parameters:  6.1256 4.2552
python compute_params_flops.py --cfg ../../projects/FGIR-KD/configs/cub_weakaugs.yaml --image_size 448 --model_name deit_tiny_patch16_224
python compute_params_flops.py --cfg ../../projects/FGIR-KD/configs/cub_weakaugs.yaml --image_size 448 --model_name deit_small_patch16_224
python compute_params_flops.py --cfg ../../projects/FGIR-KD/configs/cub_weakaugs.yaml --image_size 448 --model_name deit_base_patch16_224
python compute_params_flops.py --cfg ../../projects/FGIR-KD/configs/cub_weakaugs.yaml --image_size 448 --model_name vit_base_patch16_224
python compute_params_flops.py --cfg ../../projects/FGIR-KD/configs/cub_weakaugs.yaml --image_size 448 --model_name vit_base_r50_s16_224
python compute_params_flops.py --cfg ../../projects/FGIR-KD/configs/cub_weakaugs.yaml --image_size 448 --model_name pvt_v2_b0
python compute_params_flops.py --cfg ../../projects/FGIR-KD/configs/cub_weakaugs.yaml --image_size 448 --model_name pvt_v2_b3
python compute_params_flops.py --cfg ../../projects/FGIR-KD/configs/cub_weakaugs.yaml --image_size 448 --model_name vitfs_tiny_patch16_gap_reg4_dinov2_bn

# copy latex table with current format (--no-escape for $ and other special characters)
python tably.py results_bmvc25/tables/ours_vits_is448.csv
python tably.py results_bmvc25/tables/ours_vits_is448.csv
python tably.py results_bmvc25/tables/ours_vits_is448.csv
python tably.py results_bmvc25/tables/ours_vits_is448.csv
python tably.py results_bmvc25/tables/ours_vits_is448.csv
python tably.py results_bmvc25/tables/ours_vits_is448.csv
python tably.py results_bmvc25/tables/ours_vits_is448.csv


python plot.py --input_file results_bmvc25/tables/ours_vits_is448_avg.csv --y_var_name "Accuracy (%)" --x_var_name "Train Images" --size_var_name "Parameters (M)" --type_plot scatter --output_file vit_acc_vs_images_vs_params --title "Accuracy vs Train Images with Parameters as Marker Size" --log_scale_x --sizes 40 400 --hue_var_name Method --x_label "Number of Train Images (Log-scale)" --loc_legend "lower left" --fig_size 6 4
python plot.py --input_file results_bmvc25/tables/ours_acc_vs_cost_is128.csv --y_var_name "Accuracy (%)" --x_var_name "Parameters (M)" --size_var_name "FLOPs (G)" --type_plot scatter --output_file acc_vs_params_vs_flops --title "Accuracy vs Parameters with FLOPs as Marker Size" --sizes 40 400 --hue_var_name Strategy --x_label "Parameters (M)" --loc_legend "lower left" --fig_size 6 4 --add_text_methods --font_size 11
