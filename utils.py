import numpy as np
import pandas as pd



# SERIAL_REASSIGN = {
#     0: 'No Augmentation',
#     1: 'Random Erasing', 
#     2: 'Trivial Aug.', 
#     3: 'CutMix', 
#     4: 'Mixup', 
#     5: 'CutMix+Mixup',
# }

# SETTING_EXPLANATIONS = [
#     'ce', 
#     'kd', 
#     'sod', 
#     'tgda',
# ]

# methods dic
    # KD DA project
    # 'resnet18_resnet101_tgda': 'RN18-RN101 (TGDA)',
    # 'vit_t16_resnet101_tgda': 'ViT T/16-RN101 (TGDA)',
    # 'vit_t16_vit_b16_tgda': 'ViT T/16-ViT B/16 (TGDA)',

    # 'resnet18_resnet101_sod': 'RN18-RN101 (SOD)',
    # 'vit_t16_resnet101_sod': 'ViT T/16-RN101 (SOD)',
    # 'vit_t16_vit_b16_sod': 'ViT T/16-ViT B/16 (SOD)',

    # 'resnet18_resnet101_kd': 'RN18-RN101 (KD)',
    # 'vit_t16_resnet101_kd': 'ViT T/16-RN101 (KD)',
    # 'vit_t16_vit_b16_kd': 'ViT T/16-ViT B/16 (KD)',

    # 'resnet18_ce': 'RN18 (CE)',
    # 'vit_t16_ce': 'ViT T/16 (CE)',

    # 'KD*': 'KD (DSSD)',
    # 'RKD*': 'RKD (DSSD)',
    # 'CRD*': 'CRD (DSSD)',
    # 'OFD*': 'OFD (DSSD)',
    # 'ReviewKD*': 'ReviewKD (DSSD)',
    # 'DKD*': 'DKD (DSSD)',
    # 'SnapMix*': 'SnapMix (DSSD)',
    # 'DSSD': 'DSSD',

    # 'RN18FT': 'RN18FT',
    # 'RN34FT': 'RN34FT',
    # 'RN50FT': 'RN50FT',
    # 'RN101FT': 'RN101FT',
    # 'RN18FT*': 'RN18FT*',
    # 'RN34FT*': 'RN34FT*',
    # 'RN50FT*': 'RN50FT*',
    # 'RN101FT*': 'RN101FT*',

    # 'RA-CNN (VGG) (IS=448 SotA)': 'RA-CNN (VGG) (IS=448 SotA)',
    # 'MA-CNN (VGG) (IS=448 SotA)': 'MA-CNN (VGG) (IS=448 SotA)',
    # 'Res-50': 'RN50FT**',
    # 'Res-101': 'RN101FT**',
    # 'PCA-Net': 'PCA-Net*',
    # 'PMG*': 'PMG*',
    # 'API-Net*': 'API-Net*',
    # 'MGN-CNN*': 'MGN-CNN*',
    # 'Res-50+DSSD': 'RN50+DSSD',


# def reassign_serial(df):
#     df['serial'] = df['serial'].apply(lambda x: SERIAL_REASSIGN.get(x, x))
#     return df


# def determine_ila_method(row):
#     if row['ila'] == True and row['ila_locs'] == '[]':
#         return "_ila_dso"
#     elif row['ila'] == True:
#         return "_ila"
#     else:
#         return ""



# def filter_bool_df(df, square_resize_random_crop=None, pretrained=None, cont_loss=None):
#     if square_resize_random_crop:
#         df = df[df['square_resize_random_crop'] == square_resize_random_crop]
    
#     if pretrained:
#         df = df[df['pretrained'] == pretrained]

#     if cont_loss:
#         df = df[df['cont_loss'] == cont_loss]

#     # if square_resize_random_crop:
#     #     df = df[df["square_resize_random_crop"]==True]
#     # if square_resize_random_crop is False:  
#     #     df = df[df["square_resize_random_crop"]==False]

#     # if pretrained:
#     #     df = df[df["pretrained"]==True]
#     # if pretrained is False:
#     #     df = df[df["pretrained"]==False]

#     # if cont_loss:
#     #     df = df[df['cont_loss']==True]
#     # if cont_loss is False:
#     #     df = df[df["cont_loss"]==False]

#     return df

# setting function
        # ((df['model_name_teacher'] == '')),
        # ((df['tgda'] == False) & (df['selector'] == '')),
        # ((df['tgda'] == False) & (df['selector'] == 'cal')),
        # ((df['tgda'] == True)),


        # (df['model_name_teacher'] == ''),
        # (df['tgda'] == False) & (df['selector'] == ''),
        # (df['tgda'] == False) & (df['selector'] == 'cal'),
        # (df['tgda'] == True),


    # df['setting'] = np.select(conditions, SETTING_EXPLANATIONS, default='')


# def add_data_aug(df):
#     conditions = [
#         (df['serial'] == 0),
#         (df['serial'] == 1),
#         (df['serial'] == 2),
#         (df['serial'] == 3),
#         (df['serial'] == 4),
#         (df['serial'] == 5),
#     ]

#     df['data_aug'] = np.select(conditions, SERIALS_EXPLANATIONS, default='')
#     return df


# standarize_df
    # df = df.fillna({'classifier': '', 'selector': '', 'adapter': '', 'prompt': '',
    #                 'model_name_teacher': '', 'base_lr': '', 'throughput': ''})

    # df = add_data_aug(df)

    # df['ila_str'] = df.apply(determine_ila_method, axis=1)
    # df['classifier_str'] = df['classifier'].apply(lambda x: f'_{x}' if x else '')
    # df['selector_str'] = df['selector'].apply(lambda x: f'_{x}' if x else '')
    # df['adapter_str'] = df['adapter'].apply(lambda x: f'_{x}' if x else '')
    # df['prompt_str'] = df['prompt'].apply(lambda x: f'_{x}' if x else '')
    # df['transfer_learning_str'] = df['transfer_learning'].apply(lambda x: '_saw' if x is True else '')
    # df['freeze_backbone_str'] = df['freeze_backbone'].apply(lambda x: '_fz' if x is True else '')
    # df['method'] = df['model_name'] + df['ila_str'] + df['classifier_str'] + df['selector_str'] + df['adapter_str'] + df['prompt_str'] + df['transfer_learning_str'] + df['freeze_backbone_str']

    # df['model_name_teacher_str'] = df['model_name_teacher'].apply(lambda x: f'_{x}' if x else '')
    # df['setting_str'] = df['setting'].apply(lambda x: f'_{x}' if x else '')

    # df['method'] = df['model_name'] + df['model_name_teacher_str'] + df['setting_str']

    # df['data_aug_str'] = df['data_aug'].apply(lambda x: f'_{x}' if x else '')

    # df['method'] = df['model_name'] + df['model_name_teacher_str'] + df['setting_str'] + df['data_aug_str']




# filter_df
    # if keep_lr:
    #     df = df[df['lr'].isin(keep_lr)]

    # if keep_epochs:
    #     df = df[df['epochs'].isin(keep_epochs)]


# def preprocess_df(
#     input_file, type='acc', keep_datasets=None, keep_methods=None, keep_serials=None, keep_lr=None, keep_epochs=None,
#     filter_datasets=None, filter_methods=None, filter_serials=None, filter_lr=None, filter_epochs=None,
#     square_resize_random_crop=True, pretrained=False, cont_loss=False):
# def preprocess_df(
#     df, type='all', keep_datasets=None, keep_methods=None, keep_serials=None,
#     filter_datasets=None, filter_methods=None, filter_serials=None,
#     keep_lr=None, # keep_epochs=None,
#     square_resize_random_crop=None, pretrained=None, cont_loss=None,
#     ):

    # filter
    # df = filter_df(df, keep_datasets, keep_methods, keep_serials, keep_lr, keep_epochs,
    #                filter_datasets, filter_methods, filter_serials, filter_lr, filter_epochs)
    # df = filter_df(
    #     df, keep_datasets, keep_methods, keep_serials,
    #     filter_datasets, filter_methods, filter_serials,
    #     keep_lr, # keep_epochs,
    # )


    # filter bool: backbones and certain projects do not have certain variables
    # so end up being filtered, also pretrained or cont loss should be mixed
    # into method description to have a single method that describes the whole
    # set of important hparams (easier to work later with pivot or
    # the filter method functions)
    # df = filter_bool_df(df, square_resize_random_crop, pretrained, cont_loss)


    # saw results (serial 91, 101 -> 1, 3)
    # to be consistent with the naming criteria in the other files / results
    # df = reassign_serial(df)

    # serial explanations
    # 'No Augmentation',
    # 'Random Erasing', 
    # 'Trivial Aug.', 
    # 'CutMix', 
    # 'Mixup', 
    # 'CutMix+Mixup',

              # keep_lr=None,):  # keep_epochs=None,):



SERIALS_EXPLANATIONS = [
    'prev',

    'prev_is128_rn18',
    'prev_is128_rn34',
    'prev_is128_rn50',
    'prev_is128_rn101',

    'prev_is448_rn18',
    'prev_is448_rn34',
    'prev_is448_rn50',
    'prev_is448_rn101',

    'prev_is128_sota',

    'prev_is448_sota_rn50',
    'prev_is448_sota_rn101',

    'teacher_224',
    'teacher_cal_224',
    'teacher_cal_448',

    # Data augmentation method (standard resolution: 224, 800 epochs) and train framework
    # 'ce',
    'ce_noaug',
    'ce_re',
    'ce_ta',
    'ce_cm',
    'ce_mu',
    'ce_cmmu',

    # 'kd',
    'kd_noaug',
    'kd_re',
    'kd_ta',
    'kd_cm',
    'kd_mu',
    'kd_cmmu',

    # 'sod',
    'sod_noaug',
    'sod_re',
    'sod_ta',
    'sod_cm',
    'sod_mu',
    'sod_cmmu',

    # 'tgda',
    'tgda_noaug',
    'tgda_re',
    'tgda_ta',
    'tgda_cm',
    'tgda_mu',
    'tgda_cmmu',

    # high-resolution no pt
    'hr_ce_200',
    'hr_sod_200',
    'hr_tgda_200',
    'hr_tgda_800',

    'hr_vits_tgda_ta_ls_sd_800',
    'sr_vits_tgda_ta_ls_sd_800',
    'hr_vits_tgda_tl_aircraft',
    'hr_vits_tgda_tl_cars',
    'hr_vits_tgda_tl_cub',
    'hr_vits_ce',

    # high-resolution with pt
    'hr_sod_pt_200',
    'hr_tgda_pt_200',

    # cost metrics (time, vram/memory, flops, params)
    'lr_rn18like_train_cost',
    'lr_rn34like_train_cost',
    'lr_rn50like_train_cost',
    'lr_rn101like_train_cost',
    'lr_inference_cost_gpu',
    'lr_inference_cost_cpu',    

    # low-resolution with no pt
    'lr_rn18like',
    'lr_rn34like',
    'lr_rn50like',
    'lr_rn101like',

    # low-resolution with pt
    'lr_tgda_pt_200',
]


SETTINGS_DIC = {
    'prev': 'Previous SotA',

    'prev_is128_rn18': 'Previous SotA',
    'prev_is128_rn34': 'Previous SotA',
    'prev_is128_rn50': 'Previous SotA',
    'prev_is128_rn101': 'Previous SotA',

    'prev_is448_rn18': 'Previous SotA',
    'prev_is448_rn34': 'Previous SotA',
    'prev_is448_rn50': 'Previous SotA',
    'prev_is448_rn101': 'Previous SotA',

    'ce': 'CE',
    'ce_noaug': 'CE (HF)',
    'ce_re': 'CE (Random Erasing)',
    'ce_ta': 'CE (Trivial Aug.)',
    'ce_cm': 'CE (CutMix)',
    'ce_mu': 'CE (Mixup)',
    'ce_cmmu': 'CE (CutMix+Mixup)',

    'kd': 'KD',
    'kd_noaug': 'KD (HF)',
    'kd_re': 'KD (Random Erasing)',
    'kd_ta': 'KD (Trivial Aug.)',
    'kd_cm': 'KD (CutMix)',
    'kd_mu': 'KD (Mixup)',
    'kd_cmmu': 'KD (CutMix+Mixup)',

    'sod': 'SOD',
    'sod_noaug': 'SOD (HF)',
    'sod_re': 'SOD (Random Erasing)',
    'sod_ta': 'SOD (Trivial Aug.)',
    'sod_cm': 'SOD (CutMix)',
    'sod_mu': 'SOD (Mixup)',
    'sod_cmmu': 'SOD (CutMix+Mixup)',

    'tgda': 'TGDA',
    'tgda_noaug': 'TGDA (HF)',
    'tgda_re': 'TGDA (Random Erasing)',
    'tgda_ta': 'TGDA (Trivial Aug.)',
    'tgda_cm': 'TGDA (CutMix)',
    'tgda_mu': 'TGDA (Mixup)',
    'tgda_cmmu': 'TGDA (CutMix+Mixup)',

    # high-resolution no pt
    'hr_ce_200': 'CE',
    'hr_sod_200': 'SOD',
    'hr_tgda_200': 'TGDA',
    'hr_tgda_800': 'TGDA',

    'hr_vits_tgda_ta_ls_sd_800': 'TGDA',
    'sr_vits_tgda_ta_ls_sd_800': 'TGDA (IS=224)',
    'hr_vits_tgda_tl_aircraft': 'TGDA+TL (Air)',
    'hr_vits_tgda_tl_cars': 'TGDA+TL (Cars)',
    'hr_vits_tgda_tl_cub': 'TGDA+TL (CUB)',
    'hr_vits_ce': 'CE',

    # high-resolution with pt
    'hr_sod_pt_200': 'TGDA',
    'hr_tgda_pt_200': 'TGDA',

    # cost metrics (time, vram/memory, flops, params)
    'lr_rn18like_cost': 'TGDA',
    'lr_rn34like_cost': 'TGDA',
    'lr_rn50like_cost': 'TGDA',
    'lr_rn101like_cost': 'TGDA',

    # low-resolution with no pt
    'lr_rn18like': 'TGDA',
    'lr_rn34like': 'TGDA',
    'lr_rn50like': 'TGDA',
    'lr_rn101like': 'TGDA',

    # low-resolution with pt
    'lr_tgda_pt_200': 'TGDA',


    # others
    'prev_is128_sota': 'Previous SotA',

    'prev_is448_sota_rn50': 'Previous SotA',
    'prev_is448_sota_rn101': 'Previous SotA',
}



DATASETS_DIC = {
    'aircraft': 'Aircraft',
    'cars': 'Cars',
    'cub': 'CUB',
    'soygene': 'SoyGene',
    'soylocal': 'SoyLocal',
    'all': 'Average',
}


METHODS_DIC = {
    # IDMM: ViTs from Scratch
    'random init.': 'Scratch',
    'SimCLR': 'SimCLR',
    'SupCon': 'SupCon',
    'MoCov2': 'MoCov2',
    'MoCov3': 'MoCov3',
    'DINO': 'DINO',
    'IDMM': 'IDMM',

    'DeiT-T (IN1k PT)': 'DeiT-T (IN1k PT)',
    'DeiT-T (IDMM)': 'DeiT-T (IDMM)',
    'DeiT-B (IN1k PT)': 'DeiT-B (IN1k PT)',
    'DeiT-B (IDMM)': 'DeiT-B (IDMM)',
    'PVTv2-B0 (IN1k PT)': 'PVTv2-B0 (IN1k PT)',
    'PVTv2-B0 (IDMM)': 'PVTv2-B0 (IDMM)',
    'PVTv2-B3 (IN1k PT)': 'PVTv2-B3 (IN1k PT)',
    'PVTv2-B3 (IDMM)': 'PVTv2-B3 (IDMM)',
    'T2T-ViT-7 (IN1k PT)': 'T2T-ViT-7 (IN1k PT)',
    'T2T-ViT-7 (IDMM)': 'T2T-ViT-7 (IDMM)',


    # IS=128 KD & Data Aug Comparisons
    'Baseline (128, Res-18)': 'Baseline (RN18)',
    'KD (128, Res-18)': 'KD (RN18)',
    'RKD (128, Res-18)': 'RKD (RN18)',
    'CRD (128, Res-18)': 'CRD (RN18)',
    'OFD (128, Res-18)': 'OFD (RN18)',
    'ReviewKD (128, Res-18)': 'ReviewKD (RN18)',
    'DKD (128, Res-18)': 'DKD (RN18)',
    'SnapMix (128, Res-18)': 'SnapMix (RN18)',
    'DSSD (128, Res-18)': 'DSSD (RN18)',
    'DADKD (128, Res-18)': 'DADKD (RN18)',

    'Baseline (128, Res-34)': 'Baseline (RN34)',
    'KD (128, Res-34)': 'KD (RN34)',
    'RKD (128, Res-34)': 'RKD (RN34)',
    'CRD (128, Res-34)': 'CRD (RN34)',
    'OFD (128, Res-34)': 'OFD (RN34)',
    'ReviewKD (128, Res-34)': 'ReviewKD (RN34)',
    'DKD (128, Res-34)': 'DKD (RN34)',
    'SnapMix (128, Res-34)': 'SnapMix (RN34)',
    'DSSD (128, Res-34)': 'DSSD (RN34)',
    'DADKD (128, Res-34)': 'DADKD (RN34)',

    'Baseline (128, Res-50)': 'Baseline (RN50)',
    'KD (128, Res-50)': 'KD (RN50)',
    'RKD (128, Res-50)': 'RKD (RN50)',
    'CRD (128, Res-50)': 'CRD (RN50)',
    'OFD (128, Res-50)': 'OFD (RN50)',
    'ReviewKD (128, Res-50)': 'ReviewKD (RN50)',
    'DKD (128, Res-50)': 'DKD (RN50)',
    'SnapMix (128, Res-50)': 'SnapMix (RN50)',
    'DSSD (128, Res-50)': 'DSSD (RN50)',
    'DADKD (128, Res-50)': 'DADKD (RN50)',

    'Baseline (128, Res-101)': 'Baseline (RN101)',
    'KD (128, Res-101)': 'KD (RN101)',
    'RKD (128, Res-101)': 'RKD (RN101)',
    'CRD (128, Res-101)': 'CRD (RN101)',
    'OFD (128, Res-101)': 'OFD (RN101)',
    'ReviewKD (128, Res-101)': 'ReviewKD (RN101)',
    'DKD (128, Res-101)': 'DKD (RN101)',
    'SnapMix (128, Res-101)': 'SnapMix (RN101)',
    'DSSD (128, Res-101)': 'DSSD (RN101)',
    'DADKD (128, Res-101)': 'DADKD (RN101)',


    # IS=448 Data Aug Comparisons
    'Baseline (448, Res-18)': 'Baseline (RN18)',
    'Mixup (448, Res-18)': 'Mixup (RN18)',
    'CutMix (448, Res-18)': 'CutMix (RN18)',
    'Cutout (448, Res-18)': 'Cutout (RN18)',
    'InPS (448, Res-18)': 'InPS (RN18)',
    'SnapMix (448, Res-18)': 'SnapMix (RN18)',
    'S3Mix (448, Res-18)': 'S3Mix (RN18)',
    'S3Mix+ (448, Res-18)': 'S3Mix+ (RN18)',
    'CEKD (448, Res-18)': 'CEKD (RN18)',

    'Baseline (448, Res-34)': 'Baseline (RN34)',
    'Mixup (448, Res-34)': 'Mixup (RN34)',
    'CutMix (448, Res-34)': 'CutMix (RN34)',
    'Cutout (448, Res-34)': 'Cutout (RN34)',
    'InPS (448, Res-34)': 'InPS (RN34)',
    'SnapMix (448, Res-34)': 'SnapMix (RN34)',
    'S3Mix (448, Res-34)': 'S3Mix (RN34)',
    'S3Mix+ (448, Res-34)': 'S3Mix+ (RN34)',
    'CEKD (448, Res-34)': 'CEKD (RN34)',

    'Baseline (448, Res-50)': 'Baseline (RN50)',
    'Mixup (448, Res-50)': 'Mixup (RN50)',
    'CutMix (448, Res-50)': 'CutMix (RN50)',
    'Cutout (448, Res-50)': 'Cutout (RN50)',
    'InPS (448, Res-50)': 'InPS (RN50)',
    'SnapMix (448, Res-50)': 'SnapMix (RN50)',
    'S3Mix (448, Res-50)': 'S3Mix (RN50)',
    'S3Mix+ (448, Res-50)': 'S3Mix+ (RN50)',
    'CEKD (448, Res-50)': 'CEKD (RN50)',

    'Baseline (448, Res-101)': 'Baseline (RN101)',
    'Mixup (448, Res-101)': 'Mixup (RN101)',
    'CutMix (448, Res-101)': 'CutMix (RN101)',
    'Cutout (448, Res-101)': 'Cutout (RN101)',
    'InPS (448, Res-101)': 'InPS (RN101)',
    'SnapMix (448, Res-101)': 'SnapMix (RN101)',
    'S3Mix (448, Res-101)': 'S3Mix (RN101)',
    'S3Mix+ (448, Res-101)': 'S3Mix+ (RN101)',
    'CEKD (448, Res-101)': 'CEKD (RN101)',


    # ours
    # Backbones project: teachers
    'resnet101_teacher_cal_224': 'RN101',
    'vit_b16_teacher_cal_224': 'ViT-B/16',
    'resnet101_teacher_cal_448': 'RN101',
    'resnet101.tv_in1k_teacher_cal_448': 'RN101',
    'vit_b16_teacher_cal_448': 'ViT-B/16',

    # KD_DA
    # ResNet18 student + ResNet101 teacher
    'resnet18_ce_noaug': 'RN18 (CE, HF)',
    'resnet18_resnet101_kd_noaug': 'RN18 (KD, HF)',
    'resnet18_resnet101_sod_noaug': 'RN18 (SOD, HF)',
    'resnet18_resnet101_tgda_noaug': 'RN18 (TGDA, HF)',

    'resnet18_ce_re': 'RN18 (CE, RE)',
    'resnet18_resnet101_kd_re': 'RN18 (KD, RE)',
    'resnet18_resnet101_sod_re': 'RN18 (SOD, RE)',
    'resnet18_resnet101_tgda_re': 'RN18 (TGDA, RE)',

    'resnet18_ce_ta': 'RN18 (CE, TA)',
    'resnet18_resnet101_kd_ta': 'RN18 (KD, TA)',
    'resnet18_resnet101_sod_ta': 'RN18 (SOD, TA)',
    'resnet18_resnet101_tgda_ta': 'RN18 (TGDA, TA)',

    'resnet18_ce_cm': 'RN18 (CE, CM)',
    'resnet18_resnet101_kd_cm': 'RN18 (KD, CM)',
    'resnet18_resnet101_sod_cm': 'RN18 (SOD, CM)',
    'resnet18_resnet101_tgda_cm': 'RN18 (TGDA, CM)',

    'resnet18_ce_mu': 'RN18 (CE, MU)',
    'resnet18_resnet101_kd_mu': 'RN18 (KD, MU)',
    'resnet18_resnet101_sod_mu': 'RN18 (SOD, MU)',
    'resnet18_resnet101_tgda_mu': 'RN18 (TGDA, MU)',

    'resnet18_ce_cmmu': 'RN18 (CE, CM+MU)',
    'resnet18_resnet101_kd_cmmu': 'RN18 (KD, CM+MU)',
    'resnet18_resnet101_sod_cmmu': 'RN18 (SOD, CM+MU)',
    'resnet18_resnet101_tgda_cmmu': 'RN18 (TGDA, CM+MU)',

    'vit_t16_ce_noaug': 'ViT-T/16 (CE, HF)',
    'vit_t16_resnet101_kd_noaug': 'ViT-T/16 (KD, HF)',
    'vit_t16_resnet101_sod_noaug': 'ViT-T/16 (SOD, HF)',
    'vit_t16_resnet101_tgda_noaug': 'ViT-T/16 (TGDA, HF)',

    # ViT T-16 student + ResNet101 teacher
    'vit_t16_ce_re': 'ViT-T/16 (CE, RE)',
    'vit_t16_resnet101_kd_re': 'ViT-T/16 (KD, RE)',
    'vit_t16_resnet101_sod_re': 'ViT-T/16 (SOD, RE)',
    'vit_t16_resnet101_tgda_re': 'ViT-T/16 (TGDA, RE)',

    'vit_t16_ce_ta': 'ViT-T/16 (CE, TA)',
    'vit_t16_resnet101_kd_ta': 'ViT-T/16 (KD, TA)',
    'vit_t16_resnet101_sod_ta': 'ViT-T/16 (SOD, TA)',
    'vit_t16_resnet101_tgda_ta': 'ViT-T/16 (TGDA, TA)',

    'vit_t16_ce_cm': 'ViT-T/16 (CE, CM)',
    'vit_t16_resnet101_kd_cm': 'ViT-T/16 (KD, CM)',
    'vit_t16_resnet101_sod_cm': 'ViT-T/16 (SOD, CM)',
    'vit_t16_resnet101_tgda_cm': 'ViT-T/16 (TGDA, CM)',

    'vit_t16_ce_mu': 'ViT-T/16 (CE, MU)',
    'vit_t16_resnet101_kd_mu': 'ViT-T/16 (KD, MU)',
    'vit_t16_resnet101_sod_mu': 'ViT-T/16 (SOD, MU)',
    'vit_t16_resnet101_tgda_mu': 'ViT-T/16 (TGDA, MU)',

    'vit_t16_ce_cmmu': 'ViT-T/16 (CE, CM+MU)',
    'vit_t16_resnet101_kd_cmmu': 'ViT-T/16 (KD, CM+MU)',
    'vit_t16_resnet101_sod_cmmu': 'ViT-T/16 (SOD, CM+MU)',
    'vit_t16_resnet101_tgda_cmmu': 'ViT-T/16 (TGDA, CM+MU)',

    # ViT T-16 student + ViT B-16 teacher
    'vit_t16_vit_b16_kd_noaug': 'ViT-T/16 (ViT-B/16 KD, HF)',
    'vit_t16_vit_b16_sod_noaug': 'ViT-T/16 (ViT-B/16 SOD, HF)',
    'vit_t16_vit_b16_tgda_noaug': 'ViT-T/16 (ViT-B/16 TGDA, HF)',

    'vit_t16_vit_b16_kd_re': 'ViT-T/16 (ViT-B/16 KD, RE)',
    'vit_t16_vit_b16_sod_re': 'ViT-T/16 (ViT-B/16 SOD, RE)',
    'vit_t16_vit_b16_tgda_re': 'ViT-T/16 (ViT-B/16 TGDA, RE)',

    'vit_t16_vit_b16_kd_ta': 'ViT-T/16 (ViT-B/16 KD, TA)',
    'vit_t16_vit_b16_sod_ta': 'ViT-T/16 (ViT-B/16 SOD, TA)',
    'vit_t16_vit_b16_tgda_ta': 'ViT-T/16 (ViT-B/16 TGDA, TA)',

    'vit_t16_vit_b16_kd_cm': 'ViT-T/16 (ViT-B/16 KD, CM)',
    'vit_t16_vit_b16_sod_cm': 'ViT-T/16 (ViT-B/16 SOD, CM)',
    'vit_t16_vit_b16_tgda_cm': 'ViT-T/16 (ViT-B/16 TGDA, CM)',

    'vit_t16_vit_b16_kd_mu': 'ViT-T/16 (ViT-B/16 KD, MU)',
    'vit_t16_vit_b16_sod_mu': 'ViT-T/16 (ViT-B/16 SOD, MU)',
    'vit_t16_vit_b16_tgda_mu': 'ViT-T/16 (ViT-B/16 TGDA, MU)',

    'vit_t16_vit_b16_kd_cmmu': 'ViT-T/16 (ViT-B/16 KD, CM+MU)',
    'vit_t16_vit_b16_sod_cmmu': 'ViT-T/16 (ViT-B/16 SOD, CM+MU)',
    'vit_t16_vit_b16_tgda_cmmu': 'ViT-T/16 (ViT-B/16 TGDA, CM+MU)',


    # KD_TGDA project
    # vits from scratch
    'vit_t16_resnet101_hr_vits_tgda_ta_ls_sd_800': 'ViT (TGDA)',
    'vitfs_tiny_patch16_gap_224_resnet101_hr_vits_tgda_ta_ls_sd_800': 'ViT+ConvStem (TGDA)',
    'vitfs_tiny_patch16_gap_reg4_dinov2_resnet101_hr_vits_tgda_ta_ls_sd_800': 'ViTFS (TGDA)',

    'vit_t16_resnet101_sr_vits_tgda_ta_ls_sd_800': 'ViT (TGDA) (IS=224)',
    'vitfs_tiny_patch16_gap_reg4_dinov2_resnet101_sr_vits_tgda_ta_ls_sd_800': 'ViTFS (TGDA) (IS=224)',

    'vitfs_tiny_patch16_gap_reg4_dinov2_hr_vitfs_tgda_tl_aircraft': 'ViTFS (TL, Air)',
    'vitfs_tiny_patch16_gap_reg4_dinov2_hr_vitfs_tgda_tl_cars': 'ViTFS (TL, Cars)',
    'vitfs_tiny_patch16_gap_reg4_dinov2_hr_vitfs_tgda_tl_cub': 'ViTFS (TL, CUB)',

    'vitfs_tiny_patch16_gap_reg4_dinov2_hr_vitfs_ce': 'ViTFS (CE)',


    # low-resolution tgda from scratch
    'resnet18_resnet101_lr_rn18like': 'RN18',
    'resnet18d_resnet101_lr_rn18like': 'RN18D',
    'lrresnet22_resnet101_lr_rn18like': 'LRN22',
    'lrresnet22d_resnet101_lr_rn18like': 'LRN22D',
    'lrresnet22dwrk_resnet101_lr_rn18like': 'LRN22DWRK',
    'lrresnet18t2d_resnet101_lr_rn18like': 'LRN18T2D',
    'lrresnet18t2pdwosrnk_resnet101_lr_rn18like': 'LRN18T2PDWOSRNK',
    'lrresnet18t2pdwsrnk_resnet101_lr_rn18like': 'LRN18T2PDWSRNK',
    'lrresnet18t4pdwsrnk_resnet101_lr_rn18like': 'LRN18T4PDWSRNK',

    'resnet34_resnet101_lr_rn34like': 'RN34',
    'resnet34d_resnet101_lr_rn34like': 'RN34D',
    'lrresnet38_resnet101_lr_rn34like': 'LRN38',
    'lrresnet38d_resnet101_lr_rn34like': 'LRN38D',
    'lrresnet38dwrk_resnet101_lr_rn34like': 'LRN38DWRK',
    'lrresnet34t2d_resnet101_lr_rn34like': 'LRN34T2D',
    'lrresnet34t2pdwosrnk_resnet101_lr_rn34like': 'LRN34T2PDWOSRNK',
    'lrresnet34t2pdwsrnk_resnet101_lr_rn34like': 'LRN34T2PDWSRNK',
    'lrresnet34t4pdwsrnk_resnet101_lr_rn34like': 'LRN34T4PDWSRNK',

    'resnet50_resnet101_lr_rn50like': 'RN50',
    'resnet50d_resnet101_lr_rn50like': 'RN50D',
    'lrresnet56_resnet101_lr_rn50like': 'LRN56',
    'lrresnet56d_resnet101_lr_rn50like': 'LRN56D',
    'lrresnet56dwrk_resnet101_lr_rn50like': 'LRN56DWRK',
    'lrresnet50t2d_resnet101_lr_rn50like': 'LRN50T2D',
    'lrresnet50t2pdwosrnk_resnet101_lr_rn50like': 'LRN50T2PDWOSRNK',
    'lrresnet50t2pdwsrnk_resnet101_lr_rn50like': 'LRN50T2PDWSRNK',
    'lrresnet50t4pdwsrnk_resnet101_lr_rn50like': 'LRN50T4PDWSRNK',

    'resnet101_resnet101_lr_rn101like': 'RN101',
    'resnet101d_resnet101_lr_rn101like': 'RN101D',
    'lrresnet107_resnet101_lr_rn101like': 'LRN107',
    'lrresnet107d_resnet101_lr_rn101like': 'LRN107D',
    'lrresnet107dwrk_resnet101_lr_rn101like': 'LRN107DWRK',
    'lrresnet101t2d_resnet101_lr_rn101like': 'LRN101T2D',
    'lrresnet101t2pdwosrnk_resnet101_lr_rn101like': 'LRN101T2PDWOSRNK',
    'lrresnet101t2pdwsrnk_resnet101_lr_rn101like': 'LRN101T2PDWSRNK',
    'lrresnet101t4pdwsrnk_resnet101_lr_rn101like': 'LRN101T4PDWSRNK',


    # high-resolution from scratch
    # high-resolution from scratch (cross entropy loss)
    'resnet18_hr_ce_200': 'CE (RN18, 200)',
    'resnet34_hr_ce_200': 'CE (RN34, 200)',
    'resnet50_hr_ce_200': 'CE (RN50, 200)',

    # high-resolution from scratch with SOD (cal teacher)
    'resnet18_resnet101_hr_sod_200': 'SOD (RN18, 200)',
    'resnet34_resnet101_hr_sod_200': 'SOD (RN34, 200)',
    'resnet50_resnet101_hr_sod_200': 'SOD (RN50, 200)',

    # high-resolution from scratch with tgda
    'resnet18_resnet101_hr_tgda_200': 'TGDA (RN18, 200)',
    'resnet34_resnet101_hr_tgda_200': 'TGDA (RN34, 200)',
    'resnet50_resnet101_hr_tgda_200': 'TGDA (RN50, 200)',

    # high-resolution from scratch with tgda (800 epochs)
    'resnet18_resnet101_hr_tgda_800': 'TGDA (RN18)',
    'resnet34_resnet101_hr_tgda_800': 'TGDA (RN34)',
    'resnet50_resnet101_hr_tgda_800': 'TGDA (RN50)',


    # pretrained results
    # low-resolution + tgda + pretrained
    'resnet18_resnet101_lr_tgda_pt': 'TGDA (RN18, IS=128, PT)',
    'resnet18d_resnet101_lr_tgda_pt': 'TGDA (RN18D, IS=128, PT)',

    'resnet34_resnet101_lr_tgda_pt': 'TGDA (RN34, IS=128, PT)',
    'resnet34d_resnet101_lr_tgda_pt': 'TGDA (RN34D, IS=128, PT)',

    'resnet50_resnet101_lr_tgda_pt': 'TGDA (RN50, IS=128, PT)',
    'resnet50d_resnet101_lr_tgda_pt': 'TGDA (RN50D, IS=128, PT)',

    'resnet101_resnet101_lr_tgda_pt': 'TGDA (RN101, IS=128, PT)',
    'resnet101d_resnet101_lr_tgda_pt': 'TGDA (RN101D, IS=128, PT)',

    # high-resolution + SOD + pretrained
    'resnet18_resnet101_hr_sod_pt_200': 'SOD (RN18, PT)',
    'resnet34_resnet101_hr_sod_pt_200': 'SOD (RN18, PT)',
    'resnet50_resnet101_hr_sod_pt_200': 'SOD (RN18, PT)',

    # high-resolution + TGDA + pretrained
    'resnet18_resnet101_hr_tgda_pt_200': 'TGDA (RN18, PT)',
    'resnet34_resnet101_hr_tgda_pt_200': 'TGDA (RN34, PT)',
    'resnet50_resnet101_hr_tgda_pt_200': 'TGDA (RN50, PT)',


    # others
    # IS=128 SotA (RN101?)
    'PCA-Net (IS=128 SotA)': 'PCA-Net',
    'PMG (IS=128 SotA)': 'PMG',
    'API-Net (IS=128 SotA)': 'API-Net',
    'MGN-CNN (IS=128 SotA)': 'MGN-CNN',
    'DSSD (IS=128 SotA)': 'DSSD (IS=128 SotA)',


    # IS=448 SotA with Res-50
    'NTS (Res-50) (IS=448 SotA)': 'NTS (RN50) (IS=448 SotA)',
    'API-Net (Res-50) (IS=448 SotA)': 'API-Net (RN50) (IS=448 SotA)',
    'Cross-X (Res-50) (IS=448 SotA)': 'Cross-X (RN50) (IS=448 SotA)',
    'DCL (Res-50) (IS=448 SotA)': 'DCL (RN50) (IS=448 SotA)',
    'DTB-Net (Res-50) (IS=448 SotA)': 'DTB-Net (RN50) (IS=448 SotA)',
    'CIN (Res-50) (IS=448 SotA)': 'CIN (RN50) (IS=448 SotA)',
    'LIO (Res-50) (IS=448 SotA)': 'LIO (RN50) (IS=448 SotA)',
    'SnapMix (Res-50) (IS=448 SotA)': 'SnapMix (RN50) (IS=448 SotA)',
    'SnapMix + CEKD (Res-50) (IS=448 SotA)': 'SnapMix + CEKD (RN50) (IS=448 SotA)',


    # IS=448 SotA with Res-101
    'DTB-Net (Res-101) (IS=448 SotA)': 'DTB-Net (RN101) (IS=448 SotA)',
    'CIN (Res-101) (IS=448 SotA)': 'CIN (RN101) (IS=448 SotA)',
    'API-Net (Res-101) (IS=448 SotA)': 'API-Net (RN101) (IS=448 SotA)',
    'SnapMix (Res-101) (IS=448 SotA)': 'SnapMix (RN101) (IS=448 SotA)',
    'SnapMix + CEKD (Res-101) (IS=448 SotA)': 'SnapMix + CEKD (RN101) (IS=448 SotA)',
}


VAR_DIC = {
    'setting': 'Setting',
    'acc': 'Accuracy (%)',
    'acc_std': 'Accuracy Std. Dev. (%)',
    'epoch': 'Epoch',
    'dataset_name': 'Dataset',
    'method': 'Method',
    'family': 'Method Family',
    'flops': 'Inference FLOPs (10^9)',
    'time_train': 'Train Time (hours)',
    'vram_train': 'Train VRAM (GB)',
    'tp_train': 'Train Throughput (Images/s)',
    'trainable_percent': 'Trainable Parameters (%)',
    'no_params': 'Number of Parameters (10^6)',
    'no_params_trainable': 'Trainable Parameters (10^6)',
    'no_params_total': 'Total Parameters (10^6)',
    'no_params_trainable_total': 'Total Trainable Params. (10^6)',
    'flops_inference': 'Inference FLOPs (10^9)',
    'tp_stream': 'Stream Throughput (Images/s)',
    'vram_stream': 'Stream VRAM (GB)',
    'latency_stream': 'Stream Latency (s)',
    'tp_batched': 'Batched Throughput  (Images/s)',
    'vram_batched': 'Batched VRAM (GB)',
}


INCONSISTENT_MODELS = {
    'tv_resnet101': 'resnet101.tv_in1k',
}


def rename_var(x):
    if x in SETTINGS_DIC.keys():
        return SETTINGS_DIC[x]
    elif x in METHODS_DIC.keys():
        return METHODS_DIC[x]
    elif x in DATASETS_DIC.keys():
        return DATASETS_DIC[x]
    elif x in VAR_DIC.keys():
        return VAR_DIC[x]
    return x


def rename_vars(df, var_rename=False, args=None):
    if 'setting' in df.columns:
        df['setting'] = df['setting'].apply(rename_var)
    if 'method' in df.columns:
        df['method'] = df['method'].apply(rename_var)
    if 'dataset_name' in df.columns:
        df['dataset_name'] = df['dataset_name'].apply(rename_var)
    if 'family' in df.columns:
        df['family'] = df['family'].apply(rename_var)

    if var_rename:
        df.rename(columns=VAR_DIC, inplace=True)
        for k, v in VAR_DIC.items():
            if k == args.x_var_name:
                args.x_var_name = v
            elif k == args.y_var_name:
                args.y_var_name = v
            elif k == args.hue_var_name:
                args.hue_var_name = v
            elif k == args.style_var_name:
                args.style_var_name = v
            elif k == args.size_var_name:
                args.size_var_name = v

    return df


def add_setting(df):
    conditions = [
        (df['project_name'] == 'prev'),

        (df['project_name'] == 'prev_is128_rn18'),
        (df['project_name'] == 'prev_is128_rn34'),
        (df['project_name'] == 'prev_is128_rn50'),
        (df['project_name'] == 'prev_is128_rn101'),

        (df['project_name'] == 'prev_is448_rn18'),
        (df['project_name'] == 'prev_is448_rn34'),
        (df['project_name'] == 'prev_is448_rn50'),
        (df['project_name'] == 'prev_is448_rn101'),

        (df['project_name'] == 'prev_is128_sota'),

        (df['project_name'] == 'prev_is448_sota_rn50'),
        (df['project_name'] == 'prev_is448_sota_rn101'),


        ((df['project_name'].isin(['Backbones', 'CALBackbones', 'ParamEfficientBackbones', 'PEFGIR'])) & (df['selector'] == '') & (df['serial'] == 1)),
        ((df['project_name'].isin(['Backbones', 'CALBackbones', 'ParamEfficientBackbones', 'PEFGIR'])) & (df['selector'] == 'cal') & (df['serial'] == 1)),
        ((df['project_name'].isin(['Backbones', 'CALBackbones', 'ParamEfficientBackbones', 'PEFGIR'])) & (df['serial'] == 15)),

        ((df['model_name_teacher'] == '') & (df['serial'] == 0)),
        ((df['model_name_teacher'] == '') & (df['serial'] == 1)),
        ((df['model_name_teacher'] == '') & (df['serial'] == 2)),
        ((df['model_name_teacher'] == '') & (df['serial'] == 3)),
        ((df['model_name_teacher'] == '') & (df['serial'] == 4)),
        ((df['model_name_teacher'] == '') & (df['serial'] == 5)),

        ((df['tgda'] == False) & (df['selector'] == '') & (df['serial'] == 0)),
        ((df['tgda'] == False) & (df['selector'] == '') & (df['serial'] == 1)),
        ((df['tgda'] == False) & (df['selector'] == '') & (df['serial'] == 2)),
        ((df['tgda'] == False) & (df['selector'] == '') & (df['serial'] == 3)),
        ((df['tgda'] == False) & (df['selector'] == '') & (df['serial'] == 4)),
        ((df['tgda'] == False) & (df['selector'] == '') & (df['serial'] == 5)),

        ((df['tgda'] == False) & (df['selector'] == 'cal') & (df['serial'] == 0)),
        ((df['tgda'] == False) & (df['selector'] == 'cal') & (df['serial'] == 1)),
        ((df['tgda'] == False) & (df['selector'] == 'cal') & (df['serial'] == 2)),
        ((df['tgda'] == False) & (df['selector'] == 'cal') & (df['serial'] == 3)),
        ((df['tgda'] == False) & (df['selector'] == 'cal') & (df['serial'] == 4)),
        ((df['tgda'] == False) & (df['selector'] == 'cal') & (df['serial'] == 5)),

        ((df['tgda'] == True) & (df['serial'] == 0)),
        ((df['tgda'] == True) & (df['serial'] == 1)),
        ((df['tgda'] == True) & (df['serial'] == 2)),
        ((df['tgda'] == True) & (df['serial'] == 3)),
        ((df['tgda'] == True) & (df['serial'] == 4)),
        ((df['tgda'] == True) & (df['serial'] == 5)),

        (df['serial'] == 20),
        (df['serial'] == 21),
        (df['serial'] == 22),
        (df['serial'] == 23),

        (df['serial'] == 24),
        (df['serial'] == 25),
        (df['serial'] == 26),
        (df['serial'] == 27),
        (df['serial'] == 28),
        (df['serial'] == 29),

        (df['serial'] == 31),
        (df['serial'] == 32),

        (df['serial'] == 41),
        (df['serial'] == 42),
        (df['serial'] == 43),
        (df['serial'] == 44),
        (df['serial'] == 45),
        (df['serial'] == 46),

        (df['serial'] == 51),
        (df['serial'] == 52),
        (df['serial'] == 53),
        (df['serial'] == 54),

        (df['serial'] == 61),
    ]

    df['setting'] = np.select(conditions, SERIALS_EXPLANATIONS, default='')
    return df


def standardize_df(df):
    # methods
    df = df.fillna({'model_name_teacher': '', 'selector': '',  'tgda': False,})

    # rename timm teachers based on previous naming scheme
    df['model_name'] = df['model_name'].apply(lambda x: INCONSISTENT_MODELS.get(x, x))

    # identifier for model based on training strategy / experiment purpose
    df = add_setting(df)

    # combine all experiment variabels into one: method for easier filtering
    df['model_name_teacher_str'] = df['model_name_teacher'].apply(lambda x: f'_{x}' if x else '')
    df['setting_str'] = df['setting'].apply(lambda x: f'_{x}' if (x and 'prev' not in x) else '')
    # df['setting_str'] = df['setting'].apply(lambda x: f'_{x}' if (x and 'prev' not in x) else '')

    df['method'] = df['model_name'] + df['model_name_teacher_str'] + df['setting_str']

    # val_acc and test_acc in most cases refers to same variable so standarize name
    df.rename(columns={'val_acc': 'acc', 'test_acc': 'acc'}, inplace=True)
    return df


def keep_columns(df, type='acc'):
    if type == 'all':
        return df

    elif type == 'acc':
        keep = ['acc', 'dataset_name', 'serial', 'setting', 'method']
    elif type == 'inference_cost':
        keep = ['host', 'serial', 'setting', 'dataset_name', 'method',
                'batch_size', 'throughput', 'flops', 'max_memory']
    elif type == 'train_cost':
        keep = ['host', 'serial', 'setting', 'method', 'batch_size', 'epochs',
                'dataset_name', 'num_images_train', 'num_images_val',
                'time_total', 'flops', 'max_memory',
                'no_params_trainable', 'no_params']

    df = df[keep]
    return df


def filter_df(df, keep_datasets=None, keep_methods=None,
              keep_serials=None, keep_settings=None,
              filter_datasets=None, filter_methods=None,
              filter_serials=None, filter_settings=None):
    if keep_datasets:
        df = df[df['dataset_name'].isin(keep_datasets)]

    if keep_methods:
        df = df[df['method'].isin(keep_methods)]

    if keep_serials:
        df = df[df['serial'].isin(keep_serials)]

    if keep_settings:
        df = df[df['setting'].isin(keep_settings)]

    if filter_datasets:
        df = df[~df['dataset_name'].isin(filter_datasets)]

    if filter_methods:
        df = df[~df['method'].isin(filter_methods)]

    if filter_serials:
        df = df[~df['serial'].isin(filter_serials)]

    if filter_settings:
        df = df[df['setting'].isin(filter_settings)]

    return df


def assign_ours_identifier(df):
    # assign a variable based on method such as if 'vitfs' or 'lrres'
    # in method then sota = Ours else Previous
    df['sota'] = df['method'].apply(lambda x: 'ours' if (('lrr' in x) or ('vitfs' in x)) else 'prev')
    return df


def preprocess_df(
    df, type='all', keep_datasets=None, keep_methods=None,
    keep_serials=None, keep_settings=None,
    filter_datasets=None, filter_methods=None,
    filter_serials=None, filter_settings=None,
    ):

    # load dataset and preprocess to include method and setting columns, rename val_acc to acc
    df = standardize_df(df)

    df = filter_df(
        df, keep_datasets, keep_methods, keep_serials, keep_settings, 
        filter_datasets, filter_methods, filter_serials, filter_settings,
    )

    # drop columns
    df = keep_columns(df, type=type)
    
    # sort
    df = sort_df(df)

    # to identify ours
    # df = assign_ours_identifier(df)

    return df


def round_combine_str_mean_std(df, col='acc'):
    df[f'{col}'] = df[f'{col}'].round(2)
    df[f'{col}_std'] = df[f'{col}_std'].round(2)

    df[f'{col}_mean_std_latex'] = "$" + df[f'{col}'].astype(str) + "\\pm{" + df[f'{col}_std'].astype(str) + "}$"
    df[f'{col}_mean_std'] = df[f'{col}'].astype(str) + "+-" + df[f'{col}_std'].astype(str)

    return df


def add_all_cols_group(df, col='dataset_name'):
    subset = df.copy(deep=False)
    subset[col] = 'all'
    df = pd.concat([df, subset], axis=0, ignore_index=True)
    return df


def drop_na(df, args):
    subset = [args.x_var_name, args.y_var_name]
    if args.hue_var_name:
        subset.append(args.hue_var_name)
    if args.style_var_name:
        subset.append(args.style_var_name)
    if args.size_var_name:
        subset.append(args.size_var_name)
    df = df.dropna(subset=subset)
    return df


def sort_df(df, method_only=False):
    if method_only:
        df['method_order'] = pd.Categorical(df['method'], categories=METHODS_DIC.keys(), ordered=True)

        df = df.sort_values(by=['serial', 'setting', 'method_order'], ascending=True)
        df = df.drop(columns=['method_order'])
    else:
        df['setting_order'] = pd.Categorical(df['setting'], categories=SETTINGS_DIC.keys(), ordered=True)
        df['dataset_order'] = pd.Categorical(df['dataset_name'], categories=DATASETS_DIC.keys(), ordered=True)
        df['method_order'] = pd.Categorical(df['method'], categories=METHODS_DIC.keys(), ordered=True)

        df = df.sort_values(by=['serial', 'setting_order', 'dataset_order', 'method_order'], ascending=True)
        df = df.drop(columns=['method_order', 'dataset_order', 'setting_order'])
    return df




def group_by_family(x):
    # create families based on the settings and use these as filters

    # classifiers = ('vit_b16_cls_fz', 'vit_b16_lrblp_fz', 'vit_b16_mpncov_fz',
    #                'vit_b16_ifacls_fz', 'pedeit_base_patch16_224.fb_in1k_cls_fz',
    #                'pedeit3_base_patch16_224.fb_in1k_cls_fz')

    # pefgir = ('vit_b16_cls_psm_fz', 'vit_b16_cls_maws_fz', 'vit_b16_cal_fz',
    #           'vit_b16_avg_cls_rollout_fz', 'vit_b16_cls_glsim_fz')

    # petl = ('vit_b16_cls_vqt_fz', 'vit_b16_cls_vpt_shallow_fz', 'vit_b16_cls_vpt_deep_fz',
    #         'vit_b16_cls_convpass_fz', 'vit_b16_cls_adapter_fz',
    #         'pedeit_base_patch16_224.fb_in1k_cls_adapter_fz',
    #         'pedeit3_base_patch16_224.fb_in1k_cls_adapter_fz')

    # ufgir = ('clevit_fz', 'csdnet_fz', 'mixvit_fz', 'vit_b16_sil_fz')

    # ila = ('vit_b16_ila_dso_cls_fz', 'vit_b16_ila_cls_fz', 'vit_b16_ila_dso_cls_adapter_fz',
    #        'vit_b16_ila_dso_cls_convpass_fz', 'vit_b16_ila_cls_adapter_fz',
    #        'vit_b16_ila_dso_cls_adapter_vpt_shallow_fz', 'vit_b16_ila_cls_adapter_vpt_shallow_fz',
    #        'pedeit_base_patch16_224.fb_in1k_ila_dso_cls_adapter_fz',
    #        'pedeit3_base_patch16_224.fb_in1k_ila_dso_cls_adapter_fz')

    # saw = ['vit_b16_ila_dso_cls_adapter_saw_fz',
    #        'pedeit_base_patch16_224.fb_in1k_ila_dso_cls_adapter_saw_fz',
    #        'pedeit3_base_patch16_224.fb_in1k_ila_dso_cls_adapter_saw_fz']

    # if x in ila:
    #     return 'ila'
    # elif x in petl:
    #     return 'petl'
    # elif x in pefgir:
    #     return 'pefgir'
    # elif x in ufgir:
    #     return 'ufgir'
    # elif x in saw:
    #     return 'saw'
    # elif x in classifiers:
    #     return 'peclassifier'
    return x
