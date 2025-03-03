import numpy as np
import pandas as pd


SERIALS_EXPLANATIONS = [
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

    # 'kdct',
    'kdct_noaug',
    'kdct_re',
    'kdct_ta',
    'kdct_cm',
    'kdct_mu',
    'kdct_cmmu',

    # 'tgda',
    'tgda_noaug',
    'tgda_re',
    'tgda_ta',
    'tgda_cm',
    'tgda_mu',
    'tgda_cmmu',

    # high-resolution no pt
    'hr_ce_200',
    'hr_kdct_200',
    'hr_tgda_200',
    'hr_tgda_800',

    'hr_vits_tgda_ta_ls_sd_800',
    'sr_vits_tgda_ta_ls_sd_800',
    'sr_vits_tgdahr_ta_sd_800',
    'sr_vits_tgdahr_ta_ls_sd_800',

    # high-resolution with pt
    'hr_kdct_pt_200',
    'hr_tgda_pt_200',

    # cost metrics (time, vram/memory, flops, params)
    'lr_rn18like_cost',
    'lr_rn34like_cost',
    'lr_rn50like_cost',
    'lr_rn101like_cost',

    # low-resolution with no pt
    'lr_rn18like',
    'lr_rn34like',
    'lr_rn50like',
    'lr_rn101like',

    # low-resolution with pt
    'lr_tgda_pt_200',

    # 'No Augmentation',
    # 'Random Erasing', 
    # 'Trivial Aug.', 
    # 'CutMix', 
    # 'Mixup', 
    # 'CutMix+Mixup',
]

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
#     'kdct', 
#     'tgda',
# ]

SETTINGS_DIC = {
    'ce': 'CE',
    'ce_noaug': 'CE (No Aug.)',
    'ce_re': 'CE (Random Erasing)',
    'ce_ta': 'CE (Trivial Aug.)',
    'ce_cm': 'CE (CutMix)',
    'ce_mu': 'CE (Mixup)',
    'ce_cmmu': 'CE (CutMix+Mixup)',

    'kd': 'KD',
    'kd_noaug': 'TGDA (No Aug.)',
    'kd_re': 'KD (Random Erasing)',
    'kd_ta': 'KD (Trivial Aug.)',
    'kd_cm': 'KD (CutMix)',
    'kd_mu': 'KD (Mixup)',
    'kd_cmmu': 'KD (CutMix+Mixup)',

    'kdct': 'KDCT',
    'kdct_noaug': 'TGDA (No Aug.)',
    'kdct_re': 'KDCT (Random Erasing)',
    'kdct_ta': 'KDCT (Trivial Aug.)',
    'kdct_cm': 'KDCT (CutMix)',
    'kdct_mu': 'KDCT (Mixup)',
    'kdct_cmmu': 'KDCT (CutMix+Mixup)',

    'tgda': 'TGDA',
    'tgda_noaug': 'TGDA (No Aug.)',
    'tgda_re': 'TGDA (Random Erasing)',
    'tgda_ta': 'TGDA (Trivial Aug.)',
    'tgda_cm': 'TGDA (CutMix)',
    'tgda_mu': 'TGDA (Mixup)',
    'tgda_cmmu': 'TGDA (CutMix+Mixup)',

    # high-resolution no pt
    'hr_ce_200': 'CE',
    'hr_kdct_200': 'KDCT',
    'hr_tgda_200': 'TGDA',
    'hr_tgda_800': 'TGDA',

    'hr_vits_tgda_ta_ls_sd_800': 'TGDA',
    'sr_vits_tgda_ta_ls_sd_800': 'TGDA',
    'sr_vits_tgdahr_ta_sd_800': 'TGDA',
    'sr_vits_tgdahr_ta_ls_sd_800': 'TGDA',

    # high-resolution with pt
    'hr_kdct_pt_200': 'TGDA',
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
    # KD DA project
    'resnet18_resnet101_tgda': 'RN18-RN101 (TGDA)',
    'vit_t16_resnet101_tgda': 'ViT T/16-RN101 (TGDA)',
    'vit_t16_vit_b16_tgda': 'ViT T/16-ViT B/16 (TGDA)',

    'resnet18_resnet101_kdct': 'RN18-RN101 (KDCT)',
    'vit_t16_resnet101_kdct': 'ViT T/16-RN101 (KDCT)',
    'vit_t16_vit_b16_kdct': 'ViT T/16-ViT B/16 (KDCT)',

    'resnet18_resnet101_kd': 'RN18-RN101 (KD)',
    'vit_t16_resnet101_kd': 'ViT T/16-RN101 (KD)',
    'vit_t16_vit_b16_kd': 'ViT T/16-ViT B/16 (KD)',

    'resnet18_ce': 'RN18 (CE)',
    'vit_t16_ce': 'ViT T/16 (CE)',
}


VAR_DIC = {
    # 'serial': 'Data Augmentation',
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


# def reassign_serial(df):
#     df['serial'] = df['serial'].apply(lambda x: SERIAL_REASSIGN.get(x, x))
#     return df


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


# def determine_ila_method(row):
#     if row['ila'] == True and row['ila_locs'] == '[]':
#         return "_ila_dso"
#     elif row['ila'] == True:
#         return "_ila"
#     else:
#         return ""


def add_setting(df):
    conditions = [

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

        (df['serial'] == 31),
        (df['serial'] == 32),

        (df['serial'] == 41),
        (df['serial'] == 42),
        (df['serial'] == 43),
        (df['serial'] == 44),

        (df['serial'] == 51),
        (df['serial'] == 52),
        (df['serial'] == 53),
        (df['serial'] == 54),

        (df['serial'] == 61),

        # ((df['model_name_teacher'] == '')),
        # ((df['tgda'] == False) & (df['selector'] == '')),
        # ((df['tgda'] == False) & (df['selector'] == 'cal')),
        # ((df['tgda'] == True)),
    ]

        # (df['model_name_teacher'] == ''),
        # (df['tgda'] == False) & (df['selector'] == ''),
        # (df['tgda'] == False) & (df['selector'] == 'cal'),
        # (df['tgda'] == True),


    # df['setting'] = np.select(conditions, SETTING_EXPLANATIONS, default='')

    df['setting'] = np.select(conditions, SERIALS_EXPLANATIONS, default='')
    return df


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


def standardize_df(df):
    # methods
    # df = df.fillna({'classifier': '', 'selector': '', 'adapter': '', 'prompt': '',
    #                 'model_name_teacher': '', 'base_lr': '', 'throughput': ''})
    df = df.fillna({'model_name_teacher': '', 'selector': '',  'tgda': False,})

    df = add_setting(df)
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

    df['model_name_teacher_str'] = df['model_name_teacher'].apply(lambda x: f'_{x}' if x else '')
    df['setting_str'] = df['setting'].apply(lambda x: f'_{x}' if x else '')
    # df['data_aug_str'] = df['data_aug'].apply(lambda x: f'_{x}' if x else '')

    # df['method'] = df['model_name'] + df['model_name_teacher_str'] + df['setting_str'] + df['data_aug_str']
    df['method'] = df['model_name'] + df['model_name_teacher_str'] + df['setting_str']

    df.rename(columns={'val_acc': 'acc'}, inplace=True)
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


def filter_bool_df(df, square_resize_random_crop=None, pretrained=None, cont_loss=None):
    if square_resize_random_crop:
        df = df[df['square_resize_random_crop'] == square_resize_random_crop]
    
    if pretrained:
        df = df[df['pretrained'] == pretrained]

    if cont_loss:
        df = df[df['cont_loss'] == cont_loss]

    # if square_resize_random_crop:
    #     df = df[df["square_resize_random_crop"]==True]
    # if square_resize_random_crop is False:  
    #     df = df[df["square_resize_random_crop"]==False]

    # if pretrained:
    #     df = df[df["pretrained"]==True]
    # if pretrained is False:
    #     df = df[df["pretrained"]==False]

    # if cont_loss:
    #     df = df[df['cont_loss']==True]
    # if cont_loss is False:
    #     df = df[df["cont_loss"]==False]

    return df


def filter_df(df, keep_datasets=None, keep_methods=None, keep_serials=None,
              filter_datasets=None, filter_methods=None, filter_serials=None,
              keep_lr=None,):  # keep_epochs=None,):
    if keep_datasets:
        df = df[df['dataset_name'].isin(keep_datasets)]

    if keep_methods:
        df = df[df['method'].isin(keep_methods)]

    if keep_serials:
        df = df[df['serial'].isin(keep_serials)]

    if filter_datasets:
        df = df[~df['dataset_name'].isin(filter_datasets)]

    if filter_methods:
        df = df[~df['method'].isin(filter_methods)]

    if filter_serials:
        df = df[~df['serial'].isin(filter_serials)]

    # project specific filters
    if keep_lr:
        df = df[df['lr'].isin(keep_lr)]

    # if keep_epochs:
    #     df = df[df['epochs'].isin(keep_epochs)]

    return df


# def preprocess_df(
#     input_file, type='acc', keep_datasets=None, keep_methods=None, keep_serials=None, keep_lr=None, keep_epochs=None,
#     filter_datasets=None, filter_methods=None, filter_serials=None, filter_lr=None, filter_epochs=None,
#     square_resize_random_crop=True, pretrained=False, cont_loss=False):
def preprocess_df(
    df, type='all', keep_datasets=None, keep_methods=None, keep_serials=None,
    filter_datasets=None, filter_methods=None, filter_serials=None,
    keep_lr=None, # keep_epochs=None,
    square_resize_random_crop=None, pretrained=None, cont_loss=None,
    ):

    # load dataset and preprocess to include method and setting columns, rename val_acc to acc
    df = standardize_df(df)

    # filter
    # df = filter_df(df, keep_datasets, keep_methods, keep_serials, keep_lr, keep_epochs,
    #                filter_datasets, filter_methods, filter_serials, filter_lr, filter_epochs)
    df = filter_df(
        df, keep_datasets, keep_methods, keep_serials,
        filter_datasets, filter_methods, filter_serials,
        keep_lr, # keep_epochs,
    )

    # filter bool
    df = filter_bool_df(df, square_resize_random_crop, pretrained, cont_loss)

    # drop columns
    df = keep_columns(df, type=type)
    
    # sort
    df = sort_df(df)

    # saw results (serial 91, 101 -> 1, 3)
    # to be consistent with the naming criteria in the other files / results
    # df = reassign_serial(df)

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
    classifiers = ('vit_b16_cls_fz', 'vit_b16_lrblp_fz', 'vit_b16_mpncov_fz',
                   'vit_b16_ifacls_fz', 'pedeit_base_patch16_224.fb_in1k_cls_fz',
                   'pedeit3_base_patch16_224.fb_in1k_cls_fz')

    pefgir = ('vit_b16_cls_psm_fz', 'vit_b16_cls_maws_fz', 'vit_b16_cal_fz',
              'vit_b16_avg_cls_rollout_fz', 'vit_b16_cls_glsim_fz')

    petl = ('vit_b16_cls_vqt_fz', 'vit_b16_cls_vpt_shallow_fz', 'vit_b16_cls_vpt_deep_fz',
            'vit_b16_cls_convpass_fz', 'vit_b16_cls_adapter_fz',
            'pedeit_base_patch16_224.fb_in1k_cls_adapter_fz',
            'pedeit3_base_patch16_224.fb_in1k_cls_adapter_fz')

    ufgir = ('clevit_fz', 'csdnet_fz', 'mixvit_fz', 'vit_b16_sil_fz')

    ila = ('vit_b16_ila_dso_cls_fz', 'vit_b16_ila_cls_fz', 'vit_b16_ila_dso_cls_adapter_fz',
           'vit_b16_ila_dso_cls_convpass_fz', 'vit_b16_ila_cls_adapter_fz',
           'vit_b16_ila_dso_cls_adapter_vpt_shallow_fz', 'vit_b16_ila_cls_adapter_vpt_shallow_fz',
           'pedeit_base_patch16_224.fb_in1k_ila_dso_cls_adapter_fz',
           'pedeit3_base_patch16_224.fb_in1k_ila_dso_cls_adapter_fz')

    saw = ['vit_b16_ila_dso_cls_adapter_saw_fz',
           'pedeit_base_patch16_224.fb_in1k_ila_dso_cls_adapter_saw_fz',
           'pedeit3_base_patch16_224.fb_in1k_ila_dso_cls_adapter_saw_fz']

    if x in ila:
        return 'ila'
    elif x in petl:
        return 'petl'
    elif x in pefgir:
        return 'pefgir'
    elif x in ufgir:
        return 'ufgir'
    elif x in saw:
        return 'saw'
    elif x in classifiers:
        return 'peclassifier'
    return x
