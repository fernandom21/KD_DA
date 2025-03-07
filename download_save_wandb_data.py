import os
import argparse
import wandb
import pandas as pd


CONFIG_COLS = [
    'project_name', 'serial', 'dataset_name', 'model_name', 'freeze_backbone',

    'model_name_teacher', 'selector', 'tgda', 'pretrained',  'ckpt_path_teacher',
    'square_resize_random_crop',
    'cont_loss', 'cont_temp', 'loss_cont_weight', 'pooling_function',

    'lr', 'base_lr', 'opt', 'weight_decay', 'seed', 'epochs', 'image_size',
    'batch_size', 'num_images_train', 'num_images_val',
]

SUMMARY_COLS = [
    'val_acc', 'val_loss', 'train_acc', 'train_loss',
    'time_total', 'max_memory', 'flops',
    'no_params', 'no_params_trainable', 'throughput'
]

SORT_COLS = [
    'dataset_name', 'serial', 'model_name',
]


DATASETS = ['aircraft', 'cars', 'cub', 'soygene', 'soylocal']
TEACHERS = ['vit_b16', 'resnet101', 'resnet101', 'tv_resnet101']


def get_wandb_project_runs(project, serials=None, teachers_data=False, da_data=False):
    api = wandb.Api()

    if teachers_data:
        runs = api.runs(path=project, per_page=2000, filters={'$and': [
            {'config.serial': {'$in': serials}},
            {'config.test_only': False},
            {'config.freeze_backbone': False},
            {'config.dataset_name': {'$in': DATASETS}},
            {'config.model_name': {'$in': TEACHERS}},
        ]})

    elif da_data:
        runs = api.runs(path=project, per_page=2000, filters={'$and': [
            {'config.serial': {'$in': serials}},
            # lr != 0.0005, square_resize_random_crop: False, cont_loss: True
            # move to different serials later (different lr are stage 1)
            {'config.lr': 0.0005},
            {'config.square_resize_random_crop': True},
            {'config.cont_loss': False},
        ]})

    elif serials:
        # runs = api.runs(path=project, per_page=2000,
        #                filters={'$or': [{'config.serial': s} for s in serials]})
        runs = api.runs(path=project, per_page=2000,
                        # filters={'$and': [{'config.serial': 1}, {'config.dataset_name': 'cub'}]}
                        # filters={'$or': [{'config.serial': s} for s in serials]}
                        # nin (not in) also an option
                        filters={'config.serial': {'$in': serials}}
                        )
    else:
        runs = api.runs(path=project, per_page=2000)

    print('Downloaded runs: ', len(runs))
    return runs


def make_df(runs, config_cols, summary_cols):
    data_list_dics = []

    for i, run in enumerate(runs):
        run_data = {}
        try:
            host = {'host': run.metadata.get('host', None)}
        except:
            # print(run)
            host = {'host': None}
        cfg = {col: run.config.get(col, None) for col in config_cols}
        summary = {col: run.summary.get(col, None) for col in summary_cols}

        run_data.update(host)
        run_data.update(cfg)
        run_data.update(summary)

        data_list_dics.append(run_data)

        if (i + 1) % 100 == 0:
            print(f'{i}/{len(runs)}')

    df = pd.DataFrame.from_dict(data_list_dics)
    print(df.head())

    return df


def sort_save_df(df, fp, sort_cols=['serial']):
    df = df.sort_values(by=sort_cols, ascending=[True for _ in sort_cols])
    df.to_csv(fp, header=True, index=False)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--project_name', type=str, default='nycu_pcs/KD_TGDA',
                        help='project_entity/project_name')
    # filters
    parser.add_argument('--serials', nargs='+', type=int,
                        default=[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 31, 32,
                                 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 61])
    parser.add_argument('--teachers_data', action='store_true')
    parser.add_argument('--da_data', action='store_true')
    parser.add_argument('--config_cols', nargs='+', type=str, default=CONFIG_COLS)
    parser.add_argument('--summary_cols', nargs='+', type=str, default=SUMMARY_COLS)
    # output
    parser.add_argument('--output_file', default='kd_tgda_stage2.csv', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str, default='data',
                        help='The directory where results will be stored')
    parser.add_argument('--sort_cols', nargs='+', type=str, default=SORT_COLS)

    args= parser.parse_args()
    return args

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    args.output_file = os.path.join(args.results_dir, args.output_file)

    runs = get_wandb_project_runs(args.project_name, args.serials,
                                  args.teachers_data, args.da_data)

    df = make_df(runs, args.config_cols, args.summary_cols)

    sort_save_df(df, args.output_file, args.sort_cols)

    return 0

if __name__ == '__main__':
    main()
