import os
import argparse

import pandas as pd

from utils import preprocess_df, add_all_cols_group, \
    round_combine_str_mean_std, sort_df, rename_var, \
    DATASETS_DIC, METHODS_DIC


def aggregate_results_main(df, fp=None, serials=None,
                           add_method_avg=True, add_dataset_avg=False):
    df = df[df['serial'].isin(serials)]


    if add_method_avg:
        df = add_all_cols_group(df, 'dataset_name')
    if add_dataset_avg:
        df = add_all_cols_group(df, 'method')


    df_std = df.groupby(['serial', 'setting', 'dataset_name', 'method'], as_index=False).agg({'acc': 'std'})
    df = df.groupby(['serial', 'setting', 'dataset_name', 'method'], as_index=False).agg({'acc': 'mean'})
    df['acc_std'] = df_std['acc']


    df = sort_df(df)


    df = round_combine_str_mean_std(df, col='acc')


    if fp:
        df.to_csv(fp, header=True, index=False)
    return df



def compute_diffs(df, diff_vit=False, diff_lrrn=False):
    if diff_vit:
        ours_list = [m for m in df.index if 'vitfs' in m]
        methods = [m for m in df.index if 'vitfs' not in m]
    elif diff_lrrn:
        ours_list = [m for m in df.index if 'lrresnet' in m]
        methods = [m for m in df.index if 'lrresnet' not in m]
        # methods = [m for m in df.index if m.startswith('resnet')]
    else:
        return df

    for ours in ours_list:
        for i, (index, row) in enumerate(df.iterrows()):
            if index != ours and index in methods:
                df.loc[f'diff_{ours}_{str(index)}'] = df.loc[df.index == ours].iloc[0] - row 

    return df


def pivot_table(df, serial=1, fp=None, var='acc',
                save_diff_vit=False, save_diff_lrrn=False, rename=True):
    df = df[df['serial'] == serial].copy(deep=False)


    df = df.pivot(index='method', columns='dataset_name')[var]

    datasets = [ds for ds in DATASETS_DIC.keys() if ds in df.columns]
    df = df[datasets]


    df['method_order'] = pd.Categorical(df.index, categories=METHODS_DIC.keys(), ordered=True)
    df = df.sort_values(by=['method_order'], ascending=True)
    df = df.drop(columns=['method_order'])


    if save_diff_vit or save_diff_lrrn:
        df = compute_diffs(df, save_diff_vit, save_diff_lrrn)


    if rename:
        df.index = df.index.map(rename_var)
        df.columns = df.columns.map(rename_var)


    if fp:
        df.to_csv(fp, header=True, index=True)
    else:
        print(df)

    return 0


def summarize_acc(args):
    # load dataset and preprocess to include method and setting columns, rename val_acc to acc
    # drop columns
    # filter
    df = pd.read_csv(args.input_file)

    df = preprocess_df(
        df,
        'acc',

        getattr(args, 'keep_datasets', None),
        getattr(args, 'keep_methods', None),
        getattr(args, 'keep_serials', None),
        getattr(args, 'keep_settings', None),

        getattr(args, 'filter_datasets', None),
        getattr(args, 'filter_methods', None),
        getattr(args, 'filter_serials', None),
        getattr(args, 'filter_settings', None),
    )


    # aggregate and save main results
    fp = os.path.join(args.results_dir, args.output_file)

    df_main = aggregate_results_main(df, f'{fp}_main.csv', args.main_serials)


    for serial in args.main_serials:

        if serial in [24, 25]:
            save_diff_vit, save_diff_lrrn = True, False
        elif serial in [51, 52, 53, 54]:
            save_diff_vit, save_diff_lrrn = False, True
        else:
            save_diff_vit, save_diff_lrrn = False, False

        df_pivoted = pivot_table(df_main, serial, f'{fp}_{serial}_pivoted.csv',
                                 'acc', save_diff_vit, save_diff_lrrn)

        if args.save_mean_std_tables:
            pivot_table(df_main, serial, f'{fp}_{serial}_pivoted_mean_std.csv',
                                    var='acc_mean_std')
            pivot_table(df_main, serial, f'{fp}_{serial}_pivoted_mean_std_latex.csv',
                                    var='acc_mean_std_latex')


    return df, df_main


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file', type=str, 
                        default=os.path.join('data', 'kd_da_tgda_backbones_prev.csv'),
                        help='filename for input .csv file from wandb')

    parser.add_argument('--main_serials', nargs='+', type=int,
                        default=[0, 1, 2, 3, 4, 5, 6,
                                 20, 21, 22, 23,
                                 24, 25, 26, 27, 28, 29,
                                 31, 32,
                                 51, 52, 53, 54,
                                 61])

    parser.add_argument('--keep_datasets', nargs='+', type=str, default=DATASETS_DIC.keys())
    parser.add_argument('--keep_methods', nargs='+', type=str, default=None)
    parser.add_argument('--keep_serials', nargs='+', type=int, default=None)
    parser.add_argument('--keep_settings', nargs='+', type=str, default=None)

    parser.add_argument('--filter_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--filter_methods', nargs='+', type=str, default=None)
    parser.add_argument('--filter_serials', nargs='+', type=int, default=None)
    parser.add_argument('--filter_settings', nargs='+', type=str, default=None)

    # output
    parser.add_argument('--output_file', type=str, default='acc',
                        help='filename for output .csv file')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'acc'),
                        help='The directory where results will be stored')
    parser.add_argument('--save_mean_std_tables', action='store_true')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    df, df_main = summarize_acc(args)

    return df_main


if __name__ == '__main__':
    main()
