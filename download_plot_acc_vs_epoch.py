import os
import argparse
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# from utils import standarize_df, rename_vars
from utils import preprocess_df, drop_na, rename_vars


def get_wandb_project_runs(project='nycu_pcs/KD_DA', serials=[0, 1, 2, 3, 4, 5]):
    api = wandb.Api()

    runs = api.runs(path=project, per_page=2000, filters={'$and': [
        {'config.serial': {'$in': serials}},
        # lr != 0.0005, square_resize_random_crop: False, cont_loss: True
        # move to different serials later (different lr are stage 1)
        {'config.lr': 0.0005},
        {'config.square_resize_random_crop': True},
        {'config.cont_loss': False},
    ]})

    # 330: 11 runs per serial (ce, 3 kd, 3 kd+cal, 3 tgda) * 6 serials * 5 datasets
    print('Downloaded runs: ', len(runs))
    return runs


def get_train_progress_df(runs, samples=10000):
    run = runs[0]
    # history_cols = [c for c in run.history().columns if 'val' in c] + ['_step']
    history_cols = ['val_acc', 'epoch']
    cfg_cols = ['serial', 'dataset_name', 'model_name', 'project_name',
                'model_name_teacher', 'selector', 'tgda', 'pretrained']

    df = []

    for i, run in enumerate(runs):
        # history() samples from all steps but can result in inconsistent behavior
        # https://docs.wandb.ai/ref/python/public-api/run/#scan_history
        # https://github.com/wandb/wandb/issues/5994
        # https://community.wandb.ai/t/run-history-returns-different-values-on-almost-each-call/2431
        # history_temp = [row for row in run.scan_history(keys=[history_cols])]
        history_temp = run.history(samples=samples)[history_cols].dropna()

        # config hyperparameter arguments
        cfg = {col: run.config.get(col, None) for col in cfg_cols}

        # system metrics
        # https://docs.wandb.ai/guides/models/app/settings-page/system-metrics/
        # max(runs[0].history(stream='events')['system.proc.memory.rssMB'].dropna())

        # summary metrics wandb.summary() calls
        # summary = {col: run.summary.get(col, None) for col in ['best_acc']}

        history_temp = history_temp.assign(**cfg)
        # history_temp = history_temp.assign(**summary)

        df.append(history_temp)

        if i % 10 == 0:
            print(f'{i} / {len(runs)}')
            # print(history_temp)

    df = pd.concat(df, ignore_index=True)

    print(f'DF length: {len(df)}\n', df.head())
    return df


def modify_df(df, args):

    df = preprocess_df(
        df,
        'all',
        getattr(args, 'keep_datasets', None),
        getattr(args, 'keep_methods', None),
        getattr(args, 'keep_serials', None),
        
        getattr(args, 'filter_datasets', None),
        getattr(args, 'filter_methods', None),
        getattr(args, 'filter_serials', None),
    )

    df = drop_na(df, args)

    df = rename_vars(df, var_rename=True, args=args)

    print(f'DF length: {len(df)}\n', df.head())

    return df


def make_plot(args, df):
    # Seaborn Style Settings
    sns.set_theme(
        context=args.context, style=args.style, palette=args.palette,
        font=args.font_family, font_scale=args.font_scale, rc={
            "grid.linewidth": args.bg_line_width, # mod any of matplotlib rc system
            "figure.figsize": args.fig_size,
        })

    ax = sns.lineplot(x=args.x_var_name, y=args.y_var_name,
                      hue=args.hue_var_name, style=args.style_var_name,
                      markers=False, linewidth=args.line_width, data=df)

    if args.despine_top_right:
    # Remove top, right border
        sns.despine(top=False, right=True, left=False, bottom=False)

    # labels and title
    ax.set(xlabel=args.x_label, ylabel=args.y_label, title=args.title, ylim=args.y_lim)

    if args.log_scale_x:
        ax.set_xscale('log')
    if args.log_scale_y:
        ax.set_yscale('log')

    # ticks labels
    if args.x_ticks_labels:
        x_ticks = ax.get_xticks() if getattr(args, 'x_ticks', None) is None else args.x_ticks
        ax.set_xticks(x_ticks , labels=args.x_ticks_labels)

    # Rotate x-axis or y-axis ticks lables
    if (args.x_rotation != None):
        plt.xticks(rotation = args.x_rotation)
    if (args.y_rotation != None):
        plt.yticks(rotation = args.y_rotation)

    # Change location of legend
    if args.hue_var_name:
        sns.move_legend(ax, loc=args.loc_legend)

    # save plot
    output_file = os.path.join(args.results_dir, f'{args.output_file}.{args.save_format}')
    plt.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
    print('Save plot to directory ', output_file)

    return 0

    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--project_name', type=str, default='nycu_pcs/KD_DA',
                        help='project_entity/project_name')
    # filters
    parser.add_argument('--serials', nargs='+', type=int,
                        default=[0, 1, 2, 3, 4, 5])

    # Subset models and datasets
    parser.add_argument('--keep_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--keep_methods', nargs='+', type=str, default=None)
    parser.add_argument('--keep_serials', nargs='+', type=int, default=None)

    parser.add_argument('--filter_datasets', nargs='+', type=str, default=None)
    parser.add_argument('--filter_methods', nargs='+', type=str, default=None)
    parser.add_argument('--filter_serials', nargs='+', type=int, default=None)

    # output
    parser.add_argument('--output_file', default='acc_vs_epoch', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'train_progress'),
                        help='The directory where results will be stored')
    parser.add_argument('--save_format', choices=['pdf', 'png', 'jpg'], default='png', type=str,
                        help='Print stats on word level if use this command')

    # Make a plot
    parser.add_argument('--x_var_name', type=str, default='epoch',
                        help='name of the variable for x')
    parser.add_argument('--y_var_name', type=str, default='acc',
                        help='name of the variable for y')
    parser.add_argument('--hue_var_name', type=str, default='method',
                        help='legend of this bar plot')
    parser.add_argument('--style_var_name', type=str, default=None,
                        help='legend of this bar plot')
    parser.add_argument('--size_var_name', type=str, default=None,)

    # style related
    parser.add_argument('--context', type=str, default='notebook',
                        help='''affects font sizes and line widths
                        # notebook (def), paper (small), talk (med), poster (large)''')
    parser.add_argument('--style', type=str, default='whitegrid',
                        help='''affects plot bg color, grid and ticks
                        # whitegrid (white bg with grids), 'white', 'darkgrid', 'ticks'
                        ''')
    parser.add_argument('--palette', type=str, nargs='+', default='colorblind',
                        help='''
                        color palette (overwritten by color)
                        # None (def), 'pastel', 'Blues' (blue tones), 'colorblind'
                        # can create a palette that highlights based on a category
                        can create palette based on conditions
                        pal = {"versicolor": "g", "setosa": "b", "virginica":"m"}
                        pal = {species: "r" if species == "versicolor" else "b" for species in df.species.unique()}
                        ''')
    parser.add_argument('--color', type=str, default=None)
    parser.add_argument('--font_family', type=str, default='serif',
                        help='font family (sans-serif or serif)')
    parser.add_argument('--font_scale', type=int, default=1.0, # 0.8 originally
                        help='adjust the scale of the fonts')
    parser.add_argument('--bg_line_width', type=int, default=0.25,
                        help='adjust the scale of the line widths')
    parser.add_argument('--line_width', type=float, default=0.75, # 0.75 originally
                        help='adjust the scale of the line widths')
    parser.add_argument('--fig_size', nargs='+', type=float, default=[6, 4], # [6, 4]
                        help='size of the plot')
    parser.add_argument('--marker', type=str, default='.',
                        help='type of marker for line plot ".", "o", "^", "x", "*"')
    parser.add_argument('--dpi', type=int, default=300)

    parser.add_argument('--orient', type=str, default=None,
                        help='orientation of plot "v", "h"')
    parser.add_argument('--log_scale_x', action='store_true')
    parser.add_argument('--log_scale_y', action='store_true')
    parser.add_argument('--despine_top_right', action='store_true')

    # Set title, labels and ticks
    parser.add_argument('--title', type=str,
                        default='Top-1 Accuracy vs Epoch',
                        # on SoyLocal for DeiT3 EViT with KR=10%
                        help='title of the plot')
    parser.add_argument('--x_label', type=str, default='Epoch',
                        help='x label of the plot')
    parser.add_argument('--y_label', type=str, default='Accuracy (%)',
                        help='y label of the plot')
    parser.add_argument('--y_lim', nargs='*', type=int, default=None,
                        help='limits for y axis (suggest --ylim 0 100)')
    parser.add_argument('--x_ticks', nargs='+', type=int, default=None)
    parser.add_argument('--x_ticks_labels', nargs='+', type=str, default=None,
                        help='labels of x-axis ticks')
    parser.add_argument('--x_rotation', type=int, default=None,
                        help='rotation of x-axis lables')
    parser.add_argument('--y_rotation', type=int, default=None,
                        help='rotation of y-axis lables')

    # Change location of legend
    parser.add_argument('--loc_legend', type=str, default='lower right',
                        help='location of legend options are upper, lower, left right, center')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()
    args.title = args.title.replace('\\n', '\n')
    os.makedirs(args.results_dir, exist_ok=True)

    if args.color:
        # single color for whole palette (sns defaults to 6 colors)
        args.palette = [args.color for _ in range(len(args.subset_models))]

    runs = get_wandb_project_runs(args.project_name, args.serials)

    df = get_train_progress_df(runs)

    df = modify_df(df, args)

    make_plot(args, df)

    return 0

if __name__ == '__main__':
    main()

