import os
import argparse
import pandas as pd


def compute_seen_images(args):
    df_stats = pd.read_csv(args.input_file)

    num_images = []

    for ds_pt in args.dataset_name_pt:
        images_pt = df_stats[df_stats['dataset_name'] == ds_pt]['num_train'].iloc[0]

        for epochs_pt in args.epochs_train_pt:
            total_images_pt = images_pt * epochs_pt
            num_images.append({'ds_epochs': f'{ds_pt}_{epochs_pt}', 'num_train': total_images_pt})


    for ds_ft in args.dataset_name_ft:
        images_ft = df_stats[df_stats['dataset_name'] == ds_ft]['num_train'].iloc[0]

        for epochs_ft in args.epochs_train_ft:
            total_images_ft = images_ft * epochs_ft
            num_images.append({'ds_epochs': f'{ds_ft}_{epochs_ft}', 'num_train': total_images_ft})


    for ds_pt in args.dataset_name_pt:
        images_pt = df_stats[df_stats['dataset_name'] == ds_pt]['num_train'].iloc[0]

        for ds_ft in args.dataset_name_ft:
            images_ft = df_stats[df_stats['dataset_name'] == ds_ft]['num_train'].iloc[0]

            for epochs_pt in args.epochs_train_pt:
                total_images_pt = images_pt * epochs_pt

                for epochs_ft in args.epochs_train_ft:
                    total_images_ft = images_ft * epochs_ft

                    total_images_pt_ft = total_images_pt + total_images_ft

                    ratio_total_ft = total_images_pt_ft / total_images_ft

                    combination_name = f'{ds_pt}_{epochs_pt}_{ds_ft}_{epochs_ft}'
                    num_images.append({'ds_epochs': combination_name,
                                       'num_train': total_images_pt_ft,
                                       'ratio_total_ft': ratio_total_ft})

                    print(combination_name, ratio_total_ft)


    print(num_images)

    df = pd.DataFrame.from_dict(num_images)
    df.to_csv(args.output_file, header=True, index=False)

    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file', type=str,
                        default=os.path.join('data', 'stats_datasets.csv'),
                        help='.csv with datasets info')
    # filters
    parser.add_argument('--dataset_name_pt', nargs='+', type=int,
                        default=['imagenet1k', 'imagenet21k'])
    parser.add_argument('--epochs_train_pt', nargs='+', type=int,
                        # 90 on in21k / in1k for og rn/tv
                        # 100 certain ssl (mocov3)
                        # most new: 300 (deit, a1 timm rn)
                        # deit3: 400
                        # some ssl: 800 (deit, mae), 1000 (deit), 1600 (mae)
                        default=[90, 100, 300, 400, 600, 800, 1000, 1200])

    parser.add_argument('--dataset_name_ft', nargs='+', type=str,
                        default=['aircraft', 'cars', 'cub'])
    parser.add_argument('--epochs_train_ft', nargs='+', type=int,
                        default=[50, 200, 800])

    # output
    parser.add_argument('--output_file', default='cost_images.csv', type=str,
                        help='File path')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'cost'),
                        help='The directory where results will be stored')

    args= parser.parse_args()
    return args

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    args.output_file = os.path.join(args.results_dir, args.output_file)

    compute_seen_images(args)

    return 0

if __name__ == '__main__':
    main()

