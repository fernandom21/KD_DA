import os
import argparse
import pandas as pd


def stack_two_df(df_1, df_2, fp):
    print('Original lengths: ', len(df_1), len(df_2))
    df = pd.concat([df_1, df_2], ignore_index=True)
    df.to_csv(fp, header=True, index=False)
    print('Stacked length: ', len(df))
    return df


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_df_1', type=str,
                        default=os.path.join('data', 'kd_da_stage2.csv'),
                        help='project_entity/project_name')
    parser.add_argument('--input_df_2', type=str,
                        default=os.path.join('data', 'kd_tgda_stage2.csv'),
                        help='project_entity/project_name')

    # output
    parser.add_argument('--output_file', type=str,
                        default='kd_da_tgda_stage2.csv',
                        help='File path')
    parser.add_argument('--results_dir', type=str, default='data',
                        help='The directory where results will be stored')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    args.output_file = os.path.join(args.results_dir, args.output_file)

    df_1 = pd.read_csv(args.input_df_1)
    df_2 = pd.read_csv(args.input_df_2)

    df = stack_two_df(df_1, df_2, args.output_file)

    return 0


if __name__ == '__main__':
    main()

