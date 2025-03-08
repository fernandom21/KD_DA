import os
import argparse
import pandas as pd


def unpivot_add_project_serial(args):
    df = pd.read_csv(args.input_file)

    df_melted = df.melt(id_vars=['model_name'], var_name='dataset_name', value_name='val_acc')
    df_melted['project_name'] = args.project_name
    df_melted['serial'] = args.serial
    df_melted['model_name'] = df_melted['model_name'].apply(lambda x: f'{x}{args.suffix}')

    df_melted.to_csv(args.output_file, header=True, index=False)

    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file', type=str,
                        default=os.path.join('data', 'tables_previous_mod',
                                             'vits_is224_scratch_vs_ssl.csv'),
                        help='.csv with datasets info')

    parser.add_argument('--project_name', type=str, default='previous')
    parser.add_argument('--serial', type=str, default=25)
    parser.add_argument('--suffix', type=str, default='')

    # output
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'melted_tables'),
                        help='The directory where results will be stored')

    args= parser.parse_args()
    return args

def main():
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    fn = os.path.split(args.input_file)[-1]
    args.output_file = os.path.join(args.results_dir, fn)

    unpivot_add_project_serial(args)

    return 0

if __name__ == '__main__':
    main()

