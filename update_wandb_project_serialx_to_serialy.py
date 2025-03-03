import re
import argparse
import wandb


def update_wandb_project_serialx_to_serialy(project, x, y):
    api = wandb.Api()

    runs = api.runs(path=project, filters={'$and': [
        {'config.serial': x},
    ]})

    len(runs)

    for run in runs:
        print(f'Current name: {run.name}')

        new_name = re.sub(f'_{x}$', f'_{y}', run.name)
        run.name = new_name
        run.config['serial'] = y
        run.update()

        print(f'Updated run name to {run.name}')

    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--project_name', type=str, default='nycu_pcs/KD_TGDA',
                        help='project_entity/project_name')
    # filters
    parser.add_argument('--serial_x', type=int, default=33)
    parser.add_argument('--serial_y', type=int, default=53)
    args= parser.parse_args()
    return args

def main():
    args = parse_args()

    update_wandb_project_serialx_to_serialy(args.project_name, args.serial_x, args.serial_y)

    return 0


if __name__ == '__main__':
    main()
