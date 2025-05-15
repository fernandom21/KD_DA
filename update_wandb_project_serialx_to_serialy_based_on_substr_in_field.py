import re
import argparse
import wandb


def update_serialx_to_serialy_based_on_substr_in_field(project, x, y, substr, field):
    api = wandb.Api()

    runs = api.runs(path=project, filters={'$and': [
        {'config.serial': x},
    ]})

    len(runs)

    for run in runs:
        if substr in run.config[field]:
            print(f'Current name: {run.name}', run.config[field])

            new_name = re.sub(f'_{x}$', f'_{y}', run.name)
            run.name = new_name
            run.config['serial'] = y
            run.update()

            print(f'Updated run name to {run.name}')


    print(f'Finished changing from {x} to {y} based on {substr} in cfg {field}')

    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--project_name', type=str, default='nycu_pcs/KD_TGDA',
                        help='project_entity/project_name')
    # filters
    parser.add_argument('--serial_x', type=int, default=23)
    parser.add_argument('--serial_y', type=int, default=311)
    parser.add_argument('--substr', type=str, default='resnet3')
    parser.add_argument('--field', type=str, default='model_name')
    args= parser.parse_args()
    return args


def main():
    args = parse_args()

    update_serialx_to_serialy_based_on_substr_in_field(
       args.project_name, args.serial_x, args.serial_y, args.substr, args.field)

    return 0


if __name__ == '__main__':
    main()
