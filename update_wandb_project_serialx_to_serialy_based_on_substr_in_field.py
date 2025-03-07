import re
import argparse
import wandb

'''
def update_name_x_to_y(project, serial=31, x='ddwsrnk', y='pdwsrnk'):
    api = wandb.Api()

    runs = api.runs(path=project, filters={'$and': [
        {'config.serial': serial},
    ]})

    len(runs)

    for run in runs:
        if x in run.name:
            print(f'Current name: {run.name}')

            new_name = run.name.replace(x, y)
            run.name = new_name
            run.config['model_name'] = run.config['model_name'].replace(x, y)
            # run.update()

            print(f'Updated run name to {run.name}')

    return 0


def update_resnetx_serialx_to_serialy(project, substr, x, y):
    api = wandb.Api()

    runs = api.runs(path=project, filters={'$and': [
        {'config.serial': x},
    ]})

    len(runs)

    for run in runs:
        if substr in run.config['model_name']:
            print(f'Current name: {run.name}')

            # new_name = re.sub('_30$', '_32', run.name)
            new_name = re.sub(f'_{x}$', f'_{y}', run.name)
            run.name = new_name
            run.config['serial'] = y
            run.update()

            print(f'Updated run name to {run.name}')

        elif 'resnet5' in run.config['model_name']:
            print(f'Current name: {run.name}')

            new_name = re.sub('_30$', '_33', run.name)
            run.name = new_name
            run.config['serial'] = 33
            run.update()

            print(f'Updated run name to {run.name}')

        elif 'resnet10' in run.config['model_name']:
            print(f'Current name: {run.name}')

            new_name = re.sub('_30$', '_34', run.name)
            run.name = new_name
            run.config['serial'] = 34
            run.update()

            print(f'Updated run name to {run.name}')

    print('Finished')

    return 0
'''


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
    parser.add_argument('--serial_x', type=int, default=28)
    args= parser.parse_args()
    return args


def main():
    args = parse_args()

    # update_name_x_to_y(args.project_name, args.serial_x, x='ddwsrnk', y='pdwsrnk')

    # update_resnetx_serialx_to_serialy(args.project_name, 'resnet3', args.serial_x, 52)
    # update_resnetx_serialx_to_serialy(args.project_name, 'resnet5', args.serial_x, 53)
    # update_resnetx_serialx_to_serialy(args.project_name, 'resnet10', args.serial_x, 54)

    update_serialx_to_serialy_based_on_substr_in_field(
        args.project_name, args.serial_x, 26, 'aircraft', 'ckpt_path')

    update_serialx_to_serialy_based_on_substr_in_field(
        args.project_name, args.serial_x, 27, 'cars', 'ckpt_path')

    return 0


if __name__ == '__main__':
    main()
