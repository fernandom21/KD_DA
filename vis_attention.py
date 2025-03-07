import os
import glob
import argparse

import numpy as np
from PIL import Image
from einops import rearrange
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


DATASETS = ['Cars']

MODEL_NAMES_ALL = [
    'resnet101_ft',

    'resnet101_ft',

    'resnet101_cal_ft',
    'resnet101_cal_ft',
    'resnet101_cal_ft',
    'resnet101_cal_ft',

    'resnet101_ft',
    'resnet101_cal_ft',
]

VIS_TYPE_ALL = [
    'None',

    'CAM',

    'bap_0',
    'bap_1',
    'bap_2',
    'bap_3',

    'GradCAM',
    'GradCAM',
]

VIS_LABELS_ALL = [
    'Samples',

    'CAM',

    'BA-1',
    'BA-2',
    'BA-3',
    'BA-4',

    'GradCAM',
    'GradCAM (CAL)',
]

MODEL_NAMES = [
    'resnet101_ft',

    'resnet101_ft',

    'resnet101_cal_ft',
    'resnet101_cal_ft',
]
VIS_TYPE = ['None', 'CAM', 'bap_3', 'bap_5']
VIS_LABELS = ['Samples', 'CAM', 'BA-1', 'BA-2']


def search_images(images_path, test_images=False):
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = os.path.join(images_path, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print('Total image files pre-filtering', len(files_all))

    filtered = set()
    for file in files_all:
        if (('None' in file)) and test_images and 'test' in file:
            filtered.add(file)
            # filtered.add(file.replace('.png', ''))
        elif (('None' in file)) and not test_images and 'train' in file:
            filtered.add(file)

    filtered = sorted(filtered)
    print('Images after filtering: ', len(filtered), filtered)
    return filtered


def get_vis_paths(file, model_list, vis_list, test_images=False):
    paths = []

    fp, fn = os.path.split(file)
    fp, ds_mn_serial = os.path.split(fp)
    ds = ds_mn_serial.split('_')[0]
    serial = ds_mn_serial.split('_')[-1]

    for model, vis in zip(model_list, vis_list):
        ds_mn = f'{ds}_{model}_{serial}'
        split = 'test' if test_images else 'train'
        full_fn = os.path.join(fp, ds_mn, f'{vis}_{split}.png')
        paths.append(full_fn)

    return paths


def make_img_grid(args):
    images_path = os.path.join(args.file_folder, args.subfolder)
    files = search_images(images_path)

    model_list = MODEL_NAMES_ALL if args.vis_all_masks else MODEL_NAMES
    vis_list = VIS_TYPE_ALL if args.vis_all_masks else VIS_TYPE
    vis_labels_list = VIS_LABELS_ALL if args.vis_all_masks else VIS_LABELS
    datasets_list = DATASETS

    number_imgs = len(files)
    number_vis = len(vis_list)

    imgs_all = []

    args.image_size = args.image_size * args.number_images_per_ds

    for file in files:
        full_names = get_vis_paths(file, model_list, vis_list, args.test_images)
        imgs = [Image.open(fp) for fp in full_names]
        width, height = imgs[0].size
        if width >= height:
            r = width / height
            new_h = args.image_size
            new_w = int(r * args.image_size)
        else:
            r = height / width
            new_w = args.image_size
            new_h = int(r * args.image_size)
        imgs = [img.resize((new_w, new_h)) for img in imgs]

        # PIL images use shape w, h but NP uses h, w
        imgs_np = [np.array(img) for img in imgs]
        imgs_all.extend(imgs_np)

    fig = plt.figure(figsize=(number_imgs * args.number_images_per_ds, number_vis))
    grid = ImageGrid(fig, 111, nrows_ncols=(number_vis, number_imgs),
                     axes_pad=(0.01, 0.01), direction='row', aspect=True)

    for i, (ax, np_arr) in enumerate(zip(grid, imgs_all)):
        # ax.axis('off')
        ax.imshow(np_arr)

        ax.tick_params(top=False,
            bottom=False,
            left=False,
            right=False,
            labelleft=False,
            labelbottom=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        [ax.spines[side].set_visible(False) for side in ('top', 'right', 'bottom', 'left')]

        # first column in each row
        if args.label_y and ((i + 1) % number_imgs == 1 or (number_imgs == 1)):
            label = vis_labels_list[i // number_imgs]
            print(i, label)
            ax.yaxis.set_visible(True)
            ax.set_ylabel(label, fontsize=args.font_size_title)

        # last row
        if args.label_x and ((i + 1) / number_imgs > (number_vis - 1)):
            label = datasets_list[i % number_imgs]
            print(i, label)
            ax.xaxis.set_visible(True)
            ax.set_xlabel(label, fontsize=args.font_size_title)

    # fig.tight_layout()
    fig.savefig(f'{args.file_path}.{args.save_format}', dpi=args.dpi, bbox_inches='tight')
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_folder', type=str,
                        default=os.path.join('data', 'results_inference'),
                        help='name of folder which contains the results')
    parser.add_argument('--subfolder', type=str, default='cars')
    parser.add_argument('--number_images_per_ds', type=int, default=4)
    parser.add_argument('--test_images', action='store_true')
    parser.add_argument('--image_size', default=224, type=int, help='file size')
    parser.add_argument('--save_name', default='attention_cars', type=str, help='File path')
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'vis'),
                        help='The directory where results will be stored')
    parser.add_argument('--save_format', choices=['pdf', 'png', 'jpg'], default='png', type=str,
                        help='Print stats on word level if use this command')
    parser.add_argument('--vis_all_masks', action='store_true',
                        help='visualize all dfsm (layer0/11)')
    parser.add_argument('--label_x', action='store_true')
    parser.add_argument('--label_y', action='store_false')
    parser.add_argument('--font_size_title', type=int, default=12)
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    args.file_path = os.path.join(args.results_dir, args.save_name)

    make_img_grid(args)


if __name__ == '__main__':
    main()
