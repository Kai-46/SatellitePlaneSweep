
# first run plane sweep stereo
# then filter the cost volume
# third a graph cut

import json
import os
import numpy as np
from camera_model import CameraModel
import imageio
import shutil
import itertools
import copy


def cut_tiles(config, col_num_tiles, row_num_tiles):
    work_dir = config['work_dir']
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    with open(config['camera_file']) as fp:
        camera_dict = json.load(fp)

    views = [config['ref_view'], ] + config['src_views']
    camera_dict = dict([(k, camera_dict[k]) for k in views])    # sub-camera dict


    # dump the config and camera_file to the directory
    with open(os.path.join(work_dir, 'config.json'), 'w') as fp:
        json.dump(config, fp, indent=2)

    with open(os.path.join(work_dir, 'camera_dict.json'), 'w') as fp:
        json.dump(camera_dict, fp, indent=2)

    # load images and cameras
    ref_img = imageio.imread(os.path.join(config['img_dir'], config['ref_view']))
    ref_cam = CameraModel(camera_dict[config['ref_view']]) 
    src_imgs = [imageio.imread(os.path.join(config['img_dir'], src_view)) for src_view in config['src_views']]
    src_cams = [CameraModel(camera_dict[src_view]) for src_view in config['src_views']]

    # compute tile size
    height, width = ref_img.shape
    tile_w = int(width / col_num_tiles)
    tile_h = int(height / row_num_tiles)
    col_splits = [i * tile_w for i in range(col_num_tiles)]
    col_splits.append(width)
    row_splits = [i * tile_h for i in range(row_num_tiles)]
    row_splits.append(height)

    tile_info = {}
    for i in range(col_num_tiles):
        for j in range(row_num_tiles):
            subdir = os.path.join(work_dir, 'tile_{}_{}'.format(i, j))
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            config_copy = copy.deepcopy(config)
            config_copy['work_dir'] = subdir
            config_copy['img_dir'] = subdir

            col_idx1 = col_splits[i]
            col_idx2 = col_splits[i+1]
            row_idx1 = row_splits[j]
            row_idx2 = row_splits[j+1]

            subdir_base = os.path.basename(subdir)
            tile_info[subdir_base] = {}
            tile_info[subdir_base]['raw_bbx'] = [col_idx1, col_idx2, row_idx1, row_idx2]

            # we try to expand the margin, so that the reconstructed tile-wise label_map is slightly bigger
            margin = 10
            height, width = ref_img.shape
            col_idx1_expand = max([col_idx1 - margin, 0])
            col_idx2_expand = min([col_idx2 + margin, width])
            row_idx1_expand = max([row_idx1 - margin, 0])
            row_idx2_expand = min([row_idx2 + margin, height])

            tile_info[subdir_base]['expand_bbx'] = [col_idx1_expand, col_idx2_expand, row_idx1_expand, row_idx2_expand]
            tile_info[subdir_base]['sub_bbx'] = [col_idx1 - col_idx1_expand, col_idx2 - col_idx1_expand,
                                                 row_idx1 - row_idx1_expand, row_idx2 - row_idx1_expand]

            # crop the reference image and modify camera parameters
            col_idx1 = col_idx1_expand
            col_idx2 = col_idx2_expand
            row_idx1 = row_idx1_expand
            row_idx2 = row_idx2_expand
            ref_tile = ref_img[row_idx1:row_idx2, col_idx1:col_idx2]
            ref_img_name = config_copy['ref_view']
            imageio.imwrite(os.path.join(subdir, ref_img_name), ref_tile)
            fpp = open(os.path.join(subdir, 'crop_info.txt'), 'w')
            fpp.write('{} {} {} {} {}\n'.format(ref_img_name, col_idx1, col_idx2, row_idx1, row_idx2))

            camera_dict_copy = copy.deepcopy(camera_dict)
            params = camera_dict_copy[ref_img_name]
            params[0] = ref_tile.shape[1]
            params[1] = ref_tile.shape[0]
            params[4] -= col_idx1                            # width, height, fx, fy, cx, cy, s
            params[5] -= row_idx1
            camera_dict_copy[ref_img_name] = params

            # crop source images
            # first un-project the eight corner points of reference images
            points = []
            corners = list(itertools.product(*[[col_idx1, col_idx2], [row_idx1, row_idx2], [config['height_min'], config['height_max']]]))
            for corner in corners:
                plane = np.array([0.0, 0.0, 1.0, -corner[2]]).reshape((4, 1))
                point = ref_cam.unproject(corner[0], corner[1], plane)
                points.append(point)

            for k in range(len(src_cams)):
                pixels = []
                src_cam = src_cams[k]
                for point in points:
                    pixel = src_cam.project(point[0, 0], point[1, 0], point[2, 0])
                    pixels.append(list(pixel))

                pixels = np.array(pixels)
                tmp = np.min(pixels, axis=0)
                col_min = tmp[0] - margin   # add some margin
                row_min = tmp[1] - margin
                tmp = np.max(pixels, axis=0)
                col_max = tmp[0] + margin
                row_max = tmp[1] + margin

                height, width = src_imgs[k].shape
                col_min = int(max([col_min, 0]))
                row_min = int(max([row_min, 0]))
                col_max = int(min([col_max, width]))
                row_max = int(min([row_max, height]))

                src_tile = src_imgs[k][row_min:row_max, col_min:col_max]
                src_img_name = config_copy['src_views'][k]
                imageio.imwrite(os.path.join(subdir, src_img_name), src_tile)
                fpp.write('{} {} {} {} {}\n'.format(src_img_name, col_min, col_max, row_min, row_max))

                params = camera_dict_copy[src_img_name]
                params[0] = src_tile.shape[1]
                params[1] = src_tile.shape[0]
                params[4] -= col_min  # width, height, fx, fy, cx, cy, s
                params[5] -= row_min
                camera_dict_copy[src_img_name] = params

            fpp.close()

            with open(os.path.join(subdir, 'camera_dict.json'), 'w') as fpp:
                json.dump(camera_dict_copy, fpp, indent=2)

            config_copy['camera_file'] = os.path.join(subdir, 'camera_dict.json')
            with open(os.path.join(subdir, 'config.json'), 'w') as fpp:
                json.dump(config_copy, fpp, indent=2)

    with open(os.path.join(work_dir, 'tile_info.json'), 'w') as fp:
        json.dump(tile_info, fp, indent=2)

    # write tasks
    with open(os.path.join(work_dir, 'tile_tasks.txt'), 'w') as fp:
        for subdir in tile_info:
            fp.write(os.path.join(work_dir, subdir, 'config.json') + '\n')


if __name__ == '__main__':
    config_file = '/data2/kz298/core3d_result/aoi-d4-jacksonville/pss_configs/0007_0008_0009.json'
    cut_tiles(config_file, 2, 2)
