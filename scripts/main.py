from cut_tiles import cut_tiles
from process_tiles import process_tiles
import json
import sys
from estimate_height_range import estimate_height_range
import os


def main(config_file):
    with open(config_file) as fp:
        config = json.load(fp)

    # insert fields 'img_dir', 'camera_file', 'height_min', 'height_max', 'num_planes'
    config['img_dir'] = os.path.join(config['satellite_stereo_work_dir'], 'images')
    config['camera_file'] = os.path.join(config['satellite_stereo_work_dir'], 'colmap/sfm_perspective/init_ba_camera_dict.json')
    height_min, height_max = estimate_height_range(os.path.join(config['satellite_stereo_work_dir'], 'colmap/sfm_perspective/tri_ba'))
    config['height_min'] = height_min
    config['height_max'] = height_max
    config['num_planes'] = int((height_max - height_min) / config['plane_gap']) + 1

    print('height_min (meters): {}, height_max (meters): {}, num_planes: {}'.format(height_min, height_max, config['num_planes']))

    cut_tiles(config, config['col_num_tiles'], config['row_num_tiles'])
    process_tiles(config['work_dir'], config['max_processes'])


if __name__ == '__main__':
    main(sys.argv[1])
