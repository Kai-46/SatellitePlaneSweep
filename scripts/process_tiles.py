
# first run plane sweep stereo
# then filter the cost volume
# third a graph cut
import global_vars
import json
import os
import numpy as np
from satellite_stereo.lib.run_cmd import run_cmd
from satellite_stereo.lib.logger import GlobalLogger
from camera_model import CameraModel
import imageio
from binary_grid_io import read_binary_grid, write_binary_grid
from convert_label_map import convert_label_map
import time
import shutil
from multiprocessing import Pool
import imageio
import glob


log = GlobalLogger()
log.turn_on_terminal()


def worker(config_file):
    print('working on config_file: {}...'.format(config_file))

    with open(config_file) as fp:
        config = json.load(fp)

    work_dir = config['work_dir']
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    with open(config['camera_file']) as fp:
        camera_dict = json.load(fp)

    # create img_list
    with open(os.path.join(work_dir, 'img_list.txt'), 'w') as fp:
        img_name = config['ref_view']
        ref_width, ref_height = camera_dict[img_name][:2]
        params_str = ''
        for param in camera_dict[img_name][2:]: # remove width, height
            params_str += ' {}'.format(param)
        idx = 0
        fp.write('{} {}{}\n'.format(idx, img_name, params_str))

        for img_name in config['src_views']:
            params_str = ''
            for param in camera_dict[img_name][2:]: # remove width, height
                params_str += ' {}'.format(param)
            idx += 1
            fp.write('{} {}{}\n'.format(idx, img_name, params_str))
        fp.close()

    # start running the plane sweep stereo and cost volume filtering
    start_time = time.time()

    save_cost_volume = 0
    if config['run_mrf']:
        save_cost_volume = 1

    cmd = '{} --imageList {} --imageFolder {} --refImgId {} --outputFolder {} --matchingCost census ' \
          ' --windowRadius {} --nX 0.0 --nY 0.0 --nZ 1.0 --firstD {} --lastD {} --numPlanes {} ' \
          ' --filterCostVolume 1 --guidedFilterRadius {} --guidedFilterEps {} ' \
          ' --saveBest 1 --filter 1 --filterThres {} --savePointCloud 1 --saveXYZMap 1 ' \
          ' --debug 0 --saveCostVolume {} '.format(
        global_vars.PLANE_SWEEP_EXEC_PATH,
        os.path.join(work_dir, 'img_list.txt'),
        os.path.join(config['img_dir']),
        0,
        work_dir,
        config['window_radius'],
        config['height_min'],
        config['height_max'],
        config['num_planes'],
        config['guided_filter_radius'],
        config['guided_filter_eps'],
        config['filter_cost_above'],
        save_cost_volume
    )
    print('cmd: {}'.format(cmd))
    os.system(cmd)
    
    if os.path.exists(os.path.join(work_dir, 'best_label.bin')):
        os.unlink(os.path.join(work_dir, 'best_label.bin'))
    os.symlink(os.path.relpath(os.path.join(work_dir, 'best/best_plane_index.bin'), work_dir), 
               os.path.join(work_dir, 'best_label.bin'))

    till_cvf_time = time.time()
    print('pss+cvf time: {} minutes'.format((till_cvf_time - start_time) / 60.0))

    if config['run_mrf']:
        # graph cut
        cost_volumes = sorted(glob.glob(os.path.join(work_dir, 'costVolume/cost_slice_*.bin')))
        cost_volumes = [os.path.abspath(x) for x in cost_volumes]
        with open(os.path.join(work_dir, 'cost_volume_list.txt'), 'w') as fp:
            fp.write('\n'.join(cost_volumes) + '\n')

        cmd = '{} --width {} --height {} --costVolumeList {} --initPlaneIndexMap {} ' \
              ' --outputFile {} --threshold {} --penalty {}'.format(
            global_vars.GRAPH_CUT_EXEC_PATH,
            ref_width, ref_height,
            os.path.join(work_dir, 'cost_volume_list.txt'),
            os.path.join(work_dir, 'best_label.bin'),
            os.path.join(work_dir, 'best_label_after_mrf.bin'),
            config['threshold'],
            config['penalty']
        )
        print('cmd: {}'.format(cmd))
        os.system(cmd)

        # create a preview for best_label
        best_label = read_binary_grid(os.path.join(work_dir, 'best_label_after_mrf.bin'), np.float32)
        best_label = best_label[:, :, 0]
        imageio.imwrite(os.path.join(work_dir, 'best_label_after_mrf.jpg'), np.uint8(np.float32(best_label) / np.max(best_label) * 255.0))

        till_mrf_time = time.time()
        pss_cvf_time = (till_cvf_time - start_time) / 60.0
        pss_cvf_mrf_time = (till_mrf_time - start_time) / 60.0
        print('pss+cvf takes {} minutes; mrf takes an additional {} minutes'.format(pss_cvf_time, pss_cvf_mrf_time-pss_cvf_time))

    # remove intermediate cost slices to save disk space
    if os.path.exists(os.path.join(work_dir, 'costVolume')):
        shutil.rmtree(os.path.join(work_dir, 'costVolume'))


# now merge tile-wise label map
def merge_tiles(work_dir):
    print('merging tiles: {}...'.format(work_dir))

    with open(os.path.join(work_dir, 'config.json')) as fp:
        config = json.load(fp)

    with open(os.path.join(work_dir, 'tile_info.json')) as fp:
        tile_info = json.load(fp)

    tile_dirs = list(tile_info.keys())
    sweep_plane_file = os.path.join(work_dir, tile_dirs[0], 'sweepPlanes.txt')

    ref_img = imageio.imread(os.path.join(config['img_dir'], config['ref_view']))
    best_label = np.zeros(ref_img.shape[:2], dtype=np.int32)
    best_label_after_mrf = None
    if config['run_mrf']:
        best_label_after_mrf = best_label.copy()

    for tile_dir in tile_dirs:
        raw_col_idx1, raw_col_idx2, raw_row_idx1, raw_row_idx2 = tile_info[tile_dir]['raw_bbx']
        col_idx1, col_idx2, row_idx1, row_idx2 = tile_info[tile_dir]['sub_bbx']

        label = read_binary_grid(os.path.join(work_dir, tile_dir, 'best_label.bin'), np.float32)
        label = label[:, :, 0]
        best_label[raw_row_idx1:raw_row_idx2, raw_col_idx1:raw_col_idx2] = label[row_idx1:row_idx2, col_idx1:col_idx2]

        if best_label_after_mrf is not None:
            label = read_binary_grid(os.path.join(work_dir, tile_dir, 'best_label_after_mrf.bin'), np.float32)
            label = label[:, :, 0]
            best_label_after_mrf[raw_row_idx1:raw_row_idx2, raw_col_idx1:raw_col_idx2] = label[row_idx1:row_idx2, col_idx1:col_idx2]

    # convert best_label_map to point cloud
    with open(os.path.join(work_dir, 'camera_dict.json')) as fp:
        camera_dict = json.load(fp)
    params = camera_dict[config['ref_view']]
    camera = CameraModel(params)

    # expand gray scale images to 3 channels
    ref_img = np.tile(ref_img[:, :, np.newaxis], (1, 1, 3))

    planes = []
    with open(sweep_plane_file) as fp:
        for line in fp.readlines():
            tmp = line.strip()
            if tmp:
                tmp = tmp.split(',')
                tmp = [float(x) for x in tmp]
                planes.append(np.array(tmp).reshape((4, 1)))

    convert_label_map(config, camera, ref_img, best_label, planes, 
                      os.path.join(work_dir, 'pss_cvf.ply'), 
                      os.path.join(work_dir, 'pss_cvf.tif'),
                      os.path.join(work_dir, 'pss_cvf.jpg'))

    if best_label_after_mrf is not None:
        convert_label_map(config, camera, ref_img, best_label_after_mrf, planes,
                          os.path.join(work_dir, 'pss_cvf_mrf.ply'),
                          os.path.join(work_dir, 'pss_cvf_mrf.tif'),
                          os.path.join(work_dir, 'pss_cvf_mrf.jpg'))


def process_tiles(work_dir, max_processes):
    config_list = []
    with open(os.path.join(work_dir, 'tile_tasks.txt')) as fp:
        for line in fp.readlines():
            line = line.strip()
            if line:
                config_list.append(line)

    pool = Pool(processes=max_processes)
    results = []
    for config_file in config_list:
        res = pool.apply_async(worker, (config_file,))
        results.append(res)

    [r.get() for r in results]  # sync

    merge_tiles(work_dir)


if __name__ == '__main__':
    work_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville/plane_sweep_batch'
    process_tiles(work_dir)
    merge_tiles(work_dir)
