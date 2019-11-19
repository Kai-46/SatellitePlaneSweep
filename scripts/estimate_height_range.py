import global_vars
from satellite_stereo.colmap.read_model import read_model
import numpy as np


def estimate_height_range(sparse_dir, camera_model='perspective'):
    assert (camera_model == 'perspective' or camera_model == 'pinhole')

    _, _, colmap_points3D = read_model(sparse_dir, ext='.txt')

    z_values = []
    for point3D_id in colmap_points3D:
        point3D = colmap_points3D[point3D_id]
        x = point3D.xyz.reshape((3, 1))
        z_values.append(x[2, 0])

    z_values = np.array(z_values)
    min_val, max_val = np.percentile(z_values, (1, 99))
    # protective margin 20 meters
    margin = 20.0
    min_val -= margin
    max_val += margin

    # print('estimated height_min, height_max (meters): {}, {}'.format(min_val, max_val))
    return min_val, max_val


if __name__ == '__main__':
    sparse_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville/colmap/sfm_perspective/tri_ba'
    estimate_depth_range(sparse_dir)
