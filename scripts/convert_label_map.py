import global_vars
import numpy as np
import os
from satellite_stereo.lib.ply_np_converter import np2ply
from satellite_stereo.coordinate_system import local_to_global
from satellite_stereo.produce_dsm import produce_dsm_from_points
from satellite_stereo.lib.latlon_utm_converter import latlon_to_eastnorh
from multiprocessing import Pool
import json


def convert_label_map_worker(camera_model, ref_img, pixel_plane_idx_array, planes):
    cnt = pixel_plane_idx_array.shape[0]
    return_vals = np.zeros((cnt, 6))

    K_inv = camera_model.K_inv
    R = camera_model.R
    t = camera_model.t

    # function calls introduce overhead in python
    for i in range(cnt):
        x, y, idx = pixel_plane_idx_array[i, :]

        # xyz = camera_model.unproject(c, r, planes[idx])

        # plane = planes[idx]
        
        # interpolate plane equation
        idx_below = int(np.floor(idx))
        idx_up = int(np.ceil(idx))
        if idx_below < 0 or idx_up >= len(planes):
            continue
        plane = planes[idx_below]
        if idx_below != idx_up:
            plane[3] = (idx_up - idx) * (planes[idx_below][3]) + (idx - idx_below) * (planes[idx_up][3])

        point = np.array([x, y, 1.0]).reshape((3, 1))
        point = np.dot(K_inv, point)

        plane_normal = np.dot(R, plane[0:3, 0])
        plane_constant = np.dot(plane_normal.T, t) + plane[3, 0]

        depth = plane_constant / np.dot(plane_normal.T, point)
        point *= depth
        point = np.dot(R.T, point - t)

        color = ref_img[y, x, :]
        return_vals[i, :] = np.concatenate((point.flatten(), color.flatten()))

    return return_vals

# convert label map to a point cloud
# each pixel of label map is an integer
# planes is a list of 4 by 1 numpy arrays
# ref_img is width * height * channel
def convert_label_map(config, camera_model, ref_img, plane_index_map, planes, out_ply, out_tif, out_jpg):
    height, width = plane_index_map.shape
    row, col = np.meshgrid(range(height), range(width), indexing='ij')
    col = col.reshape((-1, 1))
    row = row.reshape((-1, 1))
    plane_indices = plane_index_map.reshape((-1, 1)) 
    # assume plane_index 0 is also invalid
    mask = plane_indices > 0
    col = col[mask].reshape((-1, 1))
    row = row[mask].reshape((-1, 1))
    plane_indices = plane_indices[mask].reshape((-1, 1))
    pixel_plane_idx_array = np.hstack((col, row, plane_indices))

    # divide the big array into smaller chunks for multi-processing
    max_processes = 8
    pixel_plane_idx_array = np.array_split(pixel_plane_idx_array, max_processes, axis=0)

    # launch processes
    pool = Pool(processes=max_processes)
    results = []
    for i in range(max_processes):
        res = pool.apply_async(convert_label_map_worker, 
                               (camera_model, ref_img, pixel_plane_idx_array[i], planes))
        results.append(res)

    points = [r.get() for r in results] # sync
    points = np.vstack(points)
    
    color = np.uint8(points[:, 3:6])
    lat, lon, alt = local_to_global(config['satellite_stereo_work_dir'], points[:, 0:1], points[:, 1:2], points[:, 2:3])
    east, north = latlon_to_eastnorh(lat, lon)
    points = np.hstack((east, north, alt))
    with open(os.path.join(config['satellite_stereo_work_dir'], 'aoi.json')) as fp:
        aoi_dict = json.load(fp)
    comment_1 = 'projection: UTM {}{}'.format(aoi_dict['zone_number'], aoi_dict['hemisphere'])
    comments = [comment_1,]
    np2ply(points, out_ply, color=color, comments=comments, use_double=True)

    # write dsm to tif
    produce_dsm_from_points(config['satellite_stereo_work_dir'], points, out_tif, out_jpg)


if __name__ == '__main__':
    pass
