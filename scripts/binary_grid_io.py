import os
import numpy as np
import struct


def read_binary_grid(fpath, dtype):
    with open(fpath, 'rb') as fp:
        version = struct.unpack('<B', fp.read(1))[0]
        endian = struct.unpack('<B', fp.read(1))[0]
        int_size = struct.unpack('<B', fp.read(1))[0]

        elem_size = struct.unpack('<i', fp.read(int_size))[0]
        width = struct.unpack('<i', fp.read(int_size))[0]
        height = struct.unpack('<i', fp.read(int_size))[0]
        depth = struct.unpack('<i', fp.read(int_size))[0]

        return np.fromfile(fp, dtype=dtype, count=width*height*depth).reshape((height, width, depth))


def write_binary_grid(fpath, data, dtype):
    if len(data.shape) == 2:
        data = data[:, :, np.newaxis]

    with open(fpath, 'wb') as fp:
        # version
        fp.write(struct.pack("<c", bytes([1,])))
        # little endian
        fp.write(struct.pack("<c", bytes([0,])))
        # size of int: 4
        fp.write(struct.pack("<c", bytes([4,])))
        # size of dtype (float and int are both 4-byte)
        fp.write(struct.pack("<i", 4))

        # dimension of the grid
        fp.write(struct.pack("<i", data.shape[1]))
        fp.write(struct.pack("<i", data.shape[0]))
        fp.write(struct.pack("<i", data.shape[2]))

        # save data
        data.astype(dtype=dtype).tofile(fp)


if __name__ == '__main__':
    # input_bin = '/data2/kz298/mvs3dm_result/Explorer/colmap/sfm_perspective/plane_sweep_three_view_census/cost_slice_00000.bin'
    # output_bin = '/data2/kz298/mvs3dm_result/Explorer/colmap/sfm_perspective/plane_sweep_three_view_census/cost_slice_00000.bin.new.bin'
    # dtype = np.float32

    input_bin = '/data2/kz298/mvs3dm_result/Explorer/colmap/sfm_perspective/plane_sweep_three_view_census/bestPlanes.bin'
    output_bin = input_bin + '.new.bin'
    dtype = np.int32

    data = read_binary_grid(input_bin, dtype=dtype)
    write_binary_grid(output_bin, data, dtype=dtype)
