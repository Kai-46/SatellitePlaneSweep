import os
import numpy as np
import struct


def read_float_grid(fpath):
    with open(fpath, 'rb') as fp:
        version = struct.unpack('<B', fp.read(1))[0]
        endian = struct.unpack('<B', fp.read(1))[0]
        int_size = struct.unpack('<B', fp.read(1))[0]

        elem_size = struct.unpack('<i', fp.read(int_size))[0]
        width = struct.unpack('<i', fp.read(int_size))[0]
        height = struct.unpack('<i', fp.read(int_size))[0]
        depth = struct.unpack('<i', fp.read(int_size))[0]

        return np.fromfile(fp, dtype=np.float32, count=width*height*depth).reshape((height, width, depth))

def read_cost_map(fpath):
    return read_float_grid(fpath)[:, :, 0]

def read_label_map(fpath):
    return read_float_grid(fpath)[:, :, 0]


if __name__ == '__main__':
    pass

