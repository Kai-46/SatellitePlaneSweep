# ===============================================================================================================
# Copyright (c) 2019, Cornell University. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify it under the terms of the GNU General 
# Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied 
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program. If not, see 
# <https://www.gnu.org/licenses/>.
#
# Author: Kai Zhang (kz298@cornell.edu)
#
# The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),
# Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.
# The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes.
# ===============================================================================================================


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

