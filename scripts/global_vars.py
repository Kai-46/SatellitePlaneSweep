
PLANE_SWEEP_EXEC_PATH = '/data2/kz298/SimplePlaneSweepStereo/build/bin/simplePinholePlaneSweep'
GRAPH_CUT_EXEC_PATH = '/data2/kz298/OptimizeCostVolume/build/optimizeCostVolume'

SATELLITE_STEREO_PATH = '/data2/kz298/satellite_stereo/'

import sys
from pathlib import Path
# add satellite_stereo to search path
par_dir = str(Path(SATELLITE_STEREO_PATH).parent)
if par_dir not in sys.path:
   sys.path.append(par_dir)
print(sys.path)


# link to satellite_stereo path
# import os
# par_dir = os.dirname(os.path.abspath(os.path.realpath(__file__)))
# link_file = os.path.join(par_dir, 'satellite_stereo')
# if not os.path.exists(link_file):
#     os.symlink(SATELLITE_STEREO_PATH, link_file)
