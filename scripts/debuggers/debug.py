import os

cmd = 'ffmpeg -y -framerate 25 -i  {} \
        -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
        {}'.format(os.path.join(work_dir, 'debug/compositeImage/composite_image_%05d.jpg'),
                   os.path.join(work_dir, 'debug/composite_image.mp4'))
os.system(cmd)

cmd = 'ffmpeg -y -framerate 25 -i  {} \
        -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
        {}'.format(os.path.join(work_dir, 'costVolumeFilter/cost_volume_jpg/cost_slice_%05d.jpg'),
                   os.path.join(work_dir, 'debug/cost_volume_filtered.mp4'))
os.system(cmd)
