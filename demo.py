import os, cv2
import numpy as np, os.path as op
from pipeline import Pipeline
from utils.yacs import Config
from modules.helpers import guided_upsample


OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def demo(raw_name, out_name):
    NAME = 'fgp'
    EXTENSION = '.png'
    cfg = Config('configs/demo.yaml')
    pipeline = Pipeline(cfg)

    raw_path = 'raw/' + raw_name
    bayer = np.fromfile(raw_path, dtype='uint16', sep='')
    bayer = bayer.reshape((cfg.hardware.raw_height, cfg.hardware.raw_width))

    data, _ = pipeline.execute(bayer)
    output_path = op.join(OUTPUT_DIR, NAME + out_name + EXTENSION)
    output = cv2.cvtColor(data['output'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output)

    output_path = op.join(OUTPUT_DIR, NAME + '_grayscale' + out_name + EXTENSION)
    
    src, ref = data['grayscale'], data['bayer']
    cv2.imwrite(output_path, guided_upsample(src, ref))

if __name__ == '__main__':
    demo('1_1.raw', '_1_1')
    demo('2_1.raw', '_2_1')
