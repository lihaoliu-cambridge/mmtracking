import pickle
import json
import argparse
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(
        description='converting results to json format')
    parser.add_argument('rname')
    parser.add_argument('--generatepl_config', default=None, help='config file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    rname = args.rname
    generatepl_config = args.generatepl_config
    r = pickle.load(open(rname, 'rb'))

    from mmdet.datasets import build_dataloader, build_dataset
    cfg = mmcv.Config.fromfile(generatepl_config)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    print("output path:", rname.split('.')[0])

    # To Zhongying: 
    # you need to modify the following functions in mmtrack/datasets/coco_video_dataset.py
    # I have added a new the two functions from mmdet/datasets/coco.py to mmtrack/datasets/coco_video_dataset.py directly.
    # But you also need to modify that to make it work (get the correct instance_ids).
    dataset.results2json(r, rname.split('.')[0])
