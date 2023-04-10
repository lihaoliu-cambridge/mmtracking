import numpy as np
import json
import pickle
import argparse
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(
            description='filtering results')
    parser.add_argument('rname')
    parser.add_argument('--first_frame_labeled_json', default=None, help='first frame labeled json')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    rname = args.rname

    thresh = 0.5
    print("thresh hold for each category:", thresh)

    predict_annos = mmcv.load(rname.split('.')[0] + '.bbox.json')

    filtered_annos = []
    for i in range(len(predict_annos)):
        if predict_annos[i]['score'] > thresh:
            filtered_annos.append(predict_annos[i])

    print("generated instance", len(predict_annos))
    print("total saved instance", len(filtered_annos))

    # To Zhongying: 
    # There is one more function you need to implement here, sorry I didn't have time to do it.
    # Before you save it to a json file, you also need to load the ground truth from the first frame labeled json file.
    # Not just simply merge the "prediction for unlabeled json file" and "ground truth from the first frame labeled json file",
    # You need to merge the instance_id too, I suggest you use simple overlap check to see if they re the same instance.
    # ......
    
    json.dump(filtered_annos, open(rname.split('.')[0] + '_t.bbox.json','w'))
    
