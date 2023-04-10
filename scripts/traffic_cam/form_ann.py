import json
from copy import deepcopy
import numpy as np
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser(
        description='forming annotation file')
    parser.add_argument('rname')
    parser.add_argument('--pl_name')
    parser.add_argument('--unlabeled_json', default=None, help='unlabeled json')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    rname = args.rname
    pl_name = args.pl_name
    unlabeled_json = args.unlabeled_json
    
    detection_results = json.load(open(rname.split('.')[0] + '_t_s.bbox.json'))
    unlabeled_json = json.load(open(unlabeled_json))

    b = detection_results
    j = 0
    for i in range(len(b)):
        x1, x2, y1, y2 = [b[i]['bbox'][0], b[i]['bbox'][0]+b[i]['bbox'][2], b[i]['bbox'][1], b[i]['bbox'][1]+b[i]['bbox'][3]]
        b[i]['area'] = b[i]['bbox'][2] * b[i]['bbox'][3]
        j = j + 1
        b[i]['id'] = j

    # Add the annotations to the unlabeled_json
    unlabeled_json['annotations'] = b
    
    json.dump(unlabeled_json, open(pl_name, 'w'))
