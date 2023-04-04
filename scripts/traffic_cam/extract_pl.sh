
GPUS=$1
RESULTNAME=$2
ANNFILE=$3

# Method Paper
python tools/test.py configs/cascade_rpn_selsa/crpn_faster_rcnn_r50_fpn_2x_coco_video_test.py  work_dirs/crpn_faster_rcnn_r50_fpn_2x_coco_video/latest.pth --out labels/unlabeled_1_1_trafficcam.pkl
python scripts/traffic_cam/pkl2json.py  labels/unlabeled_1_1_trafficcam.pkl --generatepl_config 'configs/cascade_rpn_selsa/crpn_faster_rcnn_r50_fpn_2x_coco_video_test.py'
python scripts/traffic_cam/filter_pl.py labels/unlabeled_1_1_trafficcam.pkl --labeled_json 'data/coco/annotations/instances_train2017.1_1-labeled.json' --unlabeled_json 'data/traffic_cam/annotations/traffic_cam_unlabel_samples.json'
python scripts/traffic_cam/filter_pl_straight_line.py labels/unlabeled_1_1_trafficcam.pkl 
python scripts/traffic_cam/form_ann.py  labels/unlabeled_1_1_trafficcam.pkl data/traffic_cam/annotations/traffic_cam_unlabel-pl.json --labeled_json 'data/traffic_cam/annotations/traffic_cam_train.json' --unlabeled_json 'data/traffic_cam/annotations/traffic_cam_unlabel_samples.json'
