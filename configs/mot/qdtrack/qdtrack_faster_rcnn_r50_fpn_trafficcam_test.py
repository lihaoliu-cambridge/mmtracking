_base_ = [
    './qdtrack_faster_rcnn_r50_fpn_trafficcam.py'
]
# dataset settings
data_root = 'data/traffic_cam/'
data = dict(
    test=dict(
        ann_file=data_root + 'annotations/coco_vid/traffic_cam_unlabeled.json',
        img_prefix=data_root + 'FirstFrame_annotate'
        ))
