# Run following commands to do semi-supervised pseudo labeling

python tools/test.py configs/mot/qdtrack/qdtrack_faster_rcnn_r50_fpn_trafficcam_test.py  work_dirs/qdtrack_faster_rcnn_r50_fpn_trafficcam/latest.pth  --out labels/qdtrack_unlabeled_1_1_trafficcam.bbox.json

python scripts/traffic_cam/pkl2json.py  labels/qdtrack_unlabeled_1_1_trafficcam.pkl --generatepl_config configs/mot/qdtrack/qdtrack_faster_rcnn_r50_fpn_trafficcam_test.py
python scripts/traffic_cam/filter_pl.py labels/qdtrack_unlabeled_1_1_trafficcam.pkl --first_frame_labeled_json data/traffic_cam/annotations/traffic_cam_first_frame_labeled.json
python scripts/traffic_cam/form_ann.py  labels/qdtrack_unlabeled_1_1_trafficcam.pkl --plname data/traffic_cam/annotations/traffic_cam_unlabel_pl_qdtrack.json --unlabeled_json 'data/traffic_cam/annotations/traffic_cam_unlabeled.json'
