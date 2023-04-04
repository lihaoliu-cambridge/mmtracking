### ask for interactive GPU with TRAFFIC-SL2-GPU project
sintr -p ampere -t 12:00:00 --nodes=1 --gpus-per-node=1 -A TRAFFIC-SL2-GPU

### cd to the directory, please specify below
cd {your_path}/mmtracking

### activate virtual environment and load modules
conda activate open-mmlab
module load cuda/11.1 cudnn/8.0_cuda-11.1


### run the python files according to each task

#### mot task: configs/mot
########## python tools/train.py configs/mot/{config_folder}/{config_file}.py
########## python tools/test.py configs/mot/{config_folder}/{config_file}.py --checkpoint work_dirs/{config_file}/latest.pth --out work_dirs/{config_file}/results.pkl --eval bbox track

##### bytetrack: configs/mot/bytetrack  #{config_folder}/{config_file}=bytetrack/bytetrack_faster_rcnn_r50_fpn_trafficcam  
python tools/train.py configs/mot/bytetrack/bytetrack_faster_rcnn_r50_fpn_trafficcam.py

python tools/test.py configs/mot/bytetrack/bytetrack_faster_rcnn_r50_fpn_trafficcam.py --checkpoint work_dirs/bytetrack_faster_rcnn_r50_fpn_trafficcam/latest.pth --out work_dirs/bytetrack_faster_rcnn_r50_fpn_trafficcam/results.pkl --eval bbox track >> bytetrack_faster_rcnn_r50_fpn_trafficcam.txt


##### deepsort: configs/mot/deepsort  #{config_folder}/{config_file}=deepsort/deepsort_faster_rcnn_r50_fpn_trafficcam
python tools/train.py configs/mot/deepsort/deepsort_faster_rcnn_r50_fpn_trafficcam.py

python tools/test.py configs/mot/deepsort/deepsort_faster_rcnn_r50_fpn_trafficcam.py --checkpoint work_dirs/deepsort_faster_rcnn_r50_fpn_trafficcam/latest.pth --out work_dirs/deepsort_faster_rcnn_r50_fpn_trafficcam/results.pkl --eval bbox track >> deepsort_faster_rcnn_r50_fpn_trafficcam.txt

##### ocsort: configs/mot/ocsort  #{config_folder}/{config_file}=ocsort/ocsort_faster_rcnn_r50_fpn_trafficcam
python tools/train.py configs/mot/ocsort/ocsort_faster_rcnn_r50_fpn_trafficcam.py

python tools/test.py configs/mot/ocsort/ocsort_faster_rcnn_r50_fpn_trafficcam.py --checkpoint work_dirs/ocsort_faster_rcnn_r50_fpn_trafficcam/latest.pth --out work_dirs/ocsort_faster_rcnn_r50_fpn_trafficcam/results.pkl --eval bbox track >> ocsort_faster_rcnn_r50_fpn_trafficcam.txt


##### qdtrack: configs/mot/qdtrack  #{config_folder}/{config_file}=qdtrack/qdtrack_faster_rcnn_r50_fpn_trafficcam
python tools/train.py configs/mot/qdtrack/qdtrack_faster_rcnn_r50_fpn_trafficcam.py

python tools/test.py configs/mot/qdtrack/qdtrack_faster_rcnn_r50_fpn_trafficcam.py --checkpoint work_dirs/qdtrack_faster_rcnn_r50_fpn_trafficcam/latest.pth --out work_dirs/qdtrack_faster_rcnn_r50_fpn_trafficcam/results.pkl --eval bbox track >> qdtrack_faster_rcnn_r50_fpn_trafficcam.txt


##### tracktor: configs/mot/tracktor  #{config_folder}/{config_file}=tracktor/tracktor_faster_rcnn_r50_fpn_trafficcam
python tools/train.py configs/mot/tracktor/tracktor_faster_rcnn_r50_fpn_trafficcam.py

python tools/test.py configs/mot/tracktor/tracktor_faster_rcnn_r50_fpn_trafficcam.py --checkpoint work_dirs/tracktor_faster_rcnn_r50_fpn_trafficcam/latest.pth --out work_dirs/tracktor_faster_rcnn_r50_fpn_trafficcam/results.pkl --eval bbox track >> tracktor_faster_rcnn_r50_fpn_trafficcam.txt



