## build the virtual environment and install relevant packaged
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10/index.html


## clone/copy-and-paste the mmtracking folder


## build the soft link
### cd to the directory, please specify below
cd /rds/project/{}/mmtracking

### soft link: for accessing the dataset
ln -s /rds/project/rds-xfbi6l4KMrM/yc443/data/tracking_v0/data_clean/ data/traffic_cam


## run the fully supervised methods
###### please refer to trafficcam_script_fully-supervised_mot.md






