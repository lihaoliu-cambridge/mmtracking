_base_ = [
    './qdtrack_faster_rcnn_r50_fpn_trafficcam.py'
]
data_root = 'data/traffic_cam/'
dataset_type = 'CocoVideoDataset'
classes = ('Motor_Bike', 'Bus', 'LMV', 'Auto', 'Bike', 'Pedestrian', 'LCV', 'E-rickshaw', 'Tractor', 'Truck')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(
        type='SeqResize',
        img_scale=(1088, 1088),
        share_params=True,
        ratio_range=(0.8, 1.2),
        keep_ratio=True,
        bbox_clip_border=False),
    dict(type='SeqPhotoMetricDistortion', share_params=True),
    # dict(
    #     type='SeqRandomCrop',
    #     share_params=False,
    #     crop_size=(1088, 1088),
    #     bbox_clip_border=False),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='MatchInstances', skip_nomatch=True),
    dict(
        type='VideoCollect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels',
            'gt_match_indices',   ###gt_match_indices (list(Tensor)): Mapping from gt_instance_ids to ref_gt_instance_ids of the same tracklet in a pair of images.
            'gt_instance_ids'
        ]),
    # dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
data = dict(
    train=[
        dict(
            type=dataset_type,
            # load_as_video=False,
            classes=classes,
            ann_file=data_root + 'annotations/coco_vid/traffic_cam_train.json',
            img_prefix=data_root + 'Fully_annotate',
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=10,
                filter_key_img=True,
                method='uniform'),
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            # load_as_video=False,
            classes=classes,
            ann_file=data_root + 'annotations/coco_vid/traffic_cam_train_extra.json',
            img_prefix=data_root + 'Fully_annotate',
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=10,
                filter_key_img=True,
                method='uniform'),
            pipeline=train_pipeline),
        dict(
            type=dataset_type,
            # load_as_video=False,
            classes=classes,
            ann_file=data_root + 'annotations/coco_vid/traffic_cam_valid.json',
            img_prefix=data_root + 'Fully_annotate',
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=10,
                filter_key_img=True,
                method='uniform'),
            pipeline=train_pipeline),
        # To Zhongying:
        # Here, we don't need traffic_cam_train_first_frame.json anymore,
        # because we have traffic_cam_unlabel_pl.json now, which included the first frame ground truth.
        dict(
            type=dataset_type,
            # load_as_video=False,
            classes=classes,
            ann_file=data_root + 'annotations/coco_vid/traffic_cam_unlabeled-pl.json',
            img_prefix=data_root + 'FirstFrame_annotate',
            ref_img_sampler=dict(
                num_ref_imgs=1,
                frame_range=10,
                filter_key_img=True,
                method='uniform'),
            pipeline=train_pipeline),
    ])