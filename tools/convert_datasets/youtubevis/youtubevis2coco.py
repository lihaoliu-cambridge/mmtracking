# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
from collections import defaultdict

import mmcv
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='YouTube-VIS to COCO Video format')
    parser.add_argument(
        '-i',
        '--input',
        help='root directory of YouTube-VIS annotations',
    )
    parser.add_argument(
        '-o',
        '--output',
        help='directory to save coco formatted label file',
    )
    parser.add_argument(
        '--version',
        choices=['2019', '2021'],
        help='The version of YouTube-VIS Dataset',
    )
    return parser.parse_args()


def convert_vis(ann_dir, save_dir, dataset_version, mode='train'):
    """Convert YouTube-VIS dataset in COCO style.

    Args:
        ann_dir (str): The path of YouTube-VIS dataset.
        save_dir (str): The path to save `VIS`.
        dataset_version (str): The version of dataset. Options are '2019',
            '2021'.
        mode (str): Convert train dataset or validation dataset or test
            dataset. Options are 'train', 'valid', 'test'. Default: 'train'.
    """
    assert dataset_version in ['2019', '2021']
    assert mode in ['train', 'valid', 'test', 'train_first_frame', 'train_extra', 'unlabeled', 'first_frame_labeled']
    VIS = defaultdict(list)
    records = dict(vid_id=1, img_id=1, ann_id=1, global_instance_id=1)
    obj_num_classes = dict()

    if dataset_version == '2019':
        official_anns = mmcv.load(osp.join(ann_dir, f'{mode}.json'))
    elif dataset_version == '2021':
        official_anns = mmcv.load(osp.join(ann_dir, mode, 'instances.json'))
    VIS['categories'] = copy.deepcopy(official_anns['categories'])

    has_annotations = False if mode in ['unlabeled'] else True
    if has_annotations:
        vid_to_anns = defaultdict(list)
        for ann_info in official_anns['annotations']:
            vid_to_anns[ann_info['video_id']].append(ann_info)

    video_infos = official_anns['videos']
    for video_info in tqdm(video_infos):
        video_name = video_info['file_names'][0].split(os.sep)[0]
        video = dict(
            id=video_info['id'],
            name=video_name,
            width=video_info['width'],
            height=video_info['height'])
        VIS['videos'].append(video)

        num_frames = len(video_info['file_names'])
        width = video_info['width']
        height = video_info['height']
        if has_annotations:
            ann_infos_in_video = vid_to_anns[video_info['id']]
            instance_id_maps = dict()

        for frame_id in range(num_frames):
            image = dict(
                file_name=video_info['file_names'][frame_id],
                height=height,
                width=width,
                id=records['img_id'],
                frame_id=frame_id,
                video_id=video_info['id'])
            VIS['images'].append(image)

            skip_empty_frames = True if mode in ['first_frame_labeled'] and frame_id != 0 else False
            
            if has_annotations and not skip_empty_frames:
                for ann_info in ann_infos_in_video:
                    bbox = ann_info['bboxes'][frame_id]
                    if bbox is None:
                        continue

                    category_id = ann_info['category_id']
                    track_id = ann_info['id']
                    segmentation = ann_info['segmentations'][frame_id]
                    area = ann_info['areas'][frame_id]
                    assert isinstance(category_id, int)
                    assert isinstance(track_id, int)
                    assert segmentation is not None
                    assert area is not None

                    if track_id in instance_id_maps:
                        instance_id = instance_id_maps[track_id]
                    else:
                        instance_id = records['global_instance_id']
                        records['global_instance_id'] += 1
                        instance_id_maps[track_id] = instance_id

                    ann = dict(
                        id=records['ann_id'],
                        video_id=video_info['id'],
                        image_id=records['img_id'],
                        category_id=category_id,
                        instance_id=instance_id,
                        bbox=bbox,
                        segmentation=segmentation,
                        area=area,
                        iscrowd=ann_info['iscrowd'])

                    if category_id not in obj_num_classes:
                        obj_num_classes[category_id] = 1
                    else:
                        obj_num_classes[category_id] += 1

                    VIS['annotations'].append(ann)
                    records['ann_id'] += 1
            records['img_id'] += 1
        records['vid_id'] += 1

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    mmcv.dump(VIS,
              osp.join(save_dir, f'traffic_cam_{mode}.json'))
    print(f'-----YouTube VIS {dataset_version} {mode}------')
    print(f'{records["vid_id"]- 1} videos')
    print(f'{records["img_id"]- 1} images')
    if has_annotations:
        print(f'{records["ann_id"] - 1} objects')
        print(f'{records["global_instance_id"] - 1} instances')
    print('-----------------------')
    if has_annotations:
        for i in range(1, len(VIS['categories']) + 1):
            class_name = VIS['categories'][i - 1]['name']
            i_of_obj_num_classes = obj_num_classes[i] if i in obj_num_classes else 0
            print(f'Class {i} {class_name} has {i_of_obj_num_classes} objects.')


def main():
    args = parse_args()

    input = '/rds/project/rds-xfbi6l4KMrM/yc443/data/tracking_v0/data_clean/annotations/youtube_vis'
    output = '/rds/project/rds-xfbi6l4KMrM/yc443/data/tracking_v0/data_clean/annotations/coco_vid'
    version = '2019'

    # for sub_set in ['train', 'valid', 'test', 'train_first_frame', 'train_extra', 'unlabeled', 'first_frame_labeled', 'sample_train', 'sample_unlabeled', 'sample_first_frame_labeled']:
    # for sub_set in ['train', 'valid', 'test', 'train_first_frame', 'train_extra', 'sample_train']:
    # for sub_set in ['unlabeled']: #, 'sample_unlabeled']:
    for sub_set in ['first_frame_labeled']: # , 'sample_first_frame_labeled']:
        # convert_vis(args.input, args.output, args.version, sub_set)
        convert_vis(input, output, version, sub_set)

if __name__ == '__main__':
    main()
