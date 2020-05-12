import os
import sys
import shutil
import argparse
import json
from shutil import copyfile


def collect_annotations_per_frame(coco_data):
    categories = {x['id']: x['name'] for x in coco_data['categories']}
    objects_frame_dict = {x['id']: {a: 0 for a in categories.values()} for x in coco_data['images']}
    frames_len = len(coco_data['images'])
    last_frame_idx = 0  # Annotations are ordered according per image, so this makes indexing faster
    for frame_idx, frame in enumerate(coco_data['images']):
        if frame_idx % 100 == 0:
            print(f"{frame_idx}/{frames_len}")
        for ann_idx in range(last_frame_idx, len(coco_data['annotations'])):
            ann = coco_data['annotations'][ann_idx]
            if frame['id'] == ann['image_id']:
                object_class = categories[ann['category_id']]
                objects_frame_dict[frame['id']][object_class] += 1
            else:
                last_frame_idx = ann_idx
                break
    return objects_frame_dict


def undersample(args):
    def find_frames_to_keep():
        # remover da maior diferenca para a menor diferença (n_vehics >> n_pedestres)
        threshold = [int((1 - args.thresh / 100) * len(coco_data['images'])),
                     int((1 + args.thresh / 100) * len(coco_data['images']))]
        frames_order = [x for x in objects_frame_dict]
        vehicle_counts = [objects_frame_dict[x]['vehicle'] for x in objects_frame_dict]
        pedestrian_counts = [objects_frame_dict[x]['pedestrian'] for x in objects_frame_dict]

        # First find what is the difference between vehicles and pedestrians per frame
        category_difference = []
        for i in range(len(frames_order)):
            category_difference.append(vehicle_counts[i] - pedestrian_counts[i])
        vehicle_ped_difference = sum(category_difference)

        # Then begin removing the frames whose difference are the largest, until the amount of vehicles is similar to pedestrians
        category_difference = list(zip(category_difference, frames_order))
        category_difference.sort(key=lambda x: x[0])
        while not (threshold[0] < vehicle_ped_difference < threshold[1]):
            del category_difference[-1]
            vehicle_ped_difference = sum(x[0] for x in category_difference)
        frames_to_keep = [x[1] for x in category_difference]
        return frames_to_keep

    def remove_entries_from_new_coco(frames_to_keep):
        with open(args.out, 'r') as f:
            new_coco_data = json.load(f)

        new_images = [x for x in new_coco_data['images'] if x['id'] in frames_to_keep]
        new_anns = [x for x in new_coco_data['annotations'] if x['image_id'] in frames_to_keep]
        new_coco_data['images'] = new_images
        new_coco_data['annotations'] = new_anns
        with open(args.out, 'w') as f:
            json.dump(new_coco_data, f)

    # Data preprocessing
    with open(args.anns, 'r') as f:
        coco_data = json.load(f)
    copyfile(args.anns, args.out)
    # Collecting all annotations PER FRAME at first
    objects_frame_dict = collect_annotations_per_frame(coco_data)
    # Removing frames where n_vehicles >> n_pedestrians, trying to keep the balance
    frames_to_keep = find_frames_to_keep()
    remove_entries_from_new_coco(frames_to_keep)


def oversample(args):
    def find_frames_to_add():
        # First find what is the difference between vehicles and pedestrians per frame
        frames_order = [x for x in objects_frame_dict]
        vehicle_counts = [objects_frame_dict[x]['vehicle'] for x in objects_frame_dict]
        pedestrian_counts = [objects_frame_dict[x]['pedestrian'] for x in objects_frame_dict]
        category_difference = []
        for i in range(len(frames_order)):
            category_difference.append(vehicle_counts[i] - pedestrian_counts[i])
        vehicle_ped_difference = sum(category_difference)

        # Then begin adding frames whose difference are the largest,
        # until the amount of PEDESTRIANS is similar to vehicles
        threshold = [int((1 - args.thresh / 100) * len(coco_data['images'])),
                     int((1 + args.thresh / 100) * len(coco_data['images']))]
        category_difference = list(zip(category_difference, frames_order))
        category_difference.sort(key=lambda x: x[0])
        frames_with_more_peds = [x for x in category_difference if x[0] < 0]

        for x in frames_with_more_peds:
            while not (threshold[0] < vehicle_ped_difference < threshold[1]):
                category_difference.append(x)
                vehicle_ped_difference = sum(x[0] for x in category_difference)
        frames_to_add = [x[1] for x in category_difference]
        return frames_to_add

    def add_entries_to_new_coco(frames_to_add):
        with open(args.out, 'r') as f:
            new_coco_data = json.load(f)

        new_images = [x for x in new_coco_data['images'] if x['id'] in frames_to_add]
        print('Copying repeated images and adding repeated annotations. This may take a while.')
        for entry_idx, entry in enumerate(new_images):
            sys.stdout.write("\r")
            sys.stdout.write(f'Img {entry_idx}/{len(new_images)}')
            sys.stdout.flush()

            # Re-adding images [and also creating new copies ones]
            old_img_name = entry['file_name']
            new_img_name = f"{entry['id']}{entry_idx}"
            shutil.copy(os.path.join(args.img_in_dir, old_img_name), os.path.join(args.img_out_dir, new_img_name + ".jpg"))
            entry['file_name'] = f"{new_img_name}.jpg"
            entry['id'] = int(new_img_name)
            new_coco_data['images'].append(entry)

            # Re-adding annotations
            for ann in new_coco_data['annotations']:
                if ann['image_id'] == old_img_name:
                    new_ann = ann
                    new_ann['image_id'] = int(new_img_name)
                    new_coco_data['annotations'].append(new_ann)

        with open(args.out, 'w') as f:
            json.dump(new_coco_data, f)

    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/coco.py#L115
    # Add repeated images and anns to coco file, but with a final idx at the end to differentiate them
    # Data preprocessing
    with open(args.anns, 'r') as f:
        coco_data = json.load(f)
    copyfile(args.anns, args.out)
    # Collecting all annotations PER FRAME at first
    objects_frame_dict = collect_annotations_per_frame(coco_data)
    # Finding which frames to (re)add to dataset
    frames_to_add = find_frames_to_add()
    add_entries_to_new_coco(frames_to_add)


if __name__ == "__main__":
    """
    Context: We have many bounding box annotations, however the total for each class is imbalanced.
    On my master's case, WAYMO on dataset vehicles appear up to 10 times more than pedestrians. Therefore, when training
    a detector on it, it becomes very biased to the vehicles class. To solve that, two options are possible:
    adding weights to the classes in the detector itself or hard mining pedestrians samples.
    On this script, I am doing the second by:
        Method A. Undersampling (removing samples)
        Method B. Oversampling (repeating samples) 
    """
    parser = argparse.ArgumentParser(description='Create database file for referencing how many samples of each frame should be collected')
    parser.add_argument("sample_type", type=str, help="choose \"oversample\" or \"undersample\"", default='oversample')
    parser.add_argument("anns", type=str, help='coco annotations file', default="waymo_skip10_train.json")
    parser.add_argument("out", type=str, help="name of new coco annotations file to be created", default="coco_balanced_anns.json")
    parser.add_argument("--img_in_dir", type=str, help="[ONLY FOR OVERSAMPLE SAMPLE TYPE] input images directory", default='/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/datasets/Waymo/skip10_dataset/imgs_jpg')
    parser.add_argument("--img_out_dir", type=str, help="[ONLY FOR OVERSAMPLE SAMPLE TYPE] specifies the directory where additional"
                                                        "image files are going to be created", default='output_test')
    parser.add_argument("--thresh", type=int, default=10,
                        help="sets upper and lower boundaries files creation proportional to len of coco anns file."
                             "e.g.: --thresh 10 will create a threshold between 90%% and 110%%")
    args = parser.parse_args()

    assert args.sample_type == "oversample" or args.sample_type == "undersample"
    if args.sample_type.lower() == "oversample":
        oversample(args)
    elif args.sample_type.lower() == "undersample":
        undersample(args)

