import copy
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
    def find_frames_to_exclude():
        # First find what is the difference between vehicles and pedestrians per frame
        vehicle_counts = {}
        pedestrian_counts = {}
        for frame in objects_frame_dict:
            vehicles = objects_frame_dict[frame]['vehicle']
            pedestrians = objects_frame_dict[frame]['pedestrian']
            vehicle_counts[frame] = vehicles
            pedestrian_counts[frame] = pedestrians

        frames_to_exclude = []
        for frame in objects_frame_dict:
            vehicles = sum(vehicle_counts.values())
            pedestrians = sum(pedestrian_counts.values())
            print(f'vehic {vehicles} peds {pedestrians} threshold {vehicles*args.thresh/100}')
            if pedestrians > vehicles * args.thresh/100:
                break
            if objects_frame_dict[frame]['vehicle'] > objects_frame_dict[frame]['pedestrian']:
                frames_to_exclude.append(frame)
                del vehicle_counts[frame]
                del pedestrian_counts[frame]
        return frames_to_exclude

    def remove_entries_from_new_coco(frames_to_exclude):
        with open(args.out, 'r') as f:
            new_coco_data = json.load(f)

        new_images = [x for x in new_coco_data['images'] if x['id'] not in frames_to_exclude]
        new_anns = [x for x in new_coco_data['annotations'] if x['image_id'] not in frames_to_exclude]
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
    frames_to_exclude = find_frames_to_exclude()
    remove_entries_from_new_coco(frames_to_exclude)


def oversample(args):
    def find_frames_to_add():
        # First find what is the difference between vehicles and pedestrians per frame
        frames_to_consider = []
        vehicle_counts = []
        pedestrian_counts = []
        category_difference = {}
        for frame in objects_frame_dict:
            vehicles = objects_frame_dict[frame]['vehicle']
            pedestrians = objects_frame_dict[frame]['pedestrian']
            if vehicles < pedestrians:
                frames_to_consider.append(frame)
            vehicle_counts.append(vehicles)
            pedestrian_counts.append(pedestrians)
            category_difference[frame] = vehicles-pedestrians

        frames_more_pedestrians = [x for x in category_difference if category_difference[x] < 0]
        new_vehicle_counts = sum(vehicle_counts)
        new_pedestrian_counts = sum(pedestrian_counts)
        frames_to_add = []
        while True:
            for frame in frames_more_pedestrians:
                if new_pedestrian_counts > new_vehicle_counts * args.thresh / 100:
                    break
                frames_to_add.append(frame)
                new_vehicle_counts += objects_frame_dict[frame]['vehicle']
                new_pedestrian_counts += objects_frame_dict[frame]['pedestrian']
                print('new_vehicle_count', new_vehicle_counts,
                      'new_pedestrian_count', new_pedestrian_counts,
                      "minimum pedestrians necessary", new_vehicle_counts * args.thresh / 100)
            if new_pedestrian_counts > new_vehicle_counts * args.thresh / 100:
                break
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

            # Re-adding images [and also creating new copies]
            new_entry = copy.deepcopy(entry)
            old_img_name = new_entry['file_name']
            new_img_name = f"{new_entry['id']}{entry_idx}"
            shutil.copy(os.path.join(args.img_in_dir, old_img_name), os.path.join(args.img_out_dir, new_img_name + ".jpg"))
            new_entry['file_name'] = f"{new_img_name}.jpg"
            new_entry['id'] = int(new_img_name)
            new_coco_data['images'].append(new_entry)

            # Re-adding annotations
            for ann in new_coco_data['annotations']:
                if ann['image_id'] == old_img_name:
                    new_ann = copy.deepcopy(ann)
                    new_ann['image_id'] = int(new_img_name)
                    new_coco_data['annotations'].append(new_ann)

        with open(args.out, 'w') as f:
            json.dump(new_coco_data, f)

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
    parser.add_argument("sample_type", type=str, help="choose \"oversample\" or \"undersample\"", default='undersample')
    parser.add_argument("anns", type=str, help='coco annotations file', default="/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/datasets/Waymo/skip10_dataset/anns_coco/waymo_skip10_train.json")
    parser.add_argument("out", type=str, help="name of new coco annotations file to be created", default="coco_balanced_anns.json")
    parser.add_argument("--img_in_dir", type=str, help="[ONLY FOR OVERSAMPLE SAMPLE TYPE] input images directory", default='/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/datasets/Waymo/skip10_dataset/imgs_jpg')
    parser.add_argument("--img_out_dir", type=str, help="[ONLY FOR OVERSAMPLE SAMPLE TYPE] specifies the directory where additional"
                                                        "image files are going to be created", default='test_dir')
    parser.add_argument("--thresh", type=int, default=100,
                        help="set proportion of annotations between pedestrians and vehicles;"
                             "e.g.: --thresh 10 will ensure that the number of pedestrians is at least 10% of vehicles;"
                             "--thresh 100 will make it so that they are roughly the same")
    args = parser.parse_args()

    assert args.sample_type == "oversample" or args.sample_type == "undersample"
    if args.sample_type.lower() == "oversample":
        oversample(args)
    elif args.sample_type.lower() == "undersample":
        undersample(args)

