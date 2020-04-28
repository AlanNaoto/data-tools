import os
import argparse
import json
from shutil import copyfile


def filter_annotations(args, coco_data):
    new_categories = [{"supercategory": "none", "id": 1, "name": args.category}]
    new_anns = []
    old_category_id = [x['id'] for x in coco_data['categories'] if x['name'] == args.category]
    old_category_id = old_category_id[0]
    for ann_idx, ann in enumerate(coco_data['annotations']):
        if ann['category_id'] == old_category_id:
            ann['category_id'] = 1  # Single class will always have this ID
            new_anns.append(ann)
    return new_anns, new_categories


def create_new_json(args):
    # Data preprocessing
    with open(args.anns, 'r') as f:
        coco_data = json.load(f)
    copyfile(args.anns, args.out)

    new_anns, new_categories = filter_annotations(args, coco_data)

    # Write on new json the data
    with open(args.out, 'r') as f:
        new_coco_data = json.load(f)
    new_coco_data['annotations'] = new_anns
    new_coco_data['categories'] = new_categories
    with open(args.out, 'w') as f:
        json.dump(new_coco_data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose annotation category to be kept from the coco file')
    parser.add_argument("anns", type=str, help='coco annotations file', default="waymo_skip10_train.json")
    parser.add_argument("category", type=str, help="name of category to keep", default="pedestrian")
    parser.add_argument("out", type=str, help="name of new coco annotations file to be created", default="waymo_skip10_train_vehicles.json")
    args = parser.parse_args()

    create_new_json(args)

