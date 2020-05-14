import json
import os
import argparse


def load_coco_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def create_new_json(carla_data, waymo_data, split, out_dir):
    carla_data['images'] += waymo_data['images']
    start_ann_id = len(carla_data['annotations'])
    for ann_idx, ann in enumerate(waymo_data['annotations']):
        ann['id'] = start_ann_id + ann_idx
        carla_data['annotations'].append(ann)
    with open(os.path.join(out_dir, split + ".json"), 'w') as f:
        json.dump(carla_data, f)
    print(f"Created {split}.json file")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates a new mixed dataset (train and val) based on data from"
                                                 " two json files")
    parser.add_argument("carla_json_dir", type=str, help="Dir containing train and val COCO JSON files for carla")
    parser.add_argument("waymo_json_dir", type=str, help="Dir containing train and val COCO JSON files for waymo")
    parser.add_argument("out_dir", type=str, help="Dir to store new json files")
    args = parser.parse_args()
    splits = ['train', 'val']

    for split in splits:
        carla_json = os.path.join(args.carla_json_dir, split + ".json")
        waymo_json = os.path.join(args.waymo_json_dir, split + ".json")
        carla_data = load_coco_data(carla_json)
        waymo_data = load_coco_data(waymo_json)
        create_new_json(carla_data, waymo_data, split, args.out_dir)
