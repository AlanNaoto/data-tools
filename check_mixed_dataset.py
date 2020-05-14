import joblib
import json


def load_json_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    for split in ['train', 'val', 'test']:
        carla_coco_ann = f"/home/alan/workspace/Mestrado/dataset/CARLA_1920x1280_skip10/anns_coco/carla_1920x1280_{split}.json"
        waymo_coco_ann = f"/home/alan/workspace/Mestrado/dataset/WAYMO_skip10/anns_coco/{split}.json"
        mixed_coco_ann = f"/home/alan/workspace/Mestrado/dataset/CARLA_1920x1280_skip10_WAYMO_skip10/anns_coco/{split}.json"
        mixed_coco_waymo_subset_ann = f"/home/alan/workspace/Mestrado/dataset/CARLA_1920x1280_skip10_WAYMO_skip10/anns_coco/{split}_waymo_subset.json"

#    carla = load_json_data(carla_coco_ann)
        waymo = load_json_data(waymo_coco_ann)
        mixed = load_json_data(mixed_coco_ann)
        mixed_waymo = load_json_data(mixed_coco_waymo_subset_ann)
        
        print('split', split)
 #   print('carla', len(carla['annotations']))
        print("waymo", len(waymo['images']))
        print("mixed", len(mixed["images"]))
        print("mixed_waymo", len(mixed_waymo['images']))
