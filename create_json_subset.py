import json
import copy


def create_subset_json(full_json, subset_json, out_json):
    # IO
    with open(full_json, 'r') as f:
        data_full = json.load(f)
    with open(subset_json, 'r') as f:
        data_subset = json.load(f)

    # Seeing which files are from waymo, and not carla
    out_data = copy.deepcopy(data_full)
    waymo_file_names = [x['file_name'] for x in data_subset['images']]
    out_data['images'] = [x for x in out_data['images'] if x['file_name'] in waymo_file_names]
    out_data['annotations'] = [x for x in out_data['annotations'] if
                                            str(x['image_id']) + ".jpg" in waymo_file_names]

    # Writing a new JSON with only waymo_skip10 annotations
    with open(out_json, 'w') as f:
        json.dump(out_data, f)


if __name__ == "__main__":
    split = "test"
    mixed_json = f"/home/alan/workspace/Mestrado/dataset/CARLA_1920x1280_skip10_WAYMO_skip10/anns_coco/{split}.json"
    waymo_skip10_json = f"/home/alan/workspace/Mestrado/dataset/WAYMO_skip10/anns_coco/{split}.json"
    out_json = f"/home/alan/workspace/Mestrado/dataset/CARLA_1920x1280_skip10_WAYMO_skip10/anns_coco/{split}_waymo_subset.json"
    create_subset_json(mixed_json, waymo_skip10_json, out_json)
