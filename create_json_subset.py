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
    mixed_json = "waymo_and_carla_skip10\\train.json"
    waymo_skip10_json = "waymo_skip10_train.json"
    out_json = "new_train.json"
    create_subset_json(mixed_json, waymo_skip10_json, out_json)
