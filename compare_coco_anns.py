import json
import os


def load_json_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    splits = ['train']
    for split in splits:
        coco_anns = [f"/home/alan/workspace/Mestrado/dataset/WAYMO_skip10/anns_coco/{split}.json", \
                     f"/home/alan/workspace/Mestrado/dataset/WAYMO_skip10/anns_coco/balanced_anns/{split}_oversampled.json", \
                     f"/home/alan/workspace/Mestrado/dataset/WAYMO_skip10/anns_coco/balanced_anns/{split}_undersampled.json"]
        data = [load_json_data(x) for x in coco_anns]

        print('split', split)
        for x in zip(coco_anns, data):
            print("annotations, images")
            print(x[0], len(x[1]['annotations']), len(x[1]['images']))
