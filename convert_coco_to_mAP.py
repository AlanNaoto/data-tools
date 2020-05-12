import json
import os


def convert_coco_bb_to_mAP(json_coco_file, out_dir):
    with open(json_coco_file, 'r') as f:
        coco_data = json.load(f)

    # Creates empty txt files for frames without anns
    for img in coco_data['images']:
        with open(os.path.join(out_dir, str(img['id']) + '.txt'), 'w') as f:
            pass

    # Creates ann txt files
    categories = {x['id']: x['name'] for x in coco_data['categories']}
    ann_len = len(coco_data['annotations'])
    for ann_idx, ann in enumerate(coco_data['annotations']):
        print(f'{ann_idx}/{ann_len}')
        with open(os.path.join(out_dir, str(ann['image_id']) + '.txt'), 'a') as f:
            f.write(categories[ann['category_id']])
            xmin, ymin, width, height = ann['bbox']
            xmax = xmin + width
            ymax = ymin + height
            bbox = [xmin, ymin, xmax, ymax]
            [f.write(" " + str(x)) for x in bbox]
            f.write('\n')


if __name__ == "__main__":
    json_coco_file = "/home/alan/workspace/Mestrado/dataset/CARLA_1920x1280_skip10_WAYMO_skip10/anns_coco/test.json"
    out_dir = "/home/alan/workspace/Mestrado/dataset/CARLA_1920x1280_skip10_WAYMO_skip10/anns_mAP"
    convert_coco_bb_to_mAP(json_coco_file, out_dir)
