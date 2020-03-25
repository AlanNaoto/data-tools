import os
import json

def read_json(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    return json_data

def normalize_yolo_bbox(yolo_bbox, images_size):
    img_width, img_height = images_size
    yolo_bbox[1] /= img_width
    yolo_bbox[2] /= img_height
    yolo_bbox[3] /= img_width
    yolo_bbox[4] /= img_height
    return yolo_bbox

def create_empty_annotation_files(json_data, output_dir):
    for img in json_data['images']:
        with open(os.path.join(output_dir, img['file_name'].replace('.jpeg', '.txt')), 'a') as f:  
            pass          
    return


def create_annotations_yolo(json_file, output_dir, images_size):
    json_data = read_json(json_file)
    os.makedirs(output_dir, exist_ok=False)

    # This is done so that every image has at least one corresponding txt file
    #create_empty_annotation_files(json_data, output_dir)

    # object_classes = json_data['categories']  # 1 = vehicle; 2 = pedestrian
    for idx, annotation in enumerate(json_data['annotations']):
        print('Working on annotation {}/{}'.format(idx, len(json_data['annotations'])))
        yolo_annotation_file_name = str(annotation['image_id']) + '.txt'
        coco_bbox = annotation['bbox']
        # For some reason there are negative entries (e.g. -1)... I am rounding them to zero
        if coco_bbox[0] < 0:
            coco_bbox[0] = 0
        if coco_bbox[1] < 0:
            coco_bbox[1] = 0

        # Yolo bbox will be a normalized one, with the coordinates being: [obj_class, xcenter, ycenter, width, height]
        object_class = annotation['category_id'] - 1  # YOLO from pytorch wants 0 indexed classes, and COCO is 1 indexed 
        bb_width = coco_bbox[2]
        bb_height = coco_bbox[3]
        bb_x_center = coco_bbox[0] + bb_width/2
        bb_y_center = coco_bbox[1] + bb_height/2
        yolo_bbox = [object_class, bb_x_center, bb_y_center, bb_width, bb_height]
        yolo_bbox = normalize_yolo_bbox(yolo_bbox, images_size)
        # Create a txt file with the data (or append to an already created txt, since many objects exist in the same image)
        with open(os.path.join(output_dir, yolo_annotation_file_name), 'a') as f:
            [f.write(str(x) + " ") for x in yolo_bbox]
            f.write('\n')


if __name__ == "__main__":
    # Input
    coco_train_json = "/home/aiss-v100/workspace/alan/dataset/coco/train_fold_1.json"
    coco_test_json = "/home/aiss-v100/workspace/alan/dataset/coco/test_fold_1.json"
    yolo_train_dir = "/home/aiss-v100/workspace/alan/dataset/YOLO_format/kfold_1/train"
    yolo_test_dir = "/home/aiss-v100/workspace/alan/dataset/YOLO_format/kfold_1/test"
    images_size = (1024, 768)

    # Process annotations
    print('Working on training anns...')
    create_annotations_yolo(coco_train_json, yolo_train_dir, images_size)
    print('Working on validation anns...')
    create_annotations_yolo(coco_test_json, yolo_test_dir, images_size)

