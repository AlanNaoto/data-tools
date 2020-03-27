import numpy as np
import sys
import cv2
import os
import xml.etree.ElementTree as ET


def get_bb_data(ann_file):
    bb_vehicles, bb_walkers = [], []
    with open(ann_file, 'r') as f:
        data = f.read().splitlines()
    vehicles_generalized = ['car', 'van', 'truck']
    for ann in data:
        object_data = ann.split(" ")
        object_class = object_data[0].lower()
        truncated = object_data[1]  # Float from 0 (non-truncated) to 1 (truncated), where truncated refers to the object leaving image boundaries
        occluded = object_data[2]  # 0 = fully visible, 1 = partly occluded 2 = largely occluded, 3 = unknown
        xmin = int(float(object_data[4]))
        ymin = int(float(object_data[5]))
        xmax = int(float(object_data[6]))
        ymax = int(float(object_data[7]))
        if object_class in vehicles_generalized:
            bb_vehicles.append([xmin, ymin, xmax, ymax, truncated, occluded])
        elif object_class == "pedestrian":
            bb_walkers.append([xmin, ymin, xmax, ymax, truncated, occluded])
    return bb_vehicles, bb_walkers


def create_xml_file(frame_name, img_path, bbs_out_dir, frame_width, frame_height, bb_vehicles_data, bb_walkers_data):
    def create_class_entry(annotation, bb_data=None, object=None):
        # Per class data
        object_element = ET.SubElement(annotation, 'object')
        name = ET.SubElement(object_element, "name")
        pose = ET.SubElement(object_element, "pose")
        truncated = ET.SubElement(object_element, "truncated")
        difficult = ET.SubElement(object_element, "difficult")
        occluded = ET.SubElement(object_element, "occluded")
        bndbox = ET.SubElement(object_element, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmax = ET.SubElement(bndbox, "xmax")
        ymin = ET.SubElement(bndbox, "ymin")
        ymax = ET.SubElement(bndbox, "ymax")

        if bb_data is not None:
            name.text = object
            xmin.text = str(bb_data[0])
            ymin.text = str(bb_data[1])
            xmax.text = str(bb_data[2])
            ymax.text = str(bb_data[3])
            truncated.text = bb_data[4]
            occluded.text = bb_data[5]
        return annotation

    # Creating general structure
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder').text = os.path.dirname(img_path)
    filename = ET.SubElement(annotation, 'filename').text = os.path.basename(img_path)
    path = ET.SubElement(annotation, 'path').text = img_path

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')

    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width').text = str(frame_width)
    height = ET.SubElement(size, 'height').text = str(frame_height)
    depth = ET.SubElement(size, 'depth').text = '3'
    segmented = ET.SubElement(annotation, 'segmented')

    # 0=vehicle, 1=pedestrian
    for vehicle in bb_vehicles_data:
        annotation = create_class_entry(annotation, vehicle, object='vehicle')
    for walker in bb_walkers_data:
        annotation = create_class_entry(annotation, walker, object='pedestrian')
    if bb_vehicles_data is None and bb_walkers_data is None:
        annotation = create_class_entry(annotation)

    # Creating the file
    tree = ET.ElementTree(annotation)
    tree.write(os.path.join(bbs_out_dir, f'{frame_name}.xml'))


def create_voc_anns(bbs_kitti, imgs_dir, bbs_out_dir):
    ann_files = [os.path.join(bbs_kitti, x) for x in os.listdir(bbs_kitti) if x.endswith(".txt")]
    img_files = [os.path.join(imgs_dir, x) for x in os.listdir(imgs_dir) if x.endswith(".jpg")]

    tmp_img = cv2.imread(img_files[0])
    frame_width = tmp_img.shape[1]
    frame_height = tmp_img.shape[0]
    for frame_idx, frame_ann_file in enumerate(ann_files):
        sys.stdout.write("\r")
        sys.stdout.write('Saving frame {0}/{1}'.format(frame_idx+1, len(ann_files)))
        sys.stdout.flush()
        bb_vehicles, bb_walkers = get_bb_data(frame_ann_file)
        img_path = img_files[frame_idx]
        frame_name = os.path.basename(frame_ann_file).replace(".txt", "")
        create_xml_file(frame_name, img_path, bbs_out_dir, frame_width, frame_height, bb_vehicles, bb_walkers)
    print('\nDone.')


if __name__ == "__main__":
    imgs_dir = "/home/alan/workspace/Mestrado/dataset/KITTI/imgs_jpg/train"
    bbs_kitti_dir = "/home/alan/workspace/Mestrado/dataset/KITTI/anns_raw"
    bbs_out_dir = "/home/alan/workspace/Mestrado/dataset/KITTI/anns_voc"
    create_voc_anns(bbs_kitti_dir, imgs_dir, bbs_out_dir)

