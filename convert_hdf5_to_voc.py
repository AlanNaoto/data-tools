import h5py
import numpy as np
import sys
import cv2
import os
import xml.etree.ElementTree as ET


def create_xml_file(folder, img_name, img_path, bbs_out_dir, frame_width, frame_height, time, bb_vehicles_data, bb_walkers_data):
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
            xmin.text = str(bb_data[0][0])
            ymin.text = str(bb_data[0][1])
            xmax.text = str(bb_data[1][0])
            ymax.text = str(bb_data[1][1])

        return annotation

    # Creating general structure
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder').text = folder
    filename = ET.SubElement(annotation, 'filename').text = img_name
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
    tree.write(os.path.join(bbs_out_dir, f'{time}.xml'))


def parse_bb_data(bb_vehicles_data, bb_walkers_data):
    bb_vehicles = []
    bb_walkers = []
    if all(bb_vehicles_data != -1):
        for bb_idx in range(0, len(bb_vehicles_data), 4):
            coordinate_min = (int(bb_vehicles_data[0 + bb_idx]), int(bb_vehicles_data[1 + bb_idx]))
            coordinate_max = (int(bb_vehicles_data[2 + bb_idx]), int(bb_vehicles_data[3 + bb_idx]))
            bb_vehicles.append([coordinate_min, coordinate_max])
    if all(bb_walkers_data != -1):
        for bb_idx in range(0, len(bb_walkers_data), 4):
            coordinate_min = (int(bb_walkers_data[0 + bb_idx]), int(bb_walkers_data[1 + bb_idx]))
            coordinate_max = (int(bb_walkers_data[2 + bb_idx]), int(bb_walkers_data[3 + bb_idx]))
            bb_walkers.append([coordinate_min, coordinate_max])
    return bb_vehicles, bb_walkers


def create_imgs_and_anns(hdf5_file, imgs_out_dir, bbs_out_dir):
    with h5py.File(hdf5_file, 'r') as file:
        frame_width = file.attrs['sensor_width']
        frame_height = file.attrs['sensor_height']
        folder = "imgs"
        for time_idx, time in enumerate(file['timestamps']['timestamps']):
            rgb_data = np.array(file['rgb'][str(time)])
            bb_vehicles_data = np.array(file['bounding_box']['vehicles'][str(time)])
            bb_walkers_data = np.array(file['bounding_box']['walkers'][str(time)])

            bb_vehicles, bb_walkers = parse_bb_data(bb_vehicles_data, bb_walkers_data)

            sys.stdout.write("\r")
            sys.stdout.write('Saving frame {0}/{1}'.format(time_idx+1, len(file['timestamps']['timestamps'])))
            sys.stdout.flush()

            img_name = f'{time}.jpg'
            img_path = os.path.join(imgs_out_dir, img_name)
            create_xml_file(folder, img_name, img_path, bbs_out_dir, frame_width, frame_height, time, bb_vehicles, bb_walkers)
            cv2.imwrite(img_path, rgb_data)
    print('\nDone.')


if __name__ == "__main__":
    hdf5_file = "/media/alan/Seagate Expansion Drive/Data/CARLA_1920x1280/raw/carla_dataset_1920x1280.hdf5"
    imgs_out_dir = "/media/alan/Seagate Expansion Drive/Data/CARLA_1920x1280/imgs_jpg"
    bbs_out_dir = "/media/alan/Seagate Expansion Drive/Data/CARLA_1920x1280/anns_VOC"
    create_imgs_and_anns(hdf5_file, imgs_out_dir, bbs_out_dir)

