import sys
import os
import xml.etree.ElementTree as ET
import cv2


def create_xml_file(imgs_input_dir, bb_output_dir, img_name, img_path, frame_width, frame_height, xml_filename, bb_vehicles, bb_pedestrians):
    def create_class_entry(annotation, bb_data=None, object=None):
        # bb_data = [[xmin, ymin, xmax, ymax, difficult], [xmin, ymin, xmax, ymax, difficult], ...]
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
            difficult.text = str(bb_data[4])

        return annotation

    # Creating general structure
    annotation = ET.Element('annotation')

    folder = ET.SubElement(annotation, 'folder').text = imgs_input_dir
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
    for vehicle in bb_vehicles:
        annotation = create_class_entry(annotation, vehicle, object='vehicle')
    for walker in bb_pedestrians:
        annotation = create_class_entry(annotation, walker, object='pedestrian')
    if bb_vehicles is None and bb_pedestrians is None:
        annotation = create_class_entry(annotation)

    # Creating the file
    tree = ET.ElementTree(annotation)
    tree.write(os.path.join(bb_output_dir, xml_filename))


if __name__ == "__main__":
    # User input
    bb_input_dir = "/media/aiss-v100/Naotop_1TB/data/WAYMO_v120/anns_custom"
    imgs_input_dir = "/media/aiss-v100/Naotop_1TB/data/WAYMO_v120/imgs_jpg"
    bb_output_dir = "/media/aiss-v100/Naotop_1TB/data/WAYMO_v120/anns_voc"

    # Finding the data to be treated
    os.makedirs(bb_output_dir, exist_ok=True)
    bb_input_files = os.listdir(bb_input_dir)
    bb_input_files_path = [os.path.join(bb_input_dir, x) for x in bb_input_files]

    # Creating one XML at a time
    for bb_file_idx in range(len(bb_input_files)):
        bb_vehicles = []
        bb_pedestrians = []
        with open(bb_input_files_path[bb_file_idx], 'r') as f:
            sys.stdout.write("\r")
            sys.stdout.write(f'Saving frame {bb_file_idx}/{len(bb_input_files)}')
            sys.stdout.flush()

            # Parse bounding box data
            file_data = f.readlines()
            file_data = [x.replace('\n', "").split() for x in file_data]
            # in x[1:-1], 1 is done to discard the name -1 to simplify the difficult label of waymo
            [bb_vehicles.append(x[1:-1]) for x in file_data if x[0] == "vehicle"]
            [bb_pedestrians.append(x[1:-1]) for x in file_data if x[0] == "pedestrian"]

            # Adjust metadata for xml
            frame_name = bb_input_files[bb_file_idx].replace(".txt", "")
            img_name = frame_name + ".jpg"
            xml_filename = frame_name + ".xml"
            img_path = os.path.join(os.path.join(imgs_input_dir, img_name))
            # Img dimensions
            img = cv2.imread(img_path)
            frame_width = img.shape[1]
            frame_height = img.shape[0]
            # Finally create the xml files
            create_xml_file(imgs_input_dir, bb_output_dir, img_name, img_path, frame_width, frame_height, xml_filename, bb_vehicles, bb_pedestrians)

