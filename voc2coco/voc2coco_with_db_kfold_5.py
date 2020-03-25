import os
import json
import xml.etree.ElementTree as ET
import glob
import sqlite3
import random
import pandas as pd


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        raise ValueError("Filename %s is supposed to be an integer." % (filename))


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.
    
    Arguments:
        xml_files {list} -- A list of xml file paths.
    
    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file):
    START_BOUNDING_BOX_ID = 1
    PRE_DEFINE_CATEGORIES = {"vehicle": 1, "pedestrian": 2}
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))
        ## The filename must be a number
        image_id = get_filename_as_int(filename)
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)
        image = {
            "file_name": filename,
            "height": height,
            "width": width,
            "id": image_id,
        }
        json_dict["images"].append(image)
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json_fp = open(json_file, "w")
    json_str = json.dumps(json_dict)
    json_fp.write(json_str)
    json_fp.close()


def get_db_data(db_files):
    town_timestamps = {}
    for town in range(len(db_files)):
        conn = sqlite3.connect(db_files[town])
        df = pd.read_sql_query("SELECT * FROM frames_analysis WHERE good_frame=1", conn)
        timestamps = list(df['timestamps'])
        town_timestamps[f'town{town+1}'] = timestamps
        conn.close()
    return town_timestamps


def save_k_folds_info(test_split, train_splits, train_xml_files, test_xml_files):
    with open(os.path.join('data', 'coco', f'kfold{test_split}_info.txt'), 'w') as file:
        file.write('-Train\n')
        file.write(f'--splits: towns {train_splits}\n')
        file.write(f'--timestamps len: {len(train_xml_files)}\n')
        file.write('-Test\n')
        file.write(f'--split: town {test_split}\n')
        file.write(f'--timestamps len: {len(test_xml_files)}')


if __name__ == "__main__":
    # K-Fold = 5
    # 4 Towns x 1 town
    # Get first and last time stamps of each FILTERED town to know which timestamps (frames) are the start/stop points
    # This way, we will have a list of timestamps for EACH TOWN -> read from DB files should be good.
    # Scramble each list of timestamps
    # Divide into 4 lists against 1 list for train x test with 5-folds
    # name structure: fold_train_x.json and fold_test_x.json where x represents the fold index
    db_dir = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/CARLA_UNREAL/Dataset/2_filtered_data/labeling_tool/results"
    xml_dir = "data/VOC/Annotations"
    db_files = glob.glob(os.path.join(db_dir, "*.db"))
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    db_data = get_db_data(db_files)

    # Begin k-fold splitting
    k_fold = 5
    for test_split in range(1, k_fold+1):
        print(f'Working on kfold {test_split}...')
        # Getting timestamps
        test_timestamps = db_data[f'town{test_split}']
        train_splits = list(range(1, k_fold+1))
        train_splits.remove(test_split)
        train_timestamps = []
        for train_split in train_splits:
            train_timestamps.extend(db_data[f'town{train_split}'])

        # Shuffling timestamps
        random.shuffle(train_timestamps)
        random.shuffle(test_timestamps)

        # Getting file names
        train_xml_files = [os.path.join('data', 'VOC', 'Annotations', str(x) + '.xml') for x in train_timestamps]
        test_xml_files = [os.path.join('data', 'VOC', 'Annotations', str(x) + '.xml') for x in test_timestamps]
        train_json_file = f'data/coco/train_fold_{test_split}.json'
        test_json_file = f'data/coco/test_fold_{test_split}.json'

        # Creates the COCO JSON annotation files
        convert(train_xml_files, train_json_file)
        convert(test_xml_files, test_json_file)
        save_k_folds_info(test_split, train_splits, train_xml_files, test_xml_files)
        print(f"Success: kfold {test_split}")

