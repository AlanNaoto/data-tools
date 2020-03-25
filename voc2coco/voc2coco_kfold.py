import os
import json
import xml.etree.ElementTree as ET
import glob
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


def save_k_folds_info(train_files_range, test_files_range, fold, json_out_dir):
    train_towns = [1, 2, 3, 4, 5]
    train_towns.pop(fold)
    with open(os.path.join(json_out_dir, f'kfold{fold}_info.txt'), 'w') as file:
        file.write('-Train\n')
        file.write(f'--splits: towns {train_towns}\n')
        file.write(f'--timestamps len: {len(train_xml_files)}\n')
        file.write('-Test\n')
        file.write(f'--split: town {fold+1}\n')
        file.write(f'--timestamps len: {len(test_xml_files)}')


if __name__ == "__main__":
    # Old premise:
    # K-Fold = 5
    # 4 Towns x 1 town [Town 1 is the first test fold, town2 is the second test fold, ...]
    # New premise: shuffle everything, make kfold
    # name structure: fold_train_x.json and fold_test_x.json where x represents the fold index
    xml_VOC_dir = "/media/aissrtx2060/Naotop_1TB1/data/CARLA_1920x1280/anns_VOC"
    json_out_dir = "/media/aissrtx2060/Naotop_1TB1/data/CARLA_1920x1280/anns_coco/kfold_shuffled"
    k_fold = 3

    xml_files = [os.path.join(xml_VOC_dir, x) for x in os.listdir(xml_VOC_dir)]
    random.shuffle(xml_files)

    total_frames = len(xml_files)
    train_split_len = int(len(xml_files) * (k_fold-1) / k_fold)
    test_split_len = int(len(xml_files) / k_fold)
    # Begin k-fold splitting
    for fold in range(k_fold):
        print(f'Working on kfold {fold+1}...')

        # Getting range of files (its okay to get sequential now since we want to include all data)
        # Basically sets the test split from the beginning of the list to the end (test, ...) -> (, test, ...) -> (, , test, ...)
        test_files_range = range(test_split_len*fold, (fold+1)*test_split_len)  # fold+1 corrects the 0-indexing
        train_files_range = set(range(total_frames)) - set(test_files_range)
        # Getting file names
        train_xml_files = [xml_files[x] for x in train_files_range]
        test_xml_files = [xml_files[x] for x in test_files_range]

        # Creates the COCO JSON annotation files
        train_json_file = os.path.join(f'{json_out_dir}', f'train_fold_{fold+1}.json')
        test_json_file = os.path.join(f'{json_out_dir}', f'test_fold_{fold+1}.json')

        convert(train_xml_files, train_json_file)
        convert(test_xml_files, test_json_file)
        #save_k_folds_info(train_files_range, test_files_range, fold, json_out_dir)  # outdated
        print(f"Success: kfold {fold+1}")
_ranñ!
