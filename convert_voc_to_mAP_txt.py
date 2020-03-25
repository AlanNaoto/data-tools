import xml
import xml.etree.ElementTree as ET
import os
import glob


if __name__ == "__main__":
    xml_dir = '/media/alan/Seagate Expansion Drive/Data/CARLA_1920x1280/VOC_bbs'
    out_dir = '/media/alan/Seagate Expansion Drive/Data/CARLA_1920x1280/mAP_gt_bbs'

    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    for file_idx, file in enumerate(xml_files):
        out_file = os.path.basename(file).replace('.xml', '.txt')
        tree = ET.parse(file)
        root = tree.getroot()
        xml_data = {'name': [], 'xmin': [], 'ymin': [], 'xmax': [], 'ymax': []}
        # tags = {}
        for idx, elem in enumerate(tree.iter()):
            # tags[idx] = [elem.tag, elem.text]
            # print('elem.tag', elem.tag)
            if elem.tag in xml_data:
                xml_data[elem.tag].append(elem.text)

        with open(os.path.join(out_dir, out_file), 'w') as f:
            for object in range(len(xml_data['name'])):
                f.write(xml_data['name'][object] + ' ')
                f.write(xml_data['xmin'][object] + ' ')
                f.write(xml_data['ymin'][object] + ' ')
                f.write(xml_data['xmax'][object] + ' ')
                f.write(xml_data['ymax'][object] + '\n')

        print(f'Processed files: {file_idx}/{len(xml_files)}')
