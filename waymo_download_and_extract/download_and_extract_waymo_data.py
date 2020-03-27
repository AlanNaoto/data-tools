import sqlite3
import os
import tarfile
import tempfile
import sys
from tqdm import tqdm
import numpy as np
import cv2
from get_data_from_tf_record import get_img_and_bbox, get_lidar_data
import tensorflow as tf
tf.enable_eager_execution()
from waymo_open_dataset import dataset_pb2 as open_dataset


class DatabaseManager:
    def __init__(self, db_filename):
        self.conn = sqlite3.connect(db_filename)
        self.c = self.conn.cursor()
        try:
            self.c.execute('''CREATE TABLE waymo_metadata 
                            (unix_microseconds text, 
                            time_of_day text, 
                            location text, 
                            weather text, 
                            start_of_sequence binary, 
                            end_of_sequence binary)''')
        except sqlite3.OperationalError:
            print("Table already exists. Not creating a new one")

    def add_data_to_database(self, frame_data):
        frame_name = str(frame_data.timestamp_micros)
        time_of_day = str(frame_data.context.stats.time_of_day)
        location = str(frame_data.context.stats.location)
        weather = str(frame_data.context.stats.weather)
        data = [frame_name, time_of_day, location, weather, 0, 0]
        self.c.execute('INSERT INTO waymo_metadata VALUES (?, ?, ?, ?, ?, ?)', data)
        self.conn.commit()

    def add_start_or_finish(self, frame_name, frame_idx):
        start = 0
        end = 0
        if frame_idx == 0:
            start = 1
        else:
            end = 1
        self.c.execute('''UPDATE waymo_metadata SET start_of_sequence = ?, end_of_sequence = ? WHERE unix_microseconds = ?''',
                       (start, end, frame_name))
        self.conn.commit()


class DatasetCreator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.rgb_dir = os.path.join(output_dir, 'imgs_jpg')
        self.bb_dir = os.path.join(output_dir, 'anns_custom')
        self.lidar_dir = os.path.join(output_dir, 'depth_npy')
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.bb_dir, exist_ok=True)
        os.makedirs(self.lidar_dir, exist_ok=True)

    def save_rgb_data(self, img_array, frame_name):
        for camera_name in img_array:
            cv2.imwrite(os.path.join(self.rgb_dir, frame_name + "_" + camera_name) + ".jpg", img_array[camera_name])

    def save_bb_data(self, bb_data, frame_name):
        # BB Format: object_class xmin ymin xmax ymax difficulty_level tracking_level
        for camera_name in bb_data:
            with open(os.path.join(self.bb_dir, frame_name + "_" + camera_name) + ".txt", 'w') as f:
                if bb_data[camera_name]:
                    for object_class in bb_data[camera_name]:
                        for object_data in bb_data[camera_name][object_class]:
                            object_data = [str(x) for x in object_data]
                            f.write(object_class + ' ' + object_data[0] + ' ' + object_data[1] + ' ' + object_data[2] + ' '
                                    + object_data[3] + ' ' + object_data[4] + ' ' + object_data[5] + '\n')

    def save_lidar_data(self, lidar_data, frame_name):
        np.save(os.path.join(self.lidar_dir, frame_name) + ".npy", lidar_data)


class TarManager:
    def __init__(self, tar_file_path, tmp_top_dir):
        # Creates tmp dir to store TFRecord files
        self.tmp_dir = tempfile.TemporaryDirectory(dir=tmp_top_dir)
        tmp_dir_path = self.tmp_dir.name
        with tarfile.open(tar_file_path, 'r') as f:
            members = f.getmembers()
            members = [x for x in members if x.name != 'LICENSE']
            print('\nExtracting TFRecord files from TAR... it may take a while')
            for member in tqdm(iterable=members, total=len(members)):
                f.extract(path=tmp_dir_path, member=member)
        self.tf_record_paths = [os.path.join(tmp_dir_path, x) for x in os.listdir(tmp_dir_path)]

    def clean_tmp_dir(self):
        self.tmp_dir.cleanup()


def create_filtered_data(tar_file_path, output_dir):
    DatasetFiltered = DatasetCreator(output_dir)
    Database = DatabaseManager(os.path.join(output_dir, 'annotation_metadata.db'))
    invalid_bb_frames = {}

    # Managing tar file and creating temporary TFRecord files on storage at a time
    TarObject = TarManager(tar_file_path, output_dir)
    tf_record_paths = TarObject.tf_record_paths

    # Cycle through each TFRecord file and begin data extraction
    for tf_idx, tf_filename in enumerate(tf_record_paths):
        print(f'Working on TFRecord {tf_idx+1} of {len(tf_record_paths)}')
        dataset = tf.data.TFRecordDataset(tf_filename, compression_type="")
        for frame_idx, data in enumerate(dataset):
            sys.stdout.write("\r")
            sys.stdout.write(f'Frame {frame_idx}/200 (probably 200)')
            sys.stdout.flush()

            # Overall frame data (e.g. data from a timestamp)
            frame_data = open_dataset.Frame()
            frame_data.ParseFromString(bytearray(data.numpy()))

            # Collect the relevant data
            frame_name = str(frame_data.timestamp_micros)
            img, bbox, valid_bb_data = get_img_and_bbox(frame_data)
            lidar_data = get_lidar_data(frame_data)

            # This assertion is made because there are frames which contains no camera FRONT, SIDE, ..., in which
            # the bounding box information is saved
            if not valid_bb_data:
                if os.path.basename(tf_filename) not in invalid_bb_frames:
                    invalid_bb_frames[os.path.basename(tf_filename)] = 0
                invalid_bb_frames[os.path.basename(tf_filename)] += 1
                continue

            # Save stuff
            DatasetFiltered.save_rgb_data(img, frame_name)
            DatasetFiltered.save_bb_data(bbox, frame_name)
            DatasetFiltered.save_lidar_data(lidar_data, frame_name)
            # Collect frame metadata
            Database.add_data_to_database(frame_data)
            if frame_idx == 0:
                Database.add_start_or_finish(frame_name, frame_idx)
        Database.add_start_or_finish(frame_name, frame_idx)
        break  # FIXME ERASEME
    TarObject.clean_tmp_dir()


def download_and_extract_data(split, dataset_version, out_dir):
    if split == 'training':
        len_tars = 32
    elif split == 'validation':
        len_tars = 8
    os.makedirs(out_dir, exist_ok=True)

    for tar_id in range(len_tars):
        print(f"Downloading {tar_id}/{len_tars} tar file.")
        tar_filename = f'{split}_{tar_id:04d}.tar'
        tar_url = f'gs://waymo_open_dataset_v_{dataset_version}/{split}/{tar_filename}'
        os.system('gsutil cp ' + tar_url + ' ' + out_dir)

        tar_file_path = os.path.join(out_dir, tar_filename)
        print(f'Working on tar {tar_id}/{len_tars} {split}')
        create_filtered_data(tar_file_path, os.path.join(out_dir, split))
        os.remove(tar_file_path)
        break # FIXME ERASEME


if __name__ == "__main__":
    out_dir = "temporario"
    split = "training"  # "training" or "validation"
    dataset_version = '1_2_0'
    download_and_extract_data(split, dataset_version, out_dir)
