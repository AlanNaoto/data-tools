import sqlite3
import os
import tarfile
import tempfile
import sys
from tqdm import tqdm
import tensorflow as tf
#tf.enable_eager_execution()
from waymo_open_dataset import dataset_pb2 as open_dataset


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

    def add_start_or_finish(self, frame_data, frame_idx):
        start = 0
        end = 0
        if frame_idx == 0:
            start = 1
        else:
            end = 1
        frame_name = str(frame_data.timestamp_micros)
        self.c.execute('''UPDATE waymo_metadata SET start_of_sequence = ?, end_of_sequence = ? WHERE unix_microseconds = ?''',
                       (start, end, frame_name))
        self.conn.commit()



def create_annotation_metadata(tar_file_paths, db_filename, output_dir):
    Database = DatabaseManager(os.path.join(output_dir, db_filename))
    # Managing tar file and creating temporary TFRecord files on storage at a time
    for tar_idx, tar_file_path in enumerate(tar_file_paths):
        print(f'Working on tar {os.path.basename(tar_file_path)} {tar_idx}/{len(tar_file_paths)}')
        TarObject = TarManager(tar_file_path, output_dir)
        tf_record_paths = TarObject.tf_record_paths

        # Each TFRecord file is a video sequence. Creating TXT to save this detail
        with open("sequences_per_tar.txt", 'a') as f:
            f.write(f'{os.path.basename(tar_file_path)}: {len(tf_record_paths)} sequences')

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
                Database.add_data_to_database(frame_data)
                if frame_idx == 0:
                    Database.add_start_or_finish(frame_data, frame_idx)
            Database.add_start_or_finish(frame_data, frame_idx)
        TarObject.clean_tmp_dir()
        with open(os.path.join(output_dir, 'processed_tar_files.txt'), 'a') as f:
             f.write(tar_file_path + '\n')

    Database.conn.close()


def exclude_tars_from_process(processed_tar_files_txt, tar_files):
    if os.path.exists(processed_tar_files_txt):
        with open(processed_tar_files_txt, 'r') as f:
            processed_tars = f.read().splitlines()
        tar_files = [x for x in tar_files if x not in processed_tars]
    else:
        print('no already processed txt tar files found. doing everything.')
    return tar_files


if __name__ == "__main__":
    """
    Ensure you have at least ~30 GB available - this space is needed to temporarily store the contents of each TAR file
    """
    # Input
    tar_dir_path = "/media/alan/Seagate Expansion Drive/Data/Waymo/raw"
    output_dir = '/media/alan/Seagate Expansion Drive/Data/Waymo/transformed_data/metadata'
    database_file = 'annotation_metadata.db'
    already_processed_tars = {'training': '', 'validation': ''}

    splits_to_work = ['training', 'validation']
    tar_files = [os.path.join(tar_dir_path, x) for x in os.listdir(tar_dir_path)]
    for split in splits_to_work:
        tar_files_split = [x for x in tar_files if split in x]
        tar_files_split = exclude_tars_from_process(already_processed_tars[split], tar_files_split)
        create_annotation_metadata(tar_files_split, database_file, output_dir)

