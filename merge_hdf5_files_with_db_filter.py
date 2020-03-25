import sys
import os
import numpy as np
import h5py
import pandas as pd
import sqlite3


class HDF5Saver:
    def __init__(self, sensor_width, sensor_height, file_path_to_save):
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height

        self.file = h5py.File(file_path_to_save, "w")
        # Creating groups to store each type of data
        self.rgb_group = self.file.create_group("rgb")
        self.depth_group = self.file.create_group("depth")
        self.ego_speed_group = self.file.create_group("ego_speed")
        self.bounding_box_group = self.file.create_group("bounding_box")
        self.bb_vehicles_group = self.bounding_box_group.create_group("vehicles")
        self.bb_walkers_group = self.bounding_box_group.create_group("walkers")
        self.timestamp_group = self.file.create_group("timestamps")

        # Storing metadata
        self.file.attrs['sensor_width'] = sensor_width
        self.file.attrs['sensor_height'] = sensor_height
        self.file.attrs['simulation_synchronization_type'] = "syncd"
        self.rgb_group.attrs['channels'] = 'R,G,B'
        self.ego_speed_group.attrs['x,y,z_velocity'] = 'in m/s'
        self.bounding_box_group.attrs['data_description'] = 'Each 4 entries in the same row present one individual actor in the scene.'
        self.bounding_box_group.attrs['bbox_format'] = '[xmin, ymin, xmax, ymax] (top left coords; right bottom coords)' \
                                                       'the vector has been flattened; therefore the data must' \
                                                       'be captured in blocks of 4 elements'
        self.timestamp_group.attrs['time_format'] = "current time in MILISSECONDS since the unix epoch " \
                                                    "(time.time()*1000 in python3)"

    def create_new_filtered_unified_hdf5(self, town_data_hdf5, town_filter_db):
        timestamps_list = []
        for town in range(len(town_data_hdf5)):
            # HDF5
            hdf5_data_path = town_data_hdf5[town]
            db_path = town_filter_db[town]
            # DB file
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT * FROM frames_analysis", conn)

            with h5py.File(hdf5_data_path, 'r') as hdf5_file:
                print(f'working on {hdf5_data_path}...')
                timestamps = list(hdf5_file['timestamps']['timestamps'])
                for time_idx, time in enumerate(timestamps):
                    sys.stdout.write("\r")
                    sys.stdout.write(
                        'Writing new HDF5. Frame {0}/{1}'.format(time_idx, len(timestamps)))
                    sys.stdout.flush()
                    # HDF5
                    rgb_data = np.array(hdf5_file['rgb'][str(time)])
                    bb_vehicles_data = np.array(hdf5_file['bounding_box']['vehicles'][str(time)])
                    bb_walkers_data = np.array(hdf5_file['bounding_box']['walkers'][str(time)])
                    depth_data = np.array(hdf5_file['depth'][str(time)])
                    speed_data = np.array(hdf5_file['ego_speed'][str(time)])
                    # Filter database
                    db_data = df.iloc[time_idx]
                    frame_is_good = db_data['good_frame']

                    if frame_is_good == 1:                        
                        self.record_data(rgb_data, depth_data, bb_vehicles_data, bb_walkers_data, speed_data, str(time))
                        timestamps_list.append(time)

        self.timestamp_group.create_dataset("timestamps", data=np.array(timestamps_list))
        self.file.close()

    def record_data(self, rgb_array, depth_array, bb_vehicle, bb_walker, ego_speed, timestamp):
        self.rgb_group.create_dataset(timestamp, data=rgb_array)
        self.depth_group.create_dataset(timestamp, data=depth_array)
        self.ego_speed_group.create_dataset(timestamp, data=ego_speed)
        self.bb_vehicles_group.create_dataset(timestamp, data=bb_vehicle)
        self.bb_walkers_group.create_dataset(timestamp, data=bb_walker)


if __name__ == "__main__":
    filter_db_files = [os.path.join('D:\\Naoto\\UFPR\\Mestrado\\9_Code\\CARLA_UNREAL\\Dataset\\2_filtered_data\\labeling_tool\\results', x)
                      for x in ['new_town01.db', 'new_town02.db', 'new_town03.db', 'new_town04.db', 'new_town05.db']]
    each_town_hdf5 = [os.path.join('E:\\Data\\CARLA\\2_bbs_filtered', x)
                      for x in ['new_town01.hdf5', 'new_town02.hdf5', 'new_town03.hdf5', 'new_town04.hdf5', 'new_town05.hdf5']]
    unified_hdf5 = 'E:\\Data\\CARLA\\3_unified\\carla_dataset.hdf5'

    HDF5Creator = HDF5Saver(sensor_width=1024, sensor_height=768, file_path_to_save=unified_hdf5)
    HDF5Creator.create_new_filtered_unified_hdf5(town_data_hdf5=each_town_hdf5, town_filter_db=filter_db_files)
