import sys
import os
import numpy as np
import h5py
import pandas as pd


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

    def create_new_filtered_unified_hdf5(self, town_data_hdf5):
        timestamps_list = []
        for town in range(len(town_data_hdf5)):
            # HDF5
            hdf5_data_path = town_data_hdf5[town]

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
    each_town_hdf5 = [os.path.join('/media/alan/Seagate Expansion Drive/Data/CARLA/1920x1280', x)
                      for x in ['Town01.hdf5', 'Town02.hdf5', 'Town03.hdf5', 'Town04.hdf5', 'Town05.hdf5']]
    unified_hdf5 = '/media/alan/Naotop_1TB/data/CARLA_1920_1280/carla_dataset_1920_1280.hdf5'

    HDF5Creator = HDF5Saver(sensor_width=1920, sensor_height=1280, file_path_to_save=unified_hdf5)
    HDF5Creator.create_new_filtered_unified_hdf5(town_data_hdf5=each_town_hdf5)
