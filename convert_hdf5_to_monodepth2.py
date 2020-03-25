import h5py
import numpy as np
import sys
import cv2
import os


def create_imgs_and_anns(hdf5_file, new_resolution=False, max_depth=None, imgs_output_dir=None, depth_output_dir=None):
    os.makedirs(depth_output_dir, exist_ok=True)
    os.makedirs(imgs_output_dir, exist_ok=True)
    with h5py.File(hdf5_file, 'r') as file:
        for time_idx, time in enumerate(file['timestamps']['timestamps']):
            sys.stdout.write("\r")
            sys.stdout.write('Saving frame {0}/{1}'.format(time_idx+1, len(file['timestamps']['timestamps'])))
            sys.stdout.flush()

            # Get data
            #rgb_data = np.array(file['rgb'][str(time)])
            depth_data = np.array(file['depth'][str(time)])
            if new_resolution:
                #rgb_data = cv2.resize(rgb_data, new_resolution)
                depth_data[depth_data > max_depth] = 0.0
                depth_data = cv2.resize(depth_data, new_resolution)  # numpy resize doesn't interpolates.
            # Save data
            #cv2.imwrite(os.path.join(imgs_output_dir, f'{time}.jpg'), rgb_data)  # png might lead to better results?
            np.save(os.path.join(depth_output_dir, f'{time}.npy'), depth_data)
         	  
    print('\nDone.')


if __name__ == "__main__":
    """
    Creates anns and imgs directories, and saves the depth data and images in each of them
    """
    hdf5_file = "/media/alan/Seagate Expansion Drive/Data/CARLA_1920x1280/raw/carla_dataset_1920x1280.hdf5"
    imgs_output_dir = "/media/alan/Seagate Expansion Drive/Data/CARLA_1024x320/imgs_jpg"
    depth_output_dir = "/media/alan/Seagate Expansion Drive/Data/CARLA_1024x320/depth_npy"
    new_resolution = (1024, 320)
    max_depth = 75.0
    create_imgs_and_anns(hdf5_file, new_resolution, max_depth, imgs_output_dir, depth_output_dir)
