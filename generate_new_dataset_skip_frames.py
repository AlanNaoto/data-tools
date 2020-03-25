import os
import shutil


def generate_subsampled_dir(in_dir, frame_interval_for_data_save, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    input_files = sorted(os.listdir(in_dir))
    for file_idx, in_file in enumerate(input_files):
        print(f'{file_idx}/{len(input_files)}')
        if file_idx % frame_interval_for_data_save == 0:
            shutil.copyfile(os.path.join(in_dir, in_file), os.path.join(out_dir, in_file))


if __name__ == "__main__":
    root_in_dir = "/home/alan/workspace/Mestrado/dataset/WAYMO_all"
    dirs_to_reference = ['anns_bb_mAP', 'imgs_jpg']
    out_dir = "/home/alan/workspace/Mestrado/dataset/WAYMO_skip10"
    frame_interval_for_data_save = 10  # if =1, then copies everything
    
    for files_dir in dirs_to_reference:
        generate_subsampled_dir(os.path.join(root_in_dir, files_dir), frame_interval_for_data_save, os.path.join(out_dir, files_dir))

