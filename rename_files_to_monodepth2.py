import os
from shutil import copyfile


def rename_files(input_dir, output_dir, file_ext=""):
    input_files = sorted(os.listdir(input_dir))
    for frame_idx, frame_file in enumerate(input_files):
        input_path = os.path.join(input_dir, frame_file) 
        output_path = os.path.join(output_dir, f"{frame_idx:05d}" + file_ext)  # 5 Trailing zeros
        os.rename(f"{input_path}", f"{output_path}")
        #copyfile(f"{input_path}", f"{output_path}")


if __name__ == "__main__":
    # Input
    imgs_in_dir = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/datasets/CARLA/CARLA_1024x320/imgs_jpg"
    imgs_out_dir = imgs_in_dir
    img_ext = ".jpg"
    depth_in_dir = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/datasets/CARLA/CARLA_1024x320/depth_npy"
    depth_out_dir = depth_in_dir
    depth_ext = ".npy"
    
    rename_files(imgs_in_dir, imgs_out_dir, img_ext)
    rename_files(depth_in_dir, depth_out_dir, depth_ext)

