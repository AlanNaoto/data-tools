import os
import cv2


def resize_imgs(imgs_in_dir, imgs_out_dir, new_resolution):
    imgs_paths = [os.path.join(imgs_in_dir, x) for x in os.listdir(imgs_in_dir)]
    for img_idx, img_path in enumerate(imgs_paths):
        img = cv2.imread(img_path)
        img = cv2.resize(img, new_resolution)
        cv2.imwrite(os.path.join(imgs_out_dir, os.path.basename(img_path)), img)
        print(f"{img_idx}/{len(imgs_paths)}")

if __name__ == "__main__":
    imgs_in_dir = "/media/aissrtx2060/Seagate Expansion Drive/Data/Waymo/transformed_data/imgs_jpg/train"
    imgs_out_dir = "/media/aissrtx2060/Seagate Expansion Drive/Data/Waymo/transformed_data/imgs_jpg_1024x320"
    new_resolution = (1024, 320)
    resize_imgs(imgs_in_dir, imgs_out_dir, new_resolution)
