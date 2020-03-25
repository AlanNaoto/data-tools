import os

img_dir = "val"

img_files = [os.path.join(img_dir, x) for x in os.listdir(img_dir)]
for img in img_files:
    new_name = img.replace(".jpg", ".jpeg")
    os.rename(img, new_name)

