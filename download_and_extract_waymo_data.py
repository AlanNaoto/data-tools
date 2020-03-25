import os


def download_and_extract_data(split, dataset_version, out_dir):
    if split == 'training':
        num_segs = 32
    elif split == 'validation':
        num_segs = 8

    os.makedirs(out_dir, exist_ok=True)

    for seg_id in range(num_segs):
        print(f"Downloading {seg_id}/{num_segs} tar file.")
        tar_file = f'{split}_{seg_id:.4}.tar'
        tar_url = f'gs://waymo_open_dataset_v_{dataset_version}/{split}/{tar_file}'
        flag = os.system('gsutil cp ' + tar_url + ' ' + out_dir)
        assert flag == 0, 'Failed to download segment %d. Make sure gsutil is installed' % seg_id
        os.system(f"cd {out_dir}; tar xf {tar_file}")
        break
        # tfrecords = sorted(glob.glob('%s/*.tfrecord'%out_dir))
        # for record in tfrecords:
        #     print("Clip %d done"%clip_id)
        #     clip_id += 1
        #     os.remove(record)
        # print("Segment %d done"%seg_id)


if __name__ == "__main__":
    out_dir = ""
    split = "training"  # "training" or "validation"
    dataset_version = '1_2_0'
    download_and_extract_data(split, dataset_version, out_dir)
