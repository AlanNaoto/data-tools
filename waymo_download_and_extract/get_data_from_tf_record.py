import cv2
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from get_lidar_data import parse_lidar_data


def get_bbox(frame_data, camera_image_data):
    """Show a camera image and the given camera labels."""
    object_decodification = {0: 'unkown', 1: 'vehicle', 2: 'pedestrian', 3: 'sign', 4: 'cyclist'}
    bbox = {}
    valid_bb_data = True
    if not frame_data.camera_labels:
        valid_bb_data = False
    # Draw the camera labels.
    for camera_labels in frame_data.camera_labels:
        # Ignore camera labels from other views (i.e. I want Front but it also gives left, right, front left, ...)
        if camera_labels.name != camera_image_data.name:
            continue
        # Iterate over the individual labels
        for label in camera_labels.labels:
            if label.detection_difficulty_level == 0:
                difficulty = "easy"
            elif label.detection_difficulty_level == 2:
                difficulty = "hard"

            if label.tracking_difficulty_level == 0:
                tracking_level = "easy"
            elif label.tracking_difficulty_level == 2:
                tracking_level = 'hard'

            object_class = object_decodification[label.type]
            # I'm not saving the other labels so that it matches my CARLA dataset
            if object_class not in bbox and (object_class == "vehicle" or object_class == "pedestrian"):
                bbox[object_class] = []
            
            if (object_class == "vehicle" or object_class == "pedestrian"):
                # Get BB
                xmin = int(label.box.center_x - 0.5 * label.box.length)
                ymin = int(label.box.center_y - 0.5 * label.box.width)
                xmax = int(xmin + label.box.length)
                ymax = int(ymin + label.box.width)
                bbox[object_class].append([xmin, ymin, xmax, ymax, difficulty, tracking_level])
    return bbox, valid_bb_data


def get_img_and_bbox(frame_data):
    bbox, valid_bb_data, img_array = {}, {}, {}
    for camera_image_data in frame_data.images:
        camera_name = open_dataset.CameraName.Name.Name(camera_image_data.name).lower()
        bbox[camera_name], valid_bb_data[camera_name] = get_bbox(frame_data, camera_image_data)
        img_array[camera_name] = cv2.cvtColor(tf.image.decode_jpeg(camera_image_data.image).numpy(), cv2.COLOR_BGR2RGB)
    return img_array, bbox, valid_bb_data


def paint_bb_on_img(img, bbox):
    color = {'': (0, 255, 0), "difficult": (0, 0, 255), "tracking_hard": (255, 0, 0)}
    for object_class in bbox:
        for object_count in range(len(bbox[object_class])):
            box = bbox[object_class][object_count]
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color[box[4]], 1)
    return img


def get_lidar_data(frame):
    projected_points_all_from_raw_data = parse_lidar_data(frame)
    return projected_points_all_from_raw_data
