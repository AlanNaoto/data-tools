import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.enable_eager_execution()

from waymo_open_dataset.utils import frame_utils


def rgba(r):
    """Generates a color based on range.

    Args:
      r: the range value of a given point.
    Returns:
      The color for a given range
    """
    c = plt.get_cmap('jet')((r % 20.0) / 20.0)
    c = list(c)
    c[-1] = 0.5  # alpha
    return c


def plot_image(camera_image):
    """Plot a cmaera image."""
    plt.figure(figsize=(20, 12))
    plt.imshow(tf.image.decode_jpeg(camera_image.image))
    plt.grid("off")


def plot_points_on_image(projected_points, camera_image, rgba_func,
                         point_size=5.0):
    """Plots points on a camera image.

    Args:
      projected_points: [N, 3] numpy array. The inner dims are
        [camera_x, camera_y, range].
      camera_image: jpeg encoded camera image.
      rgba_func: a function that generates a color from a range value.
      point_size: the point size.

    """
    plot_image(camera_image)

    xs = []
    ys = []
    colors = []

    for point in projected_points:
        xs.append(point[0])  # width, col
        ys.append(point[1])  # height, row
        colors.append(rgba_func(point[2]))
    plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")


def parse_lidar_data(frame):
    (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(
        frame)
    points, cp_points = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)
    # points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(
    #     frame,
    #     range_images,
    #     camera_projections,
    #     range_image_top_pose,
    #     ri_index=1)

    # 3d points in vehicle frame.
    points_all = np.concatenate(points, axis=0)
    # points_all_ri2 = np.concatenate(points_ri2, axis=0)
    # camera projection corresponding to each point.
    cp_points_all = np.concatenate(cp_points, axis=0)
    # cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

    images = sorted(frame.images, key=lambda i: i.name)
    cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
    # cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

    # The distance between lidar points and vehicle frame origin.
    points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

    mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)
    cp_points_all_tensor = tf.cast(tf.gather_nd(cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
    points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

    projected_points_all_from_raw_data = tf.concat(
        [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()

    #plot_points_on_image(projected_points_all_from_raw_data, images[0], rgba, point_size=5.0)
    # plt.show()
    return projected_points_all_from_raw_data
