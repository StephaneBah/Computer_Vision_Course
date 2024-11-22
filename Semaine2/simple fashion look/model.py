import numpy as np
import cv2

def kmeans(image, k=3, alpha=0.5):
    image = np.array(image)
    pixel_vals = image.reshape((-1,3))
    pixel_vals = np.float32(pixel_vals)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    #  Attribuer des couleurs données à nos clusters
    cluster_colors = np.random.randint(0, 255, size=(k, 3), dtype=np.uint8)
    segmented_data = cluster_colors[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))

    return segmented_image

def mean_shift(image, spatial_radius=5, color_radius=60, max_iter=4):
    image = np.array(image)
    mean_shift_result = cv2.pyrMeanShiftFiltering(image, sp=spatial_radius, sr=color_radius, maxLevel=max_iter)

    flat_image = mean_shift_result.reshape((-1, 3))

    unique_colors, labels = np.unique(flat_image, axis=0, return_inverse=True)

    cluster_colors = np.random.randint(0, 255, size=(len(unique_colors), 3), dtype=np.uint8)
    segmented_image = cluster_colors[labels].reshape(image.shape)

    return segmented_image
    