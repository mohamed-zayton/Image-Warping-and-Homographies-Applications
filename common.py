import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# RANSAC parameters
SAMPLE_SIZE = 5 #number of point correspondances for estimation of Homgraphy
SUCCESS_PROB = 0.995 #required probabilty of finding H with all samples being inliners

# 1.1 Getting Correspondences
# Feature of the two image using SIFT
def get_image_sift_feature(img):
    # Check if the image need to be converted to gray
    gray_img = img
    if img.ndim != 2:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # create SIFT object
    sift = cv2.SIFT_create()
    # detect SIFT features in both images
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)

    return keypoints, descriptors


# Get the matched feature between the two image
def get_matches(des1, des2, ratio=0.75):
    # Brute force matcher
    bf = cv2.BFMatcher()
    # match descriptors of both images
    matches = bf.knnMatch(des1, des2, k=2)
    matches_list = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            matches_list.append([m])

    return sorted(matches_list,key=lambda x:x[0].distance) # sort results from the best keypoints to worst

# 1.2 Compute the Homography Parameters
# The point that match the 2 image
def get_matched_pt(kpt, matched_list, points_type = 0, num_pts=50):
    # TODO: check if the function returning the desirable array shape
    """Function to extract the (x, y) points from brute-force knn matcher points

    Args:
        kpt (cv2.KeyPoint array): of shape (n, 1)
        matched_list (2D cv2.DMatch array): of shape (n, 1, 1) represting n set of x, y points
        points_type (number): pass 0 for query points, and 1 for train points
    """
    pts = np.float32([kpt[m[0].queryIdx if points_type == 0 else m[0].trainIdx].pt for m in matched_list[:num_pts]])

    return pts

# Get the homograph matrix using the SVD
def get_homograph_mat(pts_dst, pts_src):
    a_mat = np.zeros((pts_src.shape[0] * 2, 9))

    # Build the A matrix Ah=0
    for i in range(len(pts_src)):
        x = pts_src[i][0]
        y = pts_src[i][1]
        x_dash = pts_dst[i][0]
        y_dash = pts_dst[i][1]
        a_mat[i * 2] += [-x , -y, -1, 0, 0, 0, x * x_dash, y * x_dash, x_dash]
        a_mat[i * 2 + 1] += [0, 0, 0, -x, -y, -1, x * y_dash, y * y_dash, y_dash]

    U, D, V = np.linalg.svd(a_mat, full_matrices=False)
    # Smallest singular value
    homography_mat = (V[-1] / V[-1][-1]).reshape((3, 3))

    return homography_mat


def transform_with_homography(h_mat, points_array):
    # add column of ones so that matrix multiplication with homography matrix is possible
    ones_col = np.ones((points_array.shape[0], 1))
    points_array = np.concatenate((points_array, ones_col), axis=1)
    transformed_points = np.matmul(h_mat, points_array.T)
    epsilon = 1e-7 # very small value to use it during normalization to avoid division by zero
    transformed_points = transformed_points / (transformed_points[2,:].reshape(1,-1) + epsilon)
    transformed_points = transformed_points[0:2,:].T

    return transformed_points

def compute_outliers(h_mat, points_img_a, points_img_b, threshold):
    outliers_count = 0

    # transform the match point in image B to image A using the homography
    points_img_b_hat = transform_with_homography(h_mat, points_img_b)

    # let x, y be coordinate representation of points in image A
    # let x_hat, y_hat be the coordinate representation of transformed points of image B with respect to image A
    x = points_img_a[:, 0]
    y = points_img_a[:, 1]
    x_hat = points_img_b_hat[:, 0]
    y_hat = points_img_b_hat[:, 1]
    euclid_dis = np.sqrt(np.power((x_hat - x), 2) + np.power((y_hat - y), 2)).reshape(-1)
    for dis in euclid_dis:
        if dis > threshold:
            outliers_count += 1

    return outliers_count


def compute_homography_ransac(pts_dst, pts_src, CONFIDENCE_THRESH = 65, OUTLIER_DIS_THRESH = 3):
    num_all_matches =  pts_dst.shape[0]
    min_iterations = int(np.log(1.0 - SUCCESS_PROB)/np.log(1 - 0.5**SAMPLE_SIZE))

    # Let the initial error be large i.e consider all matched points as outliers
    lowest_outliers_count = num_all_matches
    best_h_mat = None

    for i in range(min_iterations):
        rand_ind = np.random.permutation(range(num_all_matches))[:SAMPLE_SIZE]
        h_mat = get_homograph_mat(pts_dst[rand_ind], pts_src[rand_ind])
        outliers_count = compute_outliers(h_mat, pts_dst, pts_src, OUTLIER_DIS_THRESH)
        if outliers_count < lowest_outliers_count:
            best_h_mat = h_mat
            lowest_outliers_count = outliers_count

    best_confidence_obtained = int(100 - (100 * lowest_outliers_count / num_all_matches))
    if best_confidence_obtained < CONFIDENCE_THRESH:
        raise Exception(f'The obtained confidence ratio was {best_confidence_obtained}% which is not higher than {CONFIDENCE_THRESH}%')

    return best_h_mat


def show_image(img, x_axes_visible = False, y_axes_visible = False):
  ax = None
  if len(img.shape) == 3:
    ax = plt.imshow(img[:,:,::-1])
  else:
    ax = plt.imshow(img, cmap='gray', vmin=0, vmax=255)

  ax.axes.get_xaxis().set_visible(x_axes_visible)
  ax.axes.get_yaxis().set_visible(y_axes_visible)
  plt.show()


def point_is_out_of_range(point, dim):
    return point[0] < 0 or point[0] >= dim[0] or point[1] < 0 or point[1] >= dim[1]

def wrap_prespective(img, h, dim):
    target_img = np.zeros((dim[1], dim[0], 3), dtype=np.float64)
    count_mat = np.zeros((dim[1], dim[0]), dtype=np.int32)
    for y in range(len(img)):
        for x in range(len(img[y])):
            curr_coord = [[x], [y], [1]]
            new_coord = np.dot(h, curr_coord)
            new_coord[0][0] /= new_coord[2][0]
            new_coord[1][0] /= new_coord[2][0]
            new_x_points = [int(math.floor(new_coord[0][0])), int(math.ceil(new_coord[0][0]))]
            new_y_points = [int(math.floor(new_coord[1][0])), int(math.ceil(new_coord[1][0]))]
            for new_x in new_x_points:
                for new_y in new_y_points:
                    if not point_is_out_of_range((new_x, new_y), dim):
                        target_img[new_y, new_x, :] += img[y, x, :]
                        count_mat[new_y, new_x] += 1

    h_inv = np.linalg.inv(h)
    for y in range(len(target_img)):
        for x in range(len(target_img[y])):
            if count_mat[y, x] == 0:
                curr_coord = [[x], [y], [1]]
                new_coord = np.dot(h_inv, curr_coord)
                new_coord[0][0] /= new_coord[2][0]
                new_coord[1][0] /= new_coord[2][0]
                new_x_points = [int(math.floor(new_coord[0][0])), int(math.ceil(new_coord[0][0]))]
                new_y_points = [int(math.floor(new_coord[1][0])), int(math.ceil(new_coord[1][0]))]
                weighted_intenisty_sum = np.array([0, 0, 0], dtype=np.float64)
                weights_sum = 0
                for new_x in new_x_points:
                    for new_y in new_y_points:
                        if not point_is_out_of_range((new_x, new_y), (img.shape[1], img.shape[0])):
                            weight = abs(new_x - new_coord[0][0]) * abs(new_y - new_coord[1][0])
                            weighted_intenisty_sum += weight * img[new_y, new_x]
                            weights_sum += weight
                
                target_img[y, x] += weighted_intenisty_sum / (weights_sum if weights_sum != 0 else 1)


            else:
                target_img[y, x, 0] = int(np.round(target_img[y, x, 0] / count_mat[y, x]))
                target_img[y, x, 1] = int(np.round(target_img[y, x, 1] / count_mat[y, x]))
                target_img[y, x, 2] = int(np.round(target_img[y, x, 2] / count_mat[y, x]))

    return target_img.astype(np.uint8)

