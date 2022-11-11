import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

# 1.1 Getting Correspondences
# Feature of the two image using SIFT


def get_image_sift_feature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # create SIFT object
    sift = cv2.SIFT_create()
    # detect SIFT features in both images
    keypoints, descriptors = sift.detectAndCompute(gray, None)
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

def get_matched_pt(kpt, matched_list, num_pts=50):
    pts = np.float32([kpt[m[0].queryIdx].pt for m in matched_list[:num_pts]])
    pts = pts.reshape(-1, 1, 2)
    return pts

# Get the homograph matrix using the SVD
def get_homograph_mat(pts_src, pts_dst):
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
    
def compute_outliers(h_mat, points_img_a, points_img_b, threshold=3):
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


def compute_homography_ransac(matches_a, matches_b, CONFIDENCE_THRESH = 65):
    num_all_matches =  matches_a.shape[0]
    # RANSAC parameters
    SAMPLE_SIZE = 5 #number of point correspondances for estimation of Homgraphy
    SUCCESS_PROB = 0.995 #required probabilty of finding H with all samples being inliners 
    min_iterations = int(np.log(1.0 - SUCCESS_PROB)/np.log(1 - 0.5**SAMPLE_SIZE))
    
    # Let the initial error be large i.e consider all matched points as outliers
    lowest_outliers_count = num_all_matches
    best_h_mat = None

    for i in range(min_iterations):
        rand_ind = np.random.permutation(range(num_all_matches))[:SAMPLE_SIZE]
        h_mat = cv2.findHomography(matches_a[rand_ind], matches_b[rand_ind])
        outliers_count = compute_outliers(h_mat, matches_a, matches_b)
        if outliers_count < lowest_outliers_count:
            best_h_mat = h_mat
            lowest_outliers_count = outliers_count
            
    best_confidence_obtained = int(100 - (100 * lowest_outliers_count / num_all_matches))
    if best_confidence_obtained < CONFIDENCE_THRESH:
        raise('Coudn\'t obtain confidence ratio higher than the CONFIDENCE THRESH')

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


def wrap_prespective(img, h, dim):
    target_img = np.zeros((dim[1], dim[0], 3), dtype=np.uint8)
    count_mat = np.zeros((dim[1], dim[0]), dtype=np.int32)
    for y in range(len(img)):
        for x in range(len(img[y])):
            curr_coord = [[x], [y], [1]]
            new_coord = np.dot(h, curr_coord)
            new_coord[0][0] /= new_coord[2][0]
            new_coord[1][0] /= new_coord[2][0]
            upper_x = int(math.ceil(new_coord[0][0]))
            lower_x = int(math.floor(new_coord[0][0]))
            upper_y = int(math.ceil(new_coord[1][0]))
            lower_y = int(math.floor(new_coord[1][0]))
            if lower_x >= 0 and lower_x < dim[0] and lower_y >= 0 and lower_y < dim[1]:
                target_img[lower_y, lower_x, :] += img[y, x, :]
                count_mat[lower_y, lower_x] += 1

            if lower_x >= 0 and lower_x < dim[0] and upper_y >= 0 and upper_y < dim[1]:
                target_img[upper_y, lower_x, :] += img[y, x, :]
                count_mat[upper_y, lower_x] += 1

            if upper_x >= 0 and upper_x < dim[0] and lower_y >= 0 and lower_y < dim[1]:
                target_img[lower_y, upper_x, :] += img[y, x, :]
                count_mat[lower_y, upper_x] += 1

            if upper_x >= 0 and upper_x < dim[0] and upper_y >= 0 and upper_y < dim[1]:
                target_img[upper_y, upper_x, :] += img[y, x, :]
                count_mat[upper_y, upper_x] += 1
            
    
    for y in range(len(target_img)):
        for x in range(len(target_img[y])):
            if count_mat[y, x] == 0:
                continue
            target_img[y, x, 0] = int(np.round(target_img[y, x, 0] / count_mat[y, x]))
            target_img[y, x, 1] = int(np.round(target_img[y, x, 1] / count_mat[y, x]))
            target_img[y, x, 2] = int(np.round(target_img[y, x, 2] / count_mat[y, x]))

    return target_img