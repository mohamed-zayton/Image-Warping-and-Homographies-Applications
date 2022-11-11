import cv2
import numpy as np

# 1.1 Getting Correspondences
# Feature of the two image using SIFT


def get_image_sift_feature(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
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
    pts = np.float32(
        [kpt[m[0].queryIdx].pt for m in matched_list[:num_pts]]).reshape(-1, 1, 2)
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