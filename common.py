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
