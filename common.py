import cv2
import numpy as np

# 1.1 Getting Correspondences
# Feature of the two image using SIFT
def get_image_sift_feature(img_path1, image_path2):
    # read the images
    img1 = cv2.imread('/content/cv_cover.jpg')
    img2 = cv2.imread('/content/im.jpg')
    # convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    # detect SIFT features in both images
    keypoints_1, descriptors_1 = sift.detectAndCompute(gray1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(gray2, None)
    return keypoints_1, descriptors_1, keypoints_2, descriptors_2

# Get the matched feature between the two image
def get_matches(kp1, des1, kp2, des2, num_pts=50, ratio=0.5, knn_num=2):
    # Brute force matcher
    bf = cv2.BFMatcher()

    # match descriptors of both images
    matches = bf.knnMatch(des1, des2, k=knn_num)
    matches_list = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            matches_list.append([m])
    return matches_list

#1.2 Compute the Homography Parameters
# The point that match the 2 image
def get_matched_pt(kpt1, kpt2, matched_list):
    pts1 = np.float32(
        [kpt1[m[0].queryIdx].pt for m in matched_list]).reshape(-1, 1, 2)
    pts2 = np.float32(
        [kpt2[m[0].trainIdx].pt for m in matched_list]).reshape(-1, 1, 2)
    return pts1, pts2

# Get the homograph matrix using the SVD
def get_homograph_mat(pts1, pts2):
    a_mat = np.zeros((pts1.shape[0] * 2, 9))
    ind = 0
    # Build the A matrix Ah=0
    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        x = pt1[0][0]
        y = pt1[0][1]
        xd = pt2[0][0]
        yd = pt2[0][1]
        x_xd = x * xd
        y_xd = y * xd
        x_yd = x * yd
        y_yd = y * yd
        a_mat[ind][0], a_mat[ind][1], a_mat[ind][2], a_mat[ind][6], a_mat[ind][7], a_mat[ind][8] = - \
            x, -y, -1, x_xd, y_xd, xd
        a_mat[ind+1][3], a_mat[ind+1][4], a_mat[ind+1][5], a_mat[ind +
                                                                 1][6], a_mat[ind+1][7], a_mat[ind+1][8] = -x, -y, -1, x_yd, y_yd, yd
        ind += 2
    U, D, V = np.linalg.svd(a_mat, full_matrices=False)
    # Smallest singular value
    homography_mat = (V[-1] / V[-1][-1]).reshape((3, 3))
    return homography_mat
