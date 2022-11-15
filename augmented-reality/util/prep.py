# Calculate homography from cover to one book frame

import numpy as np
import cv2

from common import *

def cover_to_book_homography(kp_cover, des_cover, book_gray,ransac=False)->np.ndarray:
    # Calculate homography from cover to book
    # Input: cover - cover image
    #        book - book image
    # Output: H - homography from cover to book

    assert book_gray.ndim == 2, 'book_gray must be grayscale'

    # Calculate descriptors (feature vectors)
    kp_book, des_book = get_image_sift_feature(book_gray)

    # Match the features
    matches = get_matches(des_cover, des_book)
    # Get the matched keypoints
    pts_cover = get_matched_pt(kp_cover, matches)
    pts_book = get_matched_pt(kp_book, matches, points_type=1)

    # Find the homography matrix
    mat =compute_homography_ransac(pts_book,pts_cover) if ransac else get_homograph_mat(pts_book, pts_cover)
    return mat

def trailer_dimensions(trailer_shape,cover_shape)->np.ndarray:
    # dimensions
    cover_w, cover_h = cover_shape[:2]
    trailer_w, trailer_h = trailer_shape[:2]

    # Get the corners of the cover
    corners = np.array([
            [0, 0],
            [0, cover_h-1],
            [cover_w-1, cover_h-1],
            [cover_w-1, 0]
        ],
        dtype=np.float32)

    # trailer dimensions wrt cover
    # Calculate dimensions
    x_min = int(np.min(corners[:,0]))
    x_max = int(np.max(corners[:,0]))
    y_min = int(np.min(corners[:,1]))
    y_max = int(np.max(corners[:,1]))

    # Calculate the dimensions of the new image
    new_w = min(trailer_w,int(x_max - x_min))
    new_h = min(trailer_h,int(y_max - y_min))

    # center point
    center = np.array([trailer_h/2, trailer_w/2])

    # trailer corners
    x_start = max(0, int(center[0] - new_h/2))
    x_end = min(trailer_h, int(center[0] + new_h/2))
    y_start = max(0, int(center[1] - new_w/2))
    y_end = min(trailer_w, int(center[1] + new_w/2))

    return np.array([x_start, x_end, y_start, y_end])

def prep_trailer_frame(trailer, homography, trailer_corners, cover_shape, book_shape)->np.ndarray:
    # x_start, x_end, y_start, y_end = trailer_corners

    cover_w, cover_h = cover_shape[:2]
    book_w, book_h = book_shape[:2]

    # cropped_trailer = trailer[x_start:x_end, y_start:y_end]
    cropped_trailer = cv2.resize(trailer, (cover_h, cover_w))
    cropped_trailer = wrap_prespective(cropped_trailer,homography,(book_h,book_w))
    return cropped_trailer

def overlay(cropped_trailer,book,mask, homography)->np.ndarray:
    # Create a mask of cover image and create its inverse mask also
    mask_3d = np.dstack([mask, mask, mask])
    mask = wrap_prespective(mask_3d,homography,(book.shape[1],book.shape[0]))[:,:,0]
    mask_inv = cv2.bitwise_not(mask)
    # Black-out the area of cover in book image
    book_bg = cv2.bitwise_and(book, book, mask=mask_inv)

    # Put cover in book image and modify the book image
    dst = cv2.add(book_bg, cropped_trailer)
    return dst

def out_frame(book, kp_cover, des_cover, trailer, cover_shape, mask):
    book_shape = book.shape[:2]

    book_gray = cv2.cvtColor(book, cv2.COLOR_BGR2GRAY)
    homography = cover_to_book_homography(kp_cover,des_cover, book_gray)

    # Get the trailer corners
    # trailer_corners = trailer_dimensions(trailer.shape, cover_shape)

    # Get the trailer frame
    cropped_trailer = prep_trailer_frame(trailer, homography, None, cover_shape, book_shape)

    # Overlay the trailer frame on the book
    dst = overlay(cropped_trailer, book, mask, homography)
    return dst
