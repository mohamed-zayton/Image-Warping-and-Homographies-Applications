# Augmented Reality

Augmented reality (AR) is an interactive experience that combines the real world and computer-generated content. In this project, we will create an AR experience that uses a book cover as a screen for a video. The video will be a movie trailer.

## Approach

This application can be completed using planar homography. Planar homography is a technique that allows us to warp an image to a plane. In this case, we will use the book cover as the plane. The video will be warped to the book cover and will appear as if it is playing on the book cover.

## Steps

### Detect the book cover using a feature detector

We use SIFT to detect the book cover. The scale-invariant feature transform (SIFT) is a computer vision algorithm to detect, describe, and match local features in images. SIFT is a feature detection algorithm that is robust to scale and rotation.

#### SIFT Algorithm

- Scale-space extrema detection

> The image is convolved with Gaussian filters at different scales, and then the difference of successive Gaussian-blurred images are taken. Keypoints are then taken as maxima/minima of the Difference of Gaussians (DoG) that occur at multiple scales.

- Keypoint localization

> Scale-space extrema detection produces too many keypoint candidates, some of which are unstable. The next step in the algorithm is to perform a detailed fit to the nearby data for accurate location, scale, and ratio of principal curvatures. This information allows the rejection of points which are low contrast (and are therefore sensitive to noise) or poorly localized along an edge.

- Orientation assignment

> In this step, each keypoint is assigned one or more orientations based on local image gradient directions. This is the key step in achieving invariance to rotation as the keypoint descriptor can be represented relative to this orientation and therefore achieve invariance to image rotation.

- Keypoint descriptor

> The image gradient magnitudes and orientations are sampled around the keypoint location, using the scale of the keypoint to select the level of Gaussian blur for the image. In order to achieve orientation invariance, the coordinates of the descriptor and the gradient orientations are rotated relative to the keypoint orientation. The magnitudes are further weighted by a Gaussian function with $ \sigma $  equal to one half the width of the descriptor window.

We use the OpenCV implementation of SIFT to detect the book cover

![Cover Image Detection][cover_detection]

### Matching the book cover to the video

In order to overlay the video on the book cover in every frame, we need to find the keypoints in the video frame that correspond to the keypoints in the book cover.

We first find the keypoints in the video frame using SIFT. Then, we use the Brute-Force Matcher to find the keypoints in the video frame that correspond to the keypoints in the book cover.

![Keypoint Matching][matches]

```python
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

```

### Homography

Briefly, the planar homography relates the transformation between two planes (up to a scale factor):

![Homography][homography]

Homography lets us relate two cameras viewing the same planar surface; Both the cameras and the surface that they view (generate images of) are located in the world-view coordinates. In other words, two 2D images are related by a homography H, if both view the same plane from a different angle.

the estimation of H requires the estimation of 9 parameters. In other words, H has 9 degrees of freedom.

![Homography][homoestimate]

The system λp = Hp can be solved by SVD.

```python
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

```

### Finding the position of the book cover

We can now use the homography matrix to find the corners of the book cover in each frame. We can then use these corners to warp the video to the book cover.

We get the corner points of the cover image. We then use the homography matrix to find the corner points of the cover image in the video frame.

```python
# Get the corners of the cover in the book
corners = np.array([
        [0, 0],
        [0, cover_h-1],
        [cover_w-1, cover_h-1],
        [cover_w-1, 0]
    ],
    dtype=np.float32)
corners_book = transform_with_homography(homography, corners)

# Draw the cover in the book
outline = cv2.polylines(book_frames[0].copy(), [np.int32(corners_book)], True, (255, 0, 0), 3)

plt.imshow(outline[:,:,::-1])
```

![Corners][corners]

### Preparing the movie video

#### Aspect Ratio

There are two approaches to obtain a suitable aspect ratio that fits the book cover.

The first approach is to crop the video to the same aspect ratio as the book cover. This approach is not ideal as it will result in a loss of information, which is not desirable.

The second approach is to resize the video to the book cover. Which is the approach we use.

#### Overlaying the video

If we only use the extreme values of the corners of the book cover, we will not get an aligned video, as the video will be warped. This will not give us the effect we want, which is using the book cover as a screen.

Our approach is to warp the video to the book cover.

Steps:

1. warp the frame to the book cover using the same homography matrix we used to find the corners of the book cover in the frame.

1. Create a mask of the movie in the book frame.

1. Black out the area of the book cover in the book frame.

1. Add the movie frame to the book frame.

```python
cropped_trailer = wrap_prespective(cropped_trailer, homography, (book_w, book_h))
# Create a mask of cover image and create its inverse mask also
cropped_gray = cv2.cvtColor(cropped_trailer, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(cropped_gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Black-out the area of cover in book image
book_bg = cv2.bitwise_and(book_frames[0], book_frames[0], mask=mask_inv)

# Put cover in book image and modify the book image
dst = cv2.add(book_bg, cropped_trailer)

plt.imshow(dst[:,:,::-1])
```

![Overlay][overlay]

We also show the results using the corners approach.

![Overlay][overlay_corners]

### Putting it all together

Finally, we repeat the process for all the frames in the video, and save the result.

[![video result][video_badge]][out_link]
<!-- References -->

[cover_detection]: ./img/cover_detection.png
[matches]: ./img/matches.png
[homography]: ./img/homography.jpeg
[homoestimate]: ./img/homoestimate
[corners]: ./img/corners.png
[overlay]: ./img/ready.png
[overlay_corners]: ./img/corners_overlay.png
[out_link]: https://drive.google.com/drive/folders/1-AdBXaxfd_4z_bxsJoxAFi-_UA1XCNf_?usp=share_link
[video_badge]: https://img.shields.io/badge/Video-Result-red
