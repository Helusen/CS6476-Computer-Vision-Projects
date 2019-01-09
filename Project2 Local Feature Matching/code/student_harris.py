import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage, spatial
import math

def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. For example, you could compute connected
    components and take the maximum value within each component.
    Alternatively, you could run a max() operator on each sliding window. You
    could use this to ensure that every interest point is at a local maximum
    of cornerness.

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    #############################################################################
    # TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                                                      #
    #############################################################################

    G_low = cv2.getGaussianKernel(9,2)
    filtered_image = cv2.filter2D(image,-1,G_low)
    dy, dx = np.gradient(filtered_image)

    Ix2 = dx**2
    Ixy = dx*dy
    Iy2 = dy**2

    G_high = cv2.getGaussianKernel(15,2)
    dx2_w = cv2.filter2D(Ix2, -1, G_high)
    dxy_w = cv2.filter2D(Ixy, -1, G_high)
    dy2_w = cv2.filter2D(Iy2, -1, G_high)

    alpha = 0.06

    Response = dx2_w * dy2_w - dxy_w **2 -alpha * (dx2_w + dy2_w) **2
    ## find local max
    # local_max = ndimage.maximum_filter(Response, size = (7, 7))

    corners = []
    for i in range(Response.shape[0]):
        for j in range(Response.shape[1]):
                corners.append([Response[i,j],j,i])

    sorted_corners = sorted(corners, key = lambda tup: tup[0], reverse = True)
    sorted_corners = sorted_corners[0:10000]

    ## ANMS
    radius = []

    for i in range(len(sorted_corners)):
        r_sqaure = float('inf')
        point1 = sorted_corners[i]
        for j in range(i):
            point2 = sorted_corners[j]
            tmp = (point1[1]-point2[1])**2 +(point1[2]-point2[2])**2
            r_sqaure = min(tmp, r_sqaure)
        radius.append([math.sqrt(r_sqaure), point1[0], point1[1], point1[2]])

    points_sorted = sorted(radius, key = lambda tup:tup[0], reverse = True)

    x = np.array([item[2] for item in points_sorted[:1500]])
    y = np.array([item[3] for item in points_sorted[:1500]])

    return x,y, confidences, scales, orientations

    # raise NotImplementedError('`get_interest_points` function in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    
    #############################################################################
    # TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
    # While most feature detectors simply look for local maxima in              #
    # the interest function, this can lead to an uneven distribution            #
    # of feature points across the image, e.g., points will be denser           #
    # in regions of higher contrast. To mitigate this problem, Brown,           #
    # Szeliski, and Winder (2005) only detect features that are both            #
    # local maxima and whose response value is significantly (10%)              #
    # greater than that of all of its neighbors within a radius r. The          #
    # goal is to retain only those points that are a maximum in a               #
    # neighborhood of radius r pixels. One way to do so is to sort all          #
    # points by the response strength, from large to small response.            #
    # The first entry in the list is the global maximum, which is not           #
    # suppressed at any radius. Then, we can iterate through the list           #
    # and compute the distance to each interest point ahead of it in            #
    # the list (these are pixels with even greater response strength).          #
    # The minimum of distances to a keypoint's stronger neighbors               #
    # (multiplying these neighbors by >=1.1 to add robustness) is the           #
    # radius within which the current point is a local maximum. We              #
    # call this the suppression radius of this interest point, and we           #
    # save these suppression radii. Finally, we sort the suppression            #
    # radii from large to small, and return the n keypoints                     #
    # associated with the top n suppression radii, in this sorted               #
    # orderself. Feel free to experiment with n, we used n=1500.                #
    #                                                                           #
    # See:                                                                      #
    # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
    # or                                                                        #
    # https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
    #############################################################################

    # raise NotImplementedError('adaptive non-maximal suppression in ' +
    # '`student_harris.py` needs to be implemented')

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  ##  return x,y, confidences, scales, orientations

