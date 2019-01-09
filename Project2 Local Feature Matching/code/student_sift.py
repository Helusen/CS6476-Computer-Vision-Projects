import numpy as np 
olderr = np.seterr(all='ignore')
import cv2
import scipy
from scipy import ndimage, spatial
import math

def gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    ############################################################################
    fv = np.zeros((x.size,128))
    pad_width = feature_width //2 
    image = np.pad(image,((pad_width,pad_width),(pad_width,pad_width)), mode='constant')
    dx = cv2.Sobel(image, cv2.CV_64F,1, 0, ksize = 15)
    dy = cv2.Sobel(image, cv2.CV_64F,0, 1, ksize = 15)
    features=[]

    for kp in range(0, x.size):
        histogram = np.zeros((4,4,8))
        for j in range(feature_width):
            for i in range(feature_width):
                dx_tmp = dx[(int)(y[kp])+j][(int)(x[kp])+i]
                dy_tmp = dy[(int)(y[kp])+j][(int)(x[kp])+i]
                mag = math.sqrt(dx_tmp**2 + dy_tmp**2)
                bin = np.arctan2(dy_tmp, dx_tmp)
                if bin > 1:
                    bin = 2
                if bin < -1:
                    bin = -1
                if dx_tmp >0:
                    histogram[(int)(j/4)][(int)(i/4)][math.ceil(bin+1)] += mag
                else:
                    histogram[(int)(j/4)][(int)(i/4)][math.ceil(bin+5)] += mag
        feature = np.reshape(histogram,(1,128))
        feature = feature/(feature.sum())
        features.append(feature)

    fv = np.array(features)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv


    raise NotImplementedError('`get_features` function in ' +
        '`student_sift.py` needs to be implemented')
