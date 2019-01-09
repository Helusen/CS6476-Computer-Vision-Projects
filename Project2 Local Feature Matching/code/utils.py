# Please do not modify this file.

import numpy as np
import cv2
import pickle


def im2single(im):
    im = im.astype(np.float32) / 255

    return im

def single2im(im):
    im *= 255
    im = im.astype(np.uint8)

    return im

def rgb2gray(rgb):
    """Convert RGB image to grayscale
    Args:
    - rgb: A numpy array of shape (m,n,c) representing an RGB image
    Returns:
    - gray: A numpy array of shape (m,n) representing the corresponding grayscale image
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def load_image(path):
    """
    Args:
    - path: string representing a filepath to an image
    """
    return im2single(cv2.imread(path))[:, :, ::-1]

def save_image(path, im):
    """
    Args:
    - path:
    - im: A numpy array of shape
    """
    return cv2.imwrite(path, single2im(im.copy())[:, :, ::-1])

def cheat_interest_points(eval_file, scale_factor):
    """
    This function is provided for development and debugging but cannot be used in
    the final handin. It 'cheats' by generating interest points from known
    correspondences. It will only work for the 3 image pairs with known
    correspondences.

    Args:
    - eval_file: string representing the file path to the list of known correspondences
    - scale_factor: Python float representing the scale needed to map from the original
            image coordinates to the resolution being used for the current experiment.

    Returns:
    - x1: A numpy array of shape (k,) containing ground truth x-coordinates of imgA correspondence pts
    - y1: A numpy array of shape (k,) containing ground truth y-coordinates of imgA correspondence pts
    - x2: A numpy array of shape (k,) containing ground truth x-coordinates of imgB correspondence pts
    - y2: A numpy array of shape (k,) containing ground truth y-coordinates of imgB correspondence pts
    """
    with open(eval_file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')

    return d['x1'] * scale_factor, d['y1'] * scale_factor, d['x2'] * scale_factor,\
                 d['y2'] * scale_factor

def hstack_images(imgA, imgB):
    """
    Stacks 2 images side-by-side and creates one combined image.

    Args:
    - imgA: A numpy array of shape (M,N,3) representing rgb image
    - imgB: A numpy array of shape (D,E,3) representing rgb image

    Returns:
    - newImg: A numpy array of shape (max(M,D), N+E, 3)
    """
    Height = max(imgA.shape[0], imgB.shape[0])
    Width  = imgA.shape[1] + imgB.shape[1]

    newImg = np.zeros((Height, Width, 3), dtype=imgA.dtype)
    newImg[:imgA.shape[0], :imgA.shape[1], :] = imgA
    newImg[:imgB.shape[0], imgA.shape[1]:, :] = imgB

    return newImg

def show_interest_points(img, X, Y):
    """
    Visualized interest points on an image with random colors

    Args:
    - img: A numpy array of shape (M,N,C)
    - X: A numpy array of shape (k,) containing x-locations of interest points
    - Y: A numpy array of shape (k,) containing y-locations of interest points

    Returns:
    - newImg: A numpy array of shape (M,N,C) showing the original image with
            colored circles at keypoints plotted on top of it
    """
    newImg = img.copy()
    for x, y in zip(X.astype(int), Y.astype(int)):
        cur_color = np.random.rand(3)
        newImg = cv2.circle(newImg, (x, y), 10, cur_color, -1, cv2.LINE_AA)

    return newImg

def show_correspondence_circles(imgA, imgB, X1, Y1, X2, Y2):
    """
    Visualizes corresponding points between two images by plotting circles at
    each correspondence location. Corresponding points will have the same random color.

    Args:
    - imgA: A numpy array of shape (M,N,3)
    - imgB: A numpy array of shape (D,E,3)
    - x1: A numpy array of shape (k,) containing x-locations of keypoints in imgA
    - y1: A numpy array of shape (k,) containing y-locations of keypoints in imgA
    - x2: A numpy array of shape (j,) containing x-locations of keypoints in imgB
    - y2: A numpy array of shape (j,) containing y-locations of keypoints in imgB

    Returns:
    - newImg: A numpy array of shape (max(M,D), N+E, 3)
    """
    newImg = hstack_images(imgA, imgB)
    shiftX = imgA.shape[1]
    X1 = X1.astype(np.int)
    Y1 = Y1.astype(np.int)
    X2 = X2.astype(np.int)
    Y2 = Y2.astype(np.int)

    for x1, y1, x2, y2 in zip(X1, Y1, X2, Y2):
        cur_color = np.random.rand(3)
        green = (0, 1, 0)
        newImg = cv2.circle(newImg, (x1, y1), 10, cur_color, -1, cv2.LINE_AA)
        newImg = cv2.circle(newImg, (x1, y1), 10, green, 2, cv2.LINE_AA)
        newImg = cv2.circle(newImg, (x2+shiftX, y2), 10, cur_color, -1, cv2.LINE_AA)
        newImg = cv2.circle(newImg, (x2+shiftX, y2), 10, green, 2, cv2.LINE_AA)

    return newImg

def show_correspondence_lines(imgA, imgB, X1, Y1, X2, Y2, line_colors=None):
    """
    Visualizes corresponding points between two images by drawing a line segment
    between the two images for each (x1,y1) (x2,y2) pair.

    Args:
    - imgA: A numpy array of shape (M,N,3)
    - imgB: A numpy array of shape (D,E,3)
    - x1: A numpy array of shape (k,) containing x-locations of keypoints in imgA
    - y1: A numpy array of shape (k,) containing y-locations of keypoints in imgA
    - x2: A numpy array of shape (j,) containing x-locations of keypoints in imgB
    - y2: A numpy array of shape (j,) containing y-locations of keypoints in imgB
    - line_colors: A numpy array of shape (N x 3) with colors of correspondence lines (optional)

    Returns:
    - newImg: A numpy array of shape (max(M,D), N+E, 3)
    """
    newImg = hstack_images(imgA, imgB)
    shiftX = imgA.shape[1]
    X1 = X1.astype(np.int)
    Y1 = Y1.astype(np.int)
    X2 = X2.astype(np.int)
    Y2 = Y2.astype(np.int)

    dot_colors = np.random.rand(len(X1), 3)
    if line_colors is None:
        line_colors = dot_colors

    for x1, y1, x2, y2, dot_color, line_color in zip(X1, Y1, X2, Y2, dot_colors,
            line_colors):
        newImg = cv2.circle(newImg, (x1, y1), 5, dot_color, -1)
        newImg = cv2.circle(newImg, (x2+shiftX, y2), 5, dot_color, -1)
        newImg = cv2.line(newImg, (x1, y1), (x2+shiftX, y2), line_color, 2,
                                            cv2.LINE_AA)
    return newImg

def show_ground_truth_corr(imgA, imgB, corr_file, show_lines=True):
    """
    Show the ground truth correspondeces

    Args:
    - imgA: string, representing the filepath to the first image
    - imgB: string, representing the filepath to the second image
    - corr_file: filepath to pickle (.pkl) file containing the correspondences
    - show_lines: boolean, whether to visualize the correspondences as line segments
    """
    imgA = load_image(imgA)
    imgB = load_image(imgB)
    with open(corr_file, 'rb') as f:
        d = pickle.load(f)
    if show_lines:
        return show_correspondence_lines(imgA, imgB, d['x1'], d['y1'], d['x2'], d['y2'])
    else:
        # show circles
        return show_correspondence_circles(imgA, imgB, d['x1'], d['y1'], d['x2'], d['y2'])

def load_corr_pkl_file(corr_fpath):
    """ Load ground truth correspondences from a pickle (.pkl) file. """
    with open(corr_fpath, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
    x1 = d['x1'].squeeze()
    y1 = d['y1'].squeeze()
    x2 = d['x2'].squeeze()
    y2 = d['y2'].squeeze()

    return x1,y1,x2,y2


def evaluate_correspondence(imgA, imgB, corr_fpath, scale_factor, x1_est, y1_est,
        x2_est, y2_est, confidences=None, num_req_matches=100):
    """
    Function to evaluate estimated correspondences against ground truth.

    The evaluation requires 100 matches to receive full credit
    when num_req_matches=100 because we define accuracy as:

    Accuracy = (true_pos)/(true_pos+false_pos) * min(num_matches,num_req_matches)/num_req_matches

    Args:
    - imgA: A numpy array of shape (M,N,C) representing a first image
    - imgB: A numpy array of shape (M,N,C) representing a second image
    - corr_fpath: string, representing a filepath to a .pkl file containing ground truth correspondences
    - scale_factor: scale factor on the size of the images
    - x1_est: A numpy array of shape (k,) containing estimated x-coordinates of imgA correspondence pts
    - y1_est: A numpy array of shape (k,) containing estimated y-coordinates of imgA correspondence pts
    - x2_est: A numpy array of shape (k,) containing estimated x-coordinates of imgB correspondence pts
    - y2_est: A numpy array of shape (k,) containing estimated y-coordinates of imgB correspondence pts
    - confidences: (optional) confidence values in the matches
    """
    if confidences is None:
        confidences = np.random.rand(len(x1_est))
        confidences /= np.max(confidences)

    x1_est = x1_est.squeeze() / scale_factor
    y1_est = y1_est.squeeze() / scale_factor
    x2_est = x2_est.squeeze() / scale_factor
    y2_est = y2_est.squeeze() / scale_factor

    num_matches = x1_est.shape[0]

    x1,y1,x2,y2 = load_corr_pkl_file(corr_fpath)

    good_matches = [False for _ in range(len(x1_est))]
    # array marking which GT pairs are already matched
    matched = [False for _ in range(len(x1))]

    # iterate through estimated pairs in decreasing order of confidence
    priority = np.argsort(-confidences)
    for i in priority:
        # print('Examining ({:4.0f}, {:4.0f}) to ({:4.0f}, {:4.0f})'.format(
        #     x1_est[i], y1_est[i], x2_est[i], y2_est[i]))
        cur_offset = np.asarray([x1_est[i]-x2_est[i], y1_est[i]-y2_est[i]])
        # for each x1_est find nearest ground truth point in x1
        dists = np.linalg.norm(np.vstack((x1_est[i]-x1, y1_est[i]-y1)), axis=0)
        best_matches = np.argsort(dists)

        # find the best match that is not taken yet
        for match_idx in best_matches:
            if not matched[match_idx]:
                break
        else:
            continue

        # A match is good only if
        # (1) An unmatched GT point exists within 150 pixels, and
        # (2) GT correspondence offset is within 25 pixels of estimated
        #     correspondence offset
        gt_offset = np.asarray([x1[match_idx]-x2[match_idx],
            y1[match_idx]-y2[match_idx]])
        offset_dist = np.linalg.norm(cur_offset-gt_offset)
        if (dists[match_idx] < 150.0) and (offset_dist < 25):
            good_matches[i] = True
            print('Correct')
        else:
            print('Incorrect')

    print('You found {}/{} required matches'.format(num_matches, num_req_matches))
    accuracy = np.mean(good_matches) * min(num_matches, num_req_matches)*1./num_req_matches
    print('Accuracy = {:f}'.format(accuracy))
    green = np.asarray([0, 1, 0], dtype=float)
    red = np.asarray([1, 0, 0], dtype=float)
    line_colors = np.asarray([green if m else red for m in good_matches])

    return accuracy, show_correspondence_lines(imgA, imgB,
                                               x1_est*scale_factor, y1_est*scale_factor,
                                               x2_est*scale_factor, y2_est*scale_factor,
                                               line_colors)
