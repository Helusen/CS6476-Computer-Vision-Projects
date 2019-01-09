import numpy as np
import cv2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace


def im2single(im):
    im = im.astype(np.float32) / 255
    return im

def single2im(im):
    im *= 255
    im = im.astype(np.uint8)
    return im

def load_image(path):
    return cv2.imread(path)[:,:,::-1]

def save_image(path, im):
    return cv2.imwrite(path, single2im(im.copy())[:, :, ::-1])

def evaluate_points(M, points_2d, points_3d):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to
    :param M: projection matrix 3 x 4
    :param points_2d: 2D points N x 2
    :param points_3d: 3D points N x 3
    :return:
    """
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

def visualize_points(actual_pts, projected_pts):
    """
    Visualize the actual 2D points and the projected 2D points calculated from
    the projection matrix
    You do not need to modify anything in this function, although you can if you
    want to.
    :param actual_pts: N x 2
    :param projected_pts: N x 2
    :return:
    """
    _, ax = plt.subplots()
    ax.scatter(actual_pts[:, 0], actual_pts[:, 1], c='red', marker='o',
        label='Actual points')
    ax.scatter(projected_pts[:, 0], projected_pts[:, 1], c='green', marker='+',
        label='Projected points')
    ax.legend()

def plot3dview(points_3d, camera_center):
    """
    Visualize the actual 3D points and the estimated 3D camera center.
    You do not need to modify anything in this function, although you can if you
    want to.
    :param points_3d: N x 3
    :param camera_center: 1 x 3
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue',
        marker='o', s=10, depthshade=0)
    camera_center = camera_center.squeeze()
    ax.scatter(camera_center[0],  camera_center[1], camera_center[2], c='red',
        marker='x', s=20, depthshade=0)

    # draw vertical lines connecting each point to ground
    min_z = min(points_3d[:, 2])
    for p in points_3d:
        x, y, z = p
        ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c='black', linewidth=1)
    x, y, z = camera_center
    ax.plot3D(xs=[x, x], ys=[y, y], zs=[z, min_z], c='black', linewidth=1)

def draw_epipolar_lines(F, img_left, img_right, pts_left, pts_right):
    """
    Draw the epipolar lines given the fundamental matrix, left right images
    and left right datapoints

    You do not need to modify anything in this function, although you can if
    you want to.
    :param F: 3 x 3; fundamental matrix
    :param img_left:
    :param img_right:
    :param pts_left: N x 2
    :param pts_right: N x 2
    :return:
    """
    # lines in the RIGHT image
    # corner points
    p_ul = np.asarray([0, 0, 1])
    p_ur = np.asarray([img_right.shape[1], 0, 1])
    p_bl = np.asarray([0, img_right.shape[0], 1])
    p_br = np.asarray([img_right.shape[1], img_right.shape[0], 1])

    # left and right border lines
    l_l = np.cross(p_ul, p_bl)
    l_r = np.cross(p_ur, p_br)

    fig, ax = plt.subplots()
    ax.imshow(img_right)
    ax.autoscale(False)
    ax.scatter(pts_right[:, 0], pts_right[:, 1], marker='o', s=20, c='yellow',
        edgecolors='red')
    for p in pts_left:
        p = np.hstack((p, 1))[:, np.newaxis]
        l_e = np.dot(F, p).squeeze()  # epipolar line
        p_l = np.cross(l_e, l_l)
        p_r = np.cross(l_e, l_r)
        x = [p_l[0]/p_l[2], p_r[0]/p_r[2]]
        y = [p_l[1]/p_l[2], p_r[1]/p_r[2]]
        ax.plot(x, y, linewidth=1, c='blue')

    # lines in the LEFT image
    # corner points
    p_ul = np.asarray([0, 0, 1])
    p_ur = np.asarray([img_left.shape[1], 0, 1])
    p_bl = np.asarray([0, img_left.shape[0], 1])
    p_br = np.asarray([img_left.shape[1], img_left.shape[0], 1])

    # left and right border lines
    l_l = np.cross(p_ul, p_bl)
    l_r = np.cross(p_ur, p_br)

    fig, ax = plt.subplots()
    ax.imshow(img_left)
    ax.autoscale(False)
    ax.scatter(pts_left[:, 0], pts_left[:, 1], marker='o', s=20, c='yellow',
        edgecolors='red')
    for p in pts_right:
        p = np.hstack((p, 1))[:, np.newaxis]
        l_e = np.dot(F.T, p).squeeze()  # epipolar line
        p_l = np.cross(l_e, l_l)
        p_r = np.cross(l_e, l_r)
        x = [p_l[0]/p_l[2], p_r[0]/p_r[2]]
        y = [p_l[1]/p_l[2], p_r[1]/p_r[2]]
        ax.plot(x, y, linewidth=1, c='blue')

def get_matches(pic_a, pic_b, n_feat):
    orb = cv2.ORB_create(nfeatures=int(n_feat))
    kp_a, desc_a = orb.detectAndCompute(pic_a, None)
    kp_b, desc_b = orb.detectAndCompute(pic_b, None)
    dm = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = dm.knnMatch(desc_b, desc_a, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < n.distance/1.2:
            good_matches.append(m)
    pts_a = []
    pts_b = []
    for m in good_matches:
        pts_a.append(kp_a[m.trainIdx].pt)
        pts_b.append(kp_b[m.queryIdx].pt)
    return np.asarray(pts_a), np.asarray(pts_b)

def hstack_images(imgA, imgB):
    """
    Stacks 2 images side-by-side
    :param imgA:
    :param imgB:
    :return:
    """
    Height = max(imgA.shape[0], imgB.shape[0])
    Width  = imgA.shape[1] + imgB.shape[1]

    newImg = np.zeros((Height, Width, 3), dtype=imgA.dtype)
    newImg[:imgA.shape[0], :imgA.shape[1], :] = imgA
    newImg[:imgB.shape[0], imgA.shape[1]:, :] = imgB

    return newImg

def show_correspondence2(imgA, imgB, X1, Y1, X2, Y2, line_colors=None):
    """
    Visualizes corresponding points between two images. Corresponding points will
    have the same random color.
    :param imgA:
    :param imgB:
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param line_colors: N x 3 colors of correspondence lines (optional)
    :return:
    """
    newImg = hstack_images(imgA, imgB)
    shiftX = imgA.shape[1]
    X1 = X1.astype(np.int)
    Y1 = Y1.astype(np.int)
    X2 = X2.astype(np.int)
    Y2 = Y2.astype(np.int)

    dot_colors = np.random.rand(len(X1), 3)
    if imgA.dtype == np.uint8:
        dot_colors *= 255
    if line_colors is None:
        line_colors = dot_colors

    for x1, y1, x2, y2, dot_color, line_color in zip(X1, Y1, X2, Y2, dot_colors,
            line_colors):
        newImg = cv2.circle(newImg, (x1, y1), 5, dot_color, -1)
        newImg = cv2.circle(newImg, (x2+shiftX, y2), 5, dot_color, -1)
        newImg = cv2.line(newImg, (x1, y1), (x2+shiftX, y2), line_color, 2,
                                            cv2.LINE_AA)
    return newImg