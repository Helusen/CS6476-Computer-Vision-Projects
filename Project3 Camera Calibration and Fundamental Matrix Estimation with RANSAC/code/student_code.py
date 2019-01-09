import numpy as np
from scipy import linalg



def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.
    M = np.asarray([[0.1768, 0.7018, 0.7948, 0.4613],
                    [0.6750, 0.3152, 0.1136, 0.0480],
                    [0.1020, 0.1725, 0.7244, 0.9932]])

    ###########################################################################
    # TODO: YOUR PROJECTION MATRIX CALCULATION CODE HERE
    ###########################################################################
    points_num = points_2d.shape[0]
    a = [] 
    b = []

    for n in range(points_num):
        x = points_3d[n,0]
        y = points_3d[n,1]
        z = points_3d[n,2]
        u = points_2d[n,0]
        v = points_2d[n,1]
        a.append([x,y,z,1,0,0,0,0, -u*x, -u*y, -u*z])
        b.append([u])
        a.append([0,0,0,0,x,y,z,1, -v*x, -v*y, -v*z])
        b.append([v])

    A = np.mat(a)
    B = np.mat(b)
    M = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, B))
    M = np.array(M.T)
    M = np.append(M,[1])
    M = np.reshape(M,(3,4))
   ## raise NotImplementedError('`calculate_projection_matrix` function in ' +
    #    '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.
    cc = np.asarray([1, 1, 1])

    ###########################################################################
    # TODO: YOUR CAMERA CENTER CALCULATION CODE HERE
    ###########################################################################

    cc = np.dot(np.linalg.inv(np.dot(-M[:,0:3].T, -M[:,0:3])), np.dot(-M[:,0:3].T, M[:,3]))

    # raise NotImplementedError('`calculate_camera_center` function in ' +
    #     '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return cc

def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.

    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    # Placeholder fundamental matrix
    F = np.asarray([[0, 0, -0.0004],
                    [0, 0, 0.0032],
                    [0, -0.0044, 0.1034]])

    ###########################################################################
    # TODO: YOUR FUNDAMENTAL MATRIX ESTIMATION CODE HERE
    ###########################################################################

    points_num = points_a.shape[0]
    A = []
    B = np.ones((points_num,1))

    cu_a = np.sum(points_a[:,0])/points_num
    cv_a = np.sum(points_a[:,1])/points_num

    s = points_num/np.sum(((points_a[:,0]-cu_a)**2 + (points_a[:,1]-cv_a)**2)**(1/2))
    T_a =np.dot(np.array([[s,0,0], [0,s,0], [0,0,1]]), np.array([[1,0,-cu_a],[0,1,-cv_a],[0,0,1]]))

    points_a = np.array(points_a.T)
    points_a = np.append(points_a,B)

    points_a = np.reshape(points_a, (3,points_num))
    points_a = np.dot(T_a, points_a)
    points_a = points_a.T

    cu_b = np.sum(points_b[:,0])/points_num
    cv_b = np.sum(points_b[:,1])/points_num

    s = points_num/np.sum(((points_b[:,0]-cu_b)**2 + (points_b[:,1]-cv_b)**2)**(1/2))
    T_b =np.dot(np.array([[s,0,0], [0,s,0], [0,0,1]]), np.array([[1,0,-cu_b],[0,1,-cv_b],[0,0,1]]))

    points_b = np.array(points_b.T)
    points_b = np.append(points_b,B)

    points_b = np.reshape(points_b, (3,points_num))
    points_b = np.dot(T_b, points_b)
    points_b = points_b.T

    for i in range(points_num):
        u_a = points_a[i,0]
        v_a = points_a[i,1]
        u_b = points_b[i,0]
        v_b = points_b[i,1]
        A.append([u_a*u_b, v_a*u_b, u_b, u_a*v_b, v_a*v_b, v_b, u_a, v_a])

    A = np.array(A)
    F = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, -B))
    F = np.append(F,[1])
   
    F = np.reshape(F,(3,3)).T
    F = np.dot(T_a.T, F)
    F = np.dot(F, T_b)

    F = F.T
    U,S,V = np.linalg.svd(F)
    S = np.array([[S[0],0,0],[0,S[1],0],[0,0,0]])
    F = np.dot(U, S)
    F = np.dot(F, V)

    # raise NotImplementedError('`estimate_fundamental_matrix` function in ' +
    #     '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return F

def ransac_fundamental_matrix(matches_a, matches_b):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    # Placeholder values

    ###########################################################################
    # TODO: YOUR RANSAC CODE HERE
    ###########################################################################

    matches_num = matches_a.shape[0]
    Best_count = 0
    
    for iter in range(500):
        sampled_idx = np.random.randint(0, matches_num, size = 8)
        F = estimate_fundamental_matrix(matches_a[sampled_idx, :], matches_b[sampled_idx, :])
        in_a = []
        in_b = []
        update = 0
        for i in range(matches_num):
            matches_aa = np.append(matches_a[i,:],1)
            matches_bb = np.append(matches_b[i,:],1)
            error = np.dot(matches_aa, F.T)
            error = np.dot(error, matches_bb.T)
            if abs(error) < 0.05:
                in_a.append(matches_a[i,:])
                in_b.append(matches_b[i,:])
                update +=1

        if update > Best_count:
            Best_count = update
            best_F = F
            inliers_a = in_a
            inliers_b = in_b

    inliers_a = np.array(inliers_a)
    inliers_b = np.array(inliers_b)
    # idx = np.random.randint(0, inliers_a.shape[0], size = 50)
    # inliers_a = inliers_a[idx, :]
    # inliers_b = inliers_b[idx, :]

    # raise NotImplementedError('`ransac_fundamental_matrix` function in ' +
    #     '`student_code.py` needs to be implemented')

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return best_F, inliers_a, inliers_b