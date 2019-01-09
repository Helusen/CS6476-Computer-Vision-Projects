import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from IPython.core.debugger import set_trace


def get_tiny_images(image_paths):
  """
  This feature is inspired by the simple tiny images used as features in
  80 million tiny images: a large dataset for non-parametric object and
  scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
  Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
  pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

  To build a tiny image feature, simply resize the original image to a very
  small square resolution, e.g. 16x16. You can either resize the images to
  square while ignoring their aspect ratio or you can crop the center
  square portion out of each image. Making the tiny images zero mean and
  unit length (normalizing them) will increase performance modestly.

  Useful functions:
  -   cv2.resize
  -   use load_image(path) to load a RGB images and load_image_gray(path) to
      load grayscale images

  Args:
  -   image_paths: list of N elements containing image paths

  Returns:
  -   feats: N x d numpy array of resized and then vectorized tiny images
            e.g. if the images are resized to 16x16, d would be 256
  """
  # dummy feats variable
  feats = []

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################=
  for i in range(len(image_paths)):
    image = load_image_gray(image_paths[i])
    image = cv2.resize(image, (16,16))

    ##print(image)
   
    Ir = image.flatten()
    Izm = Ir - np.mean(Ir)
    Iul = Izm/np.max(np.abs(Izm))
    feats.append(Iul)

  
  feats = np.array(feats)

  # raise NotImplementedError('`get_tiny_images` function in ' +
  #       '`student_code.py` needs to be implemented')

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return feats

def build_vocabulary(image_paths, vocab_size):
  """
  This function will sample SIFT descriptors from the training images,
  cluster them with kmeans, and then return the cluster centers.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
        http://www.vlfeat.org/matlab/vl_dsift.html
          -  frames is a N x 2 matrix of locations, which can be thrown away
          here (but possibly used for extra credit in get_bags_of_sifts if
          you're making a "spatial pyramid").
          -  descriptors is a N x 128 matrix of SIFT features
        Note: there are step, bin size, and smoothing parameters you can
        manipulate for dsift(). We recommend debugging with the 'fast'
        parameter. This approximate version of SIFT is about 20 times faster to
        compute. Also, be sure not to use the default value of step size. It
        will be very slow and you'll see relatively little performance gain
        from extremely dense sampling. You are welcome to use your own SIFT
        feature code! It will probably be slower, though.
  -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
          http://www.vlfeat.org/matlab/vl_kmeans.html
            -  X is a N x d numpy array of sampled SIFT features, where N is
               the number of features sampled. N should be pretty large!
            -  K is the number of clusters desired (vocab_size)
               cluster_centers is a K x d matrix of cluster centers. This is
               your vocabulary.

  Args:
  -   image_paths: list of image paths.
  -   vocab_size: size of vocabulary

  Returns:
  -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
      cluster center / visual word
  """
  # Load images from the training set. To save computation time, you don't
  # necessarily need to sample from all images, although it would be better
  # to do so. You can randomly sample the descriptors from each image to save
  # memory and speed up the clustering. Or you can simply call vl_dsift with
  # a large step size here, but a smaller step size in get_bags_of_sifts.
  #
  # For each loaded image, get some SIFT features. You don't have to get as
  # many SIFT features as you will in get_bags_of_sift, because you're only
  # trying to get a representative sample here.
  #
  # Once you have tens of thousands of SIFT features from many training
  # images, cluster them with kmeans. The resulting centroids are now your
  # visual word vocabulary.

  dim = 128      # length of the SIFT descriptors that you are going to compute.
  vocab = np.zeros((vocab_size,dim))


  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################
  total_SIFT_features = np.zeros((20*len(image_paths), dim))
  index = 0

  for i in range(len(image_paths)):
    image = load_image_gray(image_paths[i]).astype('float32')

    [locations, SIFT_features] = vlfeat.sift.dsift(image,fast=True,step=15)

    rand_permutation = np.random.permutation(SIFT_features.shape[0])

    for j in range(20): 
      k = rand_permutation[j]
      total_SIFT_features[j+index, :] = SIFT_features[k, :]
    index = index + 20

  vocab = vlfeat.kmeans.kmeans(total_SIFT_features.astype('float32'), vocab_size)

  # raise NotImplementedError('`build_vocabulary` function in ' +
  #       '`student_code.py` needs to be implemented')

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return vocab

def get_bags_of_sifts(image_paths, vocab_filename):
  """
  This feature representation is described in the handout, lecture
  materials, and Szeliski chapter 14.
  You will want to construct SIFT features here in the same way you
  did in build_vocabulary() (except for possibly changing the sampling
  rate) and then assign each local feature to its nearest cluster center
  and build a histogram indicating how many times each cluster was used.
  Don't forget to normalize the histogram, or else a larger image with more
  SIFT features will look very different from a smaller version of the same
  image.

  Useful functions:
  -   Use load_image(path) to load RGB images and load_image_gray(path) to load
          grayscale images
  -   frames, descriptors = vlfeat.sift.dsift(img)
          http://www.vlfeat.org/matlab/vl_dsift.html
        frames is a M x 2 matrix of locations, which can be thrown away here
          (but possibly used for extra credit in get_bags_of_sifts if you're
          making a "spatial pyramid").
        descriptors is a M x 128 matrix of SIFT features
          note: there are step, bin size, and smoothing parameters you can
          manipulate for dsift(). We recommend debugging with the 'fast'
          parameter. This approximate version of SIFT is about 20 times faster
          to compute. Also, be sure not to use the default value of step size.
          It will be very slow and you'll see relatively little performance
          gain from extremely dense sampling. You are welcome to use your own
          SIFT feature code! It will probably be slower, though.
  -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
          finds the cluster assigments for features in data
            -  data is a M x d matrix of image features
            -  vocab is the vocab_size x d matrix of cluster centers
            (vocabulary)
            -  assignments is a Mx1 array of assignments of feature vectors to
            nearest cluster centers, each element is an integer in
            [0, vocab_size)

  Args:
  -   image_paths: paths to N images
  -   vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.

  Returns:
  -   image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters or equivalently the number of entries in each image's
          histogram (vocab_size) below.
  """
  # load vocabulary
  with open(vocab_filename, 'rb') as f:
    vocab = pickle.load(f)

  # dummy features variable
  
  vocab_size = vocab.shape[0]
  feats = []

  for i in range(len(image_paths)):
    image = load_image_gray(image_paths[i]).astype('float32')
    [locations, SIFT_features] = vlfeat.sift.dsift(image,fast=True,step=10)
    SIFT_features = SIFT_features.astype('float32')

    Hist = np.zeros(vocab_size)
    D = sklearn_pairwise.pairwise_distances(SIFT_features, vocab)
    for j in D:
      closet = np.argmin(a=j,axis=0)
      Hist[closet] +=1

    Hist = Hist / np.linalg.norm(Hist)

    feats.append(Hist)

    # assignments = vlfeat.kmeans.kmeans_quantize(SIFT_features, vocab)
    # map_to_bins = np.digitize(assignments, bins)
    # Hist = np.zeros(bins.shape)
    # for j in map_to_bins:
    #   Hist[j-1] += 1
    # Hist = Hist/np.linalg.norm(Hist)
    # feats.append(Hist)

  # print(Hist.shape)
  # print(assignments.shape)
  feats = np.array(feats)

  
    

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  # raise NotImplementedError('`get_bags_of_sifts` function in ' +
  #       '`student_code.py` needs to be implemented')

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return feats

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
    metric='euclidean'):
  """
  This function will predict the category for every test image by finding
  the training image with most similar features. Instead of 1 nearest
  neighbor, you can vote based on k nearest neighbors which will increase
  performance (although you need to pick a reasonable value for k).

  Useful functions:
  -   D = sklearn_pairwise.pairwise_distances(X, Y)
        computes the distance matrix D between all pairs of rows in X and Y.
          -  X is a N x d numpy array of d-dimensional features arranged along
          N rows
          -  Y is a M x d numpy array of d-dimensional features arranged along
          N rows
          -  D is a N x M numpy array where d(i, j) is the distance between row
          i of X and row j of Y

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating
          the ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  -   metric: (optional) metric to be used for nearest neighbor.
          Can be used to select different distance functions. The default
          metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
          well for histograms

  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """
  test_labels = []

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  N = train_image_feats.shape[0]
  M = test_image_feats.shape[0]

  D = sklearn_pairwise.pairwise_distances(train_image_feats, test_image_feats)
  row_min = 0
  for i in range(0, M):
    row_min = np.min(D[i,:])
    for j in range(0, N):
      if (D[i,j] == row_min):
        test_labels.append(train_labels[j])
        break

  # raise NotImplementedError('`nearest_neighbor_classify` function in ' +
  #       '`student_code.py` needs to be implemented')

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  return  test_labels

def svm_classify(train_image_feats, train_labels, test_image_feats):
  """
  This function will train a linear SVM for every category (i.e. one vs all)
  and then use the learned linear classifiers to predict the category of
  every test image. Every test feature will be evaluated with all 15 SVMs
  and the most confident SVM will "win". Confidence, or distance from the
  margin, is W*X + B where '*' is the inner product or dot product and W and
  B are the learned hyperplane parameters.

  Useful functions:
  -   sklearn LinearSVC
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
  -   svm.fit(X, y)
  -   set(l)

  Args:
  -   train_image_feats:  N x d numpy array, where d is the dimensionality of
          the feature representation
  -   train_labels: N element list, where each entry is a string indicating the
          ground truth category for each training image
  -   test_image_feats: M x d numpy array, where d is the dimensionality of the
          feature representation. You can assume N = M, unless you have changed
          the starter code
  Returns:
  -   test_labels: M element list, where each entry is a string indicating the
          predicted category for each testing image
  """
  # categories
  categories = list(set(train_labels))
  num_categories = len(categories)
  true_label = np.zeros(len(train_labels))
  W = []
  C = []

  # construct 1 vs all SVMs for each category
  svms = {cat: LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=5) for cat in categories}

  for i in range(num_categories):
    for j in range(len(train_labels)):
      true_label[j] = (categories[i]==train_labels[j])
    binary_label = np.ones(len(true_label))*-1
    for k in range(len(binary_label)):
      if(true_label[k]==1):
        binary_label[k] = 1

    svms[categories[i]].fit(train_image_feats,binary_label,sample_weight=None)
    # print(svms[categories[i]].coef_)

    W.append(svms[categories[i]].coef_)
    C.append(svms[categories[i]].intercept_)
  
  W = np.array(W)
  C = np.array(C)


  num_test_image_feats = test_image_feats.shape[0]
  test_labels = []

  for i in range(num_test_image_feats):
    confidence = []
    for j in range(num_categories):
      confidence.append(np.dot(W[j,0,:], test_image_feats[i, :]) + C[j, :])
    max_conf = max(confidence)
    for ind in range(num_categories):
      if(confidence[ind] == max_conf):
        test_labels.append(categories[ind])

  #############################################################################
  # TODO: YOUR CODE HERE                                                      #
  #############################################################################

  # raise NotImplementedError('`svm_classify` function in ' +
  #       '`student_code.py` needs to be implemented')

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return test_labels

def build_gmm(image_paths, vocab_size):
  bin_size = 8
  data = []

  for i in range(len(image_paths)):
    image = load_imgae_gray(image_paths[i])
    [locations, SIFT_features] = vlfeat.sift.dsift(image.astype('float32'),fast=True,step=15,bin=8)
    SIFT_features = SIFT_features.astype('float32')
    data = np.hstack(data, SIFT_features)

  [means, covariances, priors] = vlfeat.gmm.gmm(SIFT_features, vocab_size)
  stats = [means,covariances,priors]
  return stats


def build_gaussian_gmm(image_paths, vocab_size):
    ## Used to build gaussian gmm
  bin_size = 8

  level = 3
  data = []
  for i in range(len(image_paths)):
    for j in range(level):
      image = load_image_gray(image_paths[i])
      G_low = cv2.getGaussianKernel(9,2)
      filtered_image = cv2.filter2D(image,-1,G_low)
      resize_image = cv2.resize(filtered_image, 0.5^(j-1))
      [locations, SIFT_features] = vlfeat.sift.dsift(resize_image.astype('float32'),fast=True,step=15,bin=8)
      SIFT_features = SIFT_features.astype('float32')
      data = np.hstack(data, SIFT_features)

  [means, covariances, priors] = vlfeat.gmm.gmm(SIFT_features, vocab_size)

  stats = [means,covariances,priors]
  return stats

def build_spyramid_gmm(image_paths, vocab_size):

  level = 2

  data = []
  for i in range(len(image_paths)):
    image = load_image_gray(image_paths[i])
    W = image.shape[0]
    L = image.shape[1]
    [locations, SIFT_features_L0] = vlfeat.sift.dsift(image.astype('float32'),fast=True,step=15,bin=8)
    data = np.hstack(data, SIFT_features_L0)

    img_L1_1 = image[0: int16(W/2), 0:int16(L/2)]
    img_L1_2 = image[0: int16(W/2), int16(L/2):L]
    img_L1_3 = image[int16(W/2):W, 0:int16(L/2)]
    img_L1_4 = image[int16(W/2):W, int16(L/2):L]
    [locations, SIFT_features_L1_1] = vlfeat.sift.dsift(img_L1_1.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L1_2] = vlfeat.sift.dsift(img_L1_2.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L1_3] = vlfeat.sift.dsift(img_L1_3.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L1_4] = vlfeat.sift.dsift(img_L1_4.astype('float32'),fast=True,step=15,bin=8)
    data = np.hstack(data, SIFT_features_L1_1, SIFT_features_L1_2,SIFT_features_L1_3,SIFT_features_L1_4)

    img_L2_1=img[0:int16(W/4),0:int16(L/4)]
    img_L2_2=img[0:int16(W/4),int16(L/4):int16(L/2)]
    img_L2_3=img[0:int16(W/4),int16(L/2):int16(3*L/4)]
    img_L2_4=img[int16(W/4):int16(W/2),int16(3*L/4):L]

    img_L2_5=img[int16(W/4):int16(W/2),0:int16(L/4)]
    img_L2_6=img[int16(W/4):int16(W/2),int16(L/4):int16(L/2)]
    img_L2_7=img[int16(W/4):int16(W/2),int16(L/2):int16(3*L/4)]
    img_L2_8=img[int16(W/4):int16(W/2),int16(3*L/4):L]

    img_L2_9=img[int16(W/2):int16(3*W/4),0:int16(L/4)]
    img_L2_10=img[int16(W/2):int16(3*W/4),int16(L/4):int16(L/2)]
    img_L2_11=img[int16(W/2):int16(3*W/4),int16(L/2):int16(3*L/4)]
    img_L2_12=img[int16(W/2):int16(3*W/4),int16(3*L/4):L]

    img_L2_13=img[int16(3*W/4):W,0:int16(L/4)]
    img_L2_14=img[int16(3*W/4):W,int16(L/4):int16(L/2)]
    img_L2_15=img[int16(3*W/4):W,int16(L/2):int16(3*L/4)]
    img_L2_16=img[int16(3*W/4):W,int16(3*L/4):L]

    [locations, SIFT_features_L2_1] = vlfeat.sift.dsift(img_L2_1.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L2_2] = vlfeat.sift.dsift(img_L2_2.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L2_3] = vlfeat.sift.dsift(img_L2_3.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L2_4] = vlfeat.sift.dsift(img_L2_4.astype('float32'),fast=True,step=15,bin=8)
    data = np.hstack(data, SIFT_features_L2_1, SIFT_features_L2_2,SIFT_features_L2_3,SIFT_features_L2_4)

    [locations, SIFT_features_L2_5] = vlfeat.sift.dsift(img_L2_5.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L2_6] = vlfeat.sift.dsift(img_L2_6.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L2_7] = vlfeat.sift.dsift(img_L2_7.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L2_8] = vlfeat.sift.dsift(img_L2_8.astype('float32'),fast=True,step=15,bin=8)
    data = np.hstack(data, SIFT_features_L2_5, SIFT_features_L2_6,SIFT_features_L2_7,SIFT_features_L2_8)

    [locations, SIFT_features_L2_9] = vlfeat.sift.dsift(img_L2_9.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L2_10] = vlfeat.sift.dsift(img_L2_10.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L2_11] = vlfeat.sift.dsift(img_L2_11.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L2_12] = vlfeat.sift.dsift(img_L2_12.astype('float32'),fast=True,step=15,bin=8)
    data = np.hstack(data, SIFT_features_L2_9, SIFT_features_L2_10,SIFT_features_L2_11,SIFT_features_L2_12)

    [locations, SIFT_features_L2_13] = vlfeat.sift.dsift(img_L2_13.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L2_14] = vlfeat.sift.dsift(img_L2_14.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L2_15] = vlfeat.sift.dsift(img_L2_15.astype('float32'),fast=True,step=15,bin=8)
    [locations, SIFT_features_L2_16] = vlfeat.sift.dsift(img_L2_16.astype('float32'),fast=True,step=15,bin=8)
    data = np.hstack(data, SIFT_features_L2_13, SIFT_features_L2_14,SIFT_features_L2_15,SIFT_features_L2_16)


  [means, covariances, priors] = vlfeat.gmm.gmm(SIFT_features, vocab_size)

  stats = [means,covariances,priors]

  return stats




def get_fisher_encoding(image_paths, stat_filename):
  with open(stat_filename, 'rb') as f:
    stats = pickle.load(f)

  means = stats[:,0:128]
  covariances = stats[:, 128:256]
  priors = stats[:,257]

  feats = []
  for i in range(len(image_paths)):
    image = load_image_gray(image_paths[i])
    [locations, SIFT_features] = vlfeat.sift.dsift(image.astype('float32'),fast=True,step=5,bin=8)
    result = vlfeats.fisher.fisher(SIFT_features.astype('float32'), means, covariances, priors, Improved = True)
    feats.append(result)

  feats = np.array(feats)

  return feats


def get_spyramid_fisher_encoding(image_paths, stat_filename):
  with open(stat_filename, 'rb') as f:
    stats = pickle.load(f)

  means = stats[:,0:128]
  covariances = stats[:, 128:256]
  priors = stats[:,257]

  feats = []
  feats_L0 = []
  feats_L1_1 = []
  feats_L1_2 = []
  feats_L1_3 = []
  feats_L1_4 = []
  feats_L2_1 = []
  feats_L2_2 = []
  feats_L2_3 = []
  feats_L2_4 = []
  feats_L2_5 = []
  feats_L2_6 = []
  feats_L2_7 = []
  feats_L2_8 = []
  feats_L2_9 = []
  feats_L2_10 = []
  feats_L2_11 = []
  feats_L2_12 = []
  feats_L2_13 = []
  feats_L2_14 = []
  feats_L2_15 = []
  feats_L2_16 = []

  for i in range(len(image_paths)):
      ##level0
    image = load_image_gray(image_paths[i])
    W = image.shape[0]
    L = image.shape[1]
    [locations, SIFT_features_L0] = vlfeat.sift.dsift(image.astype('float32'),fast=True,step=5,bin=8)
    result0 = vlfeats.fisher.fisher(SIFT_features_L0.astype('float32'), means, covariances, priors, Improved = True)
    feats_L0.append(result0)

      ##level1
    img_L1_1 = image[0: int16(W/2), 0:int16(L/2)]
    img_L1_2 = image[0: int16(W/2), int16(L/2):L]
    img_L1_3 = image[int16(W/2):W, 0:int16(L/2)]
    img_L1_4 = image[int16(W/2):W, int16(L/2):L]
    [locations, SIFT_features_L1_1] = vlfeat.sift.dsift(img_L1_1.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L1_2] = vlfeat.sift.dsift(img_L1_2.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L1_3] = vlfeat.sift.dsift(img_L1_3.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L1_4] = vlfeat.sift.dsift(img_L1_4.astype('float32'),fast=True,step=5,bin=8)
    result1_1 = vlfeats.fisher.fisher(SIFT_features_L1_1.astype('float32'), means, covariances, priors, Improved = True)
    result1_2 = vlfeats.fisher.fisher(SIFT_features_L1_2.astype('float32'), means, covariances, priors, Improved = True)
    result1_3 = vlfeats.fisher.fisher(SIFT_features_L1_3.astype('float32'), means, covariances, priors, Improved = True)
    result1_4 = vlfeats.fisher.fisher(SIFT_features_L1_4.astype('float32'), means, covariances, priors, Improved = True)
    feats_L1_1.append(result1_1)
    feats_L1_2.append(result1_2)
    feats_L1_3.append(result1_3)
    feats_L1_4.append(result1_4)

      ##level2
    img_L2_1=img[0:int16(W/4),0:int16(L/4)]
    img_L2_2=img[0:int16(W/4),int16(L/4):int16(L/2)]
    img_L2_3=img[0:int16(W/4),int16(L/2):int16(3*L/4)]
    img_L2_4=img[int16(W/4):int16(W/2),int16(3*L/4):L]

    img_L2_5=img[int16(W/4):int16(W/2),0:int16(L/4)]
    img_L2_6=img[int16(W/4):int16(W/2),int16(L/4):int16(L/2)]
    img_L2_7=img[int16(W/4):int16(W/2),int16(L/2):int16(3*L/4)]
    img_L2_8=img[int16(W/4):int16(W/2),int16(3*L/4):L]

    img_L2_9=img[int16(W/2):int16(3*W/4),0:int16(L/4)]
    img_L2_10=img[int16(W/2):int16(3*W/4),int16(L/4):int16(L/2)]
    img_L2_11=img[int16(W/2):int16(3*W/4),int16(L/2):int16(3*L/4)]
    img_L2_12=img[int16(W/2):int16(3*W/4),int16(3*L/4):L]

    img_L2_13=img[int16(3*W/4):W,0:int16(L/4)]
    img_L2_14=img[int16(3*W/4):W,int16(L/4):int16(L/2)]
    img_L2_15=img[int16(3*W/4):W,int16(L/2):int16(3*L/4)]
    img_L2_16=img[int16(3*W/4):W,int16(3*L/4):L]

    [locations, SIFT_features_L2_1] = vlfeat.sift.dsift(img_L2_1.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L2_2] = vlfeat.sift.dsift(img_L2_2.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L2_3] = vlfeat.sift.dsift(img_L2_3.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L2_4] = vlfeat.sift.dsift(img_L2_4.astype('float32'),fast=True,step=5,bin=8)

    [locations, SIFT_features_L2_5] = vlfeat.sift.dsift(img_L2_5.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L2_6] = vlfeat.sift.dsift(img_L2_6.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L2_7] = vlfeat.sift.dsift(img_L2_7.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L2_8] = vlfeat.sift.dsift(img_L2_8.astype('float32'),fast=True,step=5,bin=8)

    [locations, SIFT_features_L2_9] = vlfeat.sift.dsift(img_L2_9.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L2_10] = vlfeat.sift.dsift(img_L2_10.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L2_11] = vlfeat.sift.dsift(img_L2_11.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L2_12] = vlfeat.sift.dsift(img_L2_12.astype('float32'),fast=True,step=5,bin=8)

    [locations, SIFT_features_L2_13] = vlfeat.sift.dsift(img_L2_13.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L2_14] = vlfeat.sift.dsift(img_L2_14.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L2_15] = vlfeat.sift.dsift(img_L2_15.astype('float32'),fast=True,step=5,bin=8)
    [locations, SIFT_features_L2_16] = vlfeat.sift.dsift(img_L2_16.astype('float32'),fast=True,step=5,bin=8)

    result2_1 = vlfeats.fisher.fisher(SIFT_features_L2_1.astype('float32'), means, covariances, priors, Improved = True)
    result2_2 = vlfeats.fisher.fisher(SIFT_features_L2_2.astype('float32'), means, covariances, priors, Improved = True)
    result2_3 = vlfeats.fisher.fisher(SIFT_features_L2_3.astype('float32'), means, covariances, priors, Improved = True)
    result2_4 = vlfeats.fisher.fisher(SIFT_features_L2_4.astype('float32'), means, covariances, priors, Improved = True)

    result2_5 = vlfeats.fisher.fisher(SIFT_features_L2_5.astype('float32'), means, covariances, priors, Improved = True)
    result2_6 = vlfeats.fisher.fisher(SIFT_features_L2_6.astype('float32'), means, covariances, priors, Improved = True)
    result2_7 = vlfeats.fisher.fisher(SIFT_features_L2_7.astype('float32'), means, covariances, priors, Improved = True)
    result2_8 = vlfeats.fisher.fisher(SIFT_features_L2_8.astype('float32'), means, covariances, priors, Improved = True)

    result2_9 = vlfeats.fisher.fisher(SIFT_features_L2_9.astype('float32'), means, covariances, priors, Improved = True)
    result2_10 = vlfeats.fisher.fisher(SIFT_features_L2_10.astype('float32'), means, covariances, priors, Improved = True)
    result2_11 = vlfeats.fisher.fisher(SIFT_features_L2_11.astype('float32'), means, covariances, priors, Improved = True)
    result2_12 = vlfeats.fisher.fisher(SIFT_features_L2_12.astype('float32'), means, covariances, priors, Improved = True)

    result2_13 = vlfeats.fisher.fisher(SIFT_features_L2_13.astype('float32'), means, covariances, priors, Improved = True)
    result2_14 = vlfeats.fisher.fisher(SIFT_features_L2_14.astype('float32'), means, covariances, priors, Improved = True)
    result2_15 = vlfeats.fisher.fisher(SIFT_features_L2_15.astype('float32'), means, covariances, priors, Improved = True)
    result2_16 = vlfeats.fisher.fisher(SIFT_features_L2_16.astype('float32'), means, covariances, priors, Improved = True)

    feats_L2_1.append(result2_1)
    feats_L2_2.append(result2_2)
    feats_L2_3.append(result2_3)
    feats_L2_4.append(result2_4)
    feats_L2_5.append(result2_5)
    feats_L2_6.append(result2_6)
    feats_L2_7.append(result2_7)
    feats_L2_8.append(result2_8)
    feats_L2_9.append(result2_9)
    feats_L2_10.append(result2_10)
    feats_L2_11.append(result2_11)
    feats_L2_12.append(result2_12)
    feats_L2_13.append(result2_13)
    feats_L2_14.append(result2_14)
    feats_L2_15.append(result2_15)
    feats_L2_16.append(result2_16)

    feats = np.append(feats,feats_L0,feats_L1_1,feats_L1_2,feats_L1_3,feats_L1_4,feats_L1_5,feats_L2_1,feats_L2_2,feats_L2_3,feats_L2_4,feats_L2_5,feats_L2_6,feats_L2_7,feats_L2_8,feats_L2_9,feats_L2_10,feats_L2_11,feats_L2_12,feats_L2_13,feats_L2_14,feats_L2_15,feats_L2_16)

  feats = np.array(feats)

  return feats






    