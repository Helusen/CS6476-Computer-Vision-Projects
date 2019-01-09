import numpy as np
import cv2
import os.path as osp
from glob import glob
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
from skimage.feature import hog
from skimage import exposure, draw
import matplotlib.lines as mlines
plt.rcParams.update({'figure.max_open_warning': 0})

def load_image(path):
  im = cv2.imread(path)
  im = im[:, :, ::-1]  # BGR -> RGB
  im = im.astype(np.float32)  # for vlfeat functions
  return im

def load_image_gray(path):
  im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
  im = im.astype(np.float32)  # for vlfeat functions
  return im

def report_accuracy(confidences, label_vector):
  """
  Calculates various accuracy metrics on the given predictions
  :param confidences: 1D numpy array holding predicted confidence scores
  :param label_vector: 1D numpy array holding ground truth labels (same size
  as confidences
  :return: tp_rate, fp_rate, tn_rate, fn_rate
  """
  preds = confidences.copy()
  preds[preds >= 0] = 1
  preds[preds <  0] = -1

  tp = np.logical_and(preds > 0, preds==label_vector)
  fp = np.logical_and(preds > 0, preds!=label_vector)
  tn = np.logical_and(preds < 0, preds==label_vector)
  fn = np.logical_and(preds < 0, preds!=label_vector)

  N = len(label_vector)

  tp_rate = sum(tp) / (sum(tp)+sum(fn)) * 100
  fp_rate = sum(fp) / (sum(fp)+sum(tn)) * 100
  tn_rate = 100 - fp_rate
  fn_rate = 100 - tp_rate
  accuracy = (sum(tp)+sum(tn)) / N * 100

  print('Accuracy = {:4.3f}%\n'
        'True Positive rate = {:4.3f}%\nFalse Positive rate = {:4.3f}%\n'
        'True Negative rate = {:4.3f}%\nFalse Negative rate = {:4.3f}%'.
    format(accuracy, tp_rate, fp_rate, tn_rate, fn_rate))

  return tp_rate, fp_rate, tn_rate, fn_rate

def non_max_suppression_bbox(bboxes, confidences, img_size, verbose=False):
  """
  high confidence detections suppress all overlapping detections (including
  detections at other scales). Detections can partially overlap, but the
  center of one detection can not be within another detection.

  :param bboxes: Nx4 numpy array, where N is the number of bounding boxes. Each
  row is [xmin, ymin, xmax, ymax]
  :param confidences: size (N, ) numpy array, holding the final confidence of
  each detection
  :param img_size: the [height, width] of the image
  :param verbose: boolean
  :return: size (N, ) numpy logical array. Element i indicates if the i'th
  bounding box survives non-maximum suppression.
  """
  # truncate the bounding boxes to image dimensions
  bboxes[:, 2] = np.minimum(bboxes[:, 2], img_size[1])
  bboxes[:, 3] = np.minimum(bboxes[:, 3], img_size[0])

  # higher confidence detections get priority
  order = np.argsort(-confidences)
  confidences = confidences[order]
  bboxes = bboxes[order]

  # output indicator vector
  is_valid_bbox = np.asarray([False]*len(confidences))

  # overlap threshold above which the less confident detection is suppressed
  overlap_thresh = 0.3

  for i in range(len(confidences)):
    cur_bb = bboxes[i]
    cur_bb_is_valid = True

    for j in np.where(is_valid_bbox)[0]:
      prev_bb = bboxes[j]
      bi = [max(cur_bb[0], prev_bb[0]), max(cur_bb[1], prev_bb[1]),
            min(cur_bb[2], prev_bb[2]), min(cur_bb[3], prev_bb[3])]
      iw = bi[2] - bi[0] + 1
      ih = bi[3] - bi[1] + 1
      if (iw > 0) and (ih > 0):
        # overlap = area of intersection / area of union
        ua = (cur_bb[2]-cur_bb[0]+1) * (cur_bb[3]-cur_bb[1]+1) +\
             (prev_bb[2]-prev_bb[0]+1) * (prev_bb[3]-prev_bb[1]+1) -\
             iw*ih
        ov = (iw*ih) / ua

        if ov > overlap_thresh:
          cur_bb_is_valid = False

        # special case: center coordinate of current bbox is inside the previous
        # bbox
        cx = (cur_bb[0] + cur_bb[2]) / 2
        cy = (cur_bb[1] + cur_bb[3]) / 2
        if (cx > prev_bb[0]) and (cx < prev_bb[2]) and (cy > prev_bb[1]) and\
          (cy < prev_bb[3]):
          cur_bb_is_valid = False

        if verbose:
          print('Detection {:d}, bbox = [{:d}, {:d}, {:d}, {:d}], {:f} overlap '
                'with detection {:d} [{:d}, {:d}, {:d}, {:d}]'
            .format(i, cur_bb[0], cur_bb[1], cur_bb[2], cur_bb[3], ov, j,
          prev_bb[0], prev_bb[1], prev_bb[2], prev_bb[3]))

        if not cur_bb_is_valid:
          break

    is_valid_bbox[i] = cur_bb_is_valid

  # return back to the original order
  order = np.argsort(order)
  is_valid_bbox = is_valid_bbox[order]

  return is_valid_bbox

def voc_ap(rec, prec):
  mrec = np.hstack((0, rec, 1))
  mpre = np.hstack((0, prec, 0))

  for i in reversed(range(len(mpre)-1)):
    mpre[i] = max(mpre[i], mpre[i+1])

  i = np.where(mrec[1:] != mrec[:-1])[0] + 1
  ap = sum((mrec[i] - mrec[i-1]) * mpre[i])
  return ap

def visualize_hog(svm, feature_params):
  win_size = feature_params.get('template_size', 36)
  cell_size = feature_params.get('hog_cell_size', 6)
  n_cell = np.ceil(win_size/cell_size).astype('int')

  test_feat=svm.coef_ - np.min(svm.coef_)
  test_feat =np.reshape(test_feat,[n_cell,n_cell,31])

  radius=22
  orientations=9

  cx,cy = 48,48
  sy,sx =cy*n_cell,cx*n_cell

  n_cellsx = n_cell
  n_cellsy = n_cell

  orientation_histogram=test_feat

  orientations_arr = np.arange(orientations)
  dx_arr = radius * np.cos(orientations_arr / orientations * np.pi)
  dy_arr = radius * np.sin(orientations_arr / orientations * np.pi)
  hog_image = np.zeros((sy, sx), dtype=float)


  for x in range(n_cellsx):
      for y in range(n_cellsy):
          for o, dx, dy in zip(orientations_arr, dx_arr, dy_arr):
              
              centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
              wt = (orientation_histogram[y, x, 18+o])*2.5
              
              xmin=int(centre[0] - dx)
              xmax=int(centre[0] + dx)
              ymin=int(centre[1] + dy)
              ymax=int(centre[1] - dy)
                       
              rr, cc = draw.line(xmin,
                                 ymin,
                                 xmax,
                                 ymax)
              
              hog_image[rr, cc] = np.maximum(hog_image[rr,cc],wt)
              hog_image[rr+1, cc] = np.maximum(hog_image[rr+1,cc],wt)
              hog_image[rr, cc+1] = np.maximum(hog_image[rr,cc+1],wt)
              hog_image[rr-1, cc] = np.maximum(hog_image[rr-1,cc],wt)
              hog_image[rr, cc-1] = np.maximum(hog_image[rr,cc-1],wt)
              

  hog_image_2 = hog_image.copy()
              
  hog_image=hog_image**3/np.max(hog_image**3)
  hog_image=hog_image*255

              
  hog_image_2[hog_image_2==0]=0.5*np.max(hog_image_2)
  hog_image_2=hog_image_2/np.max(hog_image_2)
  hog_image_2=hog_image_2*255

  plt.figure(figsize=(4,4))
  plt.imshow((hog_image).astype("uint8"),cmap="gray")
  plt.axis("off")

  plt.figure(figsize=(4,4))
  plt.imshow((hog_image_2).astype("uint8"),cmap="gray")
  plt.axis("off")


def evaluate_detections(bboxes, confidences, image_ids, label_path, draw=True):
  """
  :param bboxes:
  :param confidences:
  :param image_ids:
  :param label_path:
  :param draw:
  :return:
  """
  gt_ids = []
  gt_bboxes = []
  with open(label_path, 'r') as f:
    for line in f:
      gt_id, xmin, ymin, xmax, ymax = line.split(' ')
      gt_ids.append(gt_id)
      gt_bboxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
  gt_bboxes = np.vstack(gt_bboxes)

  npos = len(gt_ids)
  gt_isclaimed = np.asarray([False]*len(gt_ids))


  # sort detections by decreasing confidence
  order = np.argsort(-confidences)
  confidences = confidences[order]
  image_ids = [image_ids[i] for i in order]
  bboxes = bboxes[order]

  # assign detections to GT objects
  nd = len(confidences)
  tp = np.asarray([False]*nd)
  fp = np.asarray([False]*nd)
  duplicate_detections = np.asarray([False]*nd)

  for d in range(nd):
    cur_gt_ids = [i for i, gt_id in enumerate(gt_ids) if gt_id == image_ids[d]]

    bb = bboxes[d]
    ovmax = -float('inf')

    for j in cur_gt_ids:
      bbgt = gt_bboxes[j]
      bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]),
            min(bb[3], bbgt[3])]
      iw = bi[2] - bi[0] + 1
      ih = bi[3] - bi[1] + 1

      if (iw > 0) and (ih > 0):
        ua = (bb[2]-bb[0]+1) * (bb[3]-bb[1]+1) +\
             (bbgt[2]-bbgt[0]+1) * (bbgt[3]-bbgt[1]+1) -\
             iw*ih
        ov = iw*ih / ua
        if ov > ovmax:
          ovmax = ov
          jmax = j

    if ovmax >= 0.3:
      if not gt_isclaimed[jmax]:
        tp[d] = True
        gt_isclaimed[jmax] = True
      else:
        fp[d] = True
        duplicate_detections[d] = True
    else:
      fp[d] = True

  cum_fp = np.cumsum(fp)
  cum_tp = np.cumsum(tp)
  rec = cum_tp / npos
  prec = cum_tp / (cum_tp + cum_fp)
  ap = voc_ap(rec, prec)

  if draw:
    plt.figure()
    plt.plot(rec, prec, '-')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Average precision = {:4.3f}'.format(ap))

    plt.figure()
    plt.plot(cum_fp, rec, '-')
    plt.xlim(0, 300)
    plt.ylim(0, 1)
    plt.xlabel('False Positives')
    plt.ylabel('Number of correct detections (recall)')
    plt.title('This figure is meant to match Fig. 6 in Viola-Jones')

  order = np.argsort(order)
  tp = tp[order]
  fp = fp[order]
  duplicate_detections = duplicate_detections[order]

  return gt_ids, gt_bboxes, gt_isclaimed, tp, fp, duplicate_detections

def visualize_detections_by_image(bboxes, confidences, image_ids, tp, fp,
    test_scn_path, label_filename, onlytp=False):
  """
  Visuaize the detection bounding boxes and ground truth on images
  :param bboxes: N x 4 numpy matrix, where N is the number of detections. Each
  row is [xmin, ymin, xmax, ymax]
  :param confidences: size (N, ) numpy array of detection confidences
  :param image_ids: N-element list of image names for each detection
  :param tp: size (N, ) numpy array of true positive indicator variables
  :param fp: size (N, ) numpy array of false positive indicator variables
  :param test_scn_path: path to directory holding test images (in .jpg format)
  :param label_filename: path to .txt file containing labels. Format is
  image_id xmin ymin xmax ymax for each row
  :param onlytp: show only true positives
  :return:
  """
  gt_ids = []
  gt_bboxes = []
  with open(label_filename, 'r') as f:
    for line in f:
      gt_id, xmin, ymin, xmax, ymax = line.split(' ')
      gt_ids.append(gt_id)
      gt_bboxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
  gt_bboxes = np.vstack(gt_bboxes)

  gt_file_list = list(set(gt_ids))

  for gt_file in gt_file_list:
    cur_test_image = load_image(osp.join(test_scn_path, gt_file))

    cur_gt_detections = [i for i, gt_id in enumerate(gt_ids) if gt_id == gt_file]
    cur_gt_bboxes = gt_bboxes[cur_gt_detections]

    cur_detections = [i for i, gt_id in enumerate(image_ids) if gt_id == gt_file]
    cur_bboxes = bboxes[cur_detections]
    cur_confidences = confidences[cur_detections]
    cur_tp = tp[cur_detections]
    cur_fp = fp[cur_detections]


    plt.figure()
    plt.imshow(cur_test_image.astype(np.uint8))

    for i, bb in enumerate(cur_bboxes):
      if cur_tp[i]:  # true positive
        plt.plot(bb[[0, 2, 2, 0, 0]], bb[[1, 1, 3, 3, 1]], 'g')
      elif cur_fp[i]:  # false positive
        if not onlytp:
          plt.plot(bb[[0, 2, 2, 0, 0]], bb[[1, 1, 3, 3, 1]], 'r')
      else:
        raise AssertionError

    for bb in cur_gt_bboxes:
      plt.plot(bb[[0, 2, 2, 0, 0]], bb[[1, 1, 3, 3, 1]], 'y')

    plt.axis("off")
    plt.title('{:s} (green=true pos, red=false pos, yellow=ground truth), '
              '{:d}/{:d} found'.format(gt_file, sum(cur_tp), len(cur_gt_bboxes)))

def visualize_detections_by_confidence(bboxes, confidences, image_ids,
    test_scn_path, label_filename, onlytp=False):
  """
  Visuaize the detection bounding boxes and ground truth on images, sorted by
  confidence
  :param bboxes: N x 4 numpy matrix, where N is the number of detections. Each
  row is [xmin, ymin, xmax, ymax]
  :param confidences: size (N, ) numpy array of detection confidences
  :param image_ids: N-element list of image names for each detection
  :param test_scn_path: path to directory holding test images (in .jpg format)
  :param label_filename: path to .txt file containing labels. Format is
  image_id xmin ymin xmax ymax for each row
  :param onlytp: show only true positives
  :return:
  """
  gt_ids = []
  gt_bboxes = []
  with open(label_filename, 'r') as f:
    for line in f:
      gt_id, xmin, ymin, xmax, ymax = line.split(' ')
      gt_ids.append(gt_id)
      gt_bboxes.append([float(xmin), float(ymin), float(xmax), float(ymax)])
  gt_bboxes = np.vstack(gt_bboxes)

  # sort detections by decreasing confidence
  order = np.argsort(-confidences)
  image_ids = [image_ids[i] for i in order]
  bboxes = bboxes[order]
  confidences = confidences[order]

  for d in range(len(confidences)):
    cur_gt_idxs = [i for i, gt_id in enumerate(gt_ids) if gt_id == image_ids[d]]
    bb = bboxes[d]
    ovmax = -float('inf')

    for j in cur_gt_idxs:
      bbgt = gt_bboxes[j]
      bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]),
            min(bb[3], bbgt[3])]
      iw = bi[2] - bi[0] + 1
      ih = bi[3] - bi[1] + 1

      if (iw > 0) and (ih > 0):
        ua = (bb[2]-bb[0]+1) * (bb[3]-bb[1]+1) +\
             (bbgt[2]-bbgt[0]+1) * (bbgt[3]-bbgt[1]+1) -\
             iw*ih
        ov = iw*ih / ua
        if ov > ovmax:
          ovmax = ov
          jmax = j

    if onlytp and ovmax < 0.3:
      continue

    im = load_image(osp.join(test_scn_path, image_ids[d]))
    plt.figure()
    plt.imshow(im.astype(np.uint8))
    if ovmax >= 0.3:
      bbgt = gt_bboxes[jmax]
      plt.plot(bbgt[[0, 2, 2, 0, 0]], bbgt[[1, 1, 3, 3, 1]], 'y')
      plt.plot(bb[[0, 2, 2, 0, 0]], bb[[1, 1, 3, 3, 1]], 'g')
    else:
      plt.plot(bb[[0, 2, 2, 0, 0]], bb[[1, 1, 3, 3, 1]], 'r')
    plt.title('Image {:s} [{:d}/{:d}], (green=true pos, red=false pos, '
              'yellow=ground truth)'.format(image_ids[d], d, len(confidences)))


def visualize_detections_by_image_no_gt(bboxes, confidences, image_ids,
    test_scn_path):
  """
  Visualize detection bounding boxes on images that don't have ground truth
  labels
  :param bboxes: N x 4 numpy matrix, where N is the number of detections. Each
  row is [xmin, ymin, xmax, ymax]
  :param confidences: size (N, ) numpy array of detection confidences
  :param image_ids: N-element list of image names for each detection
  :param test_scn_path: path to directory holding test images (in .jpg format)
  :return:
  """
  test_filenames = glob(osp.join(test_scn_path, '*.jpg'))

  for im_filename in test_filenames:
    test_id = im_filename.split('/')[-1]
    test_id = test_id.split('\\')[-1] # in case the file path use backslash
    cur_test_image = load_image(im_filename)
    cur_detections = [i for i, im_id in enumerate(image_ids) if im_id == test_id]
    cur_bboxes = bboxes[cur_detections]
    cur_confidences = confidences[cur_detections]

    plt.figure()
    plt.imshow(cur_test_image.astype(np.uint8))

    for bb in cur_bboxes:
      plt.plot(bb[[0, 2, 2, 0, 0]], bb[[1, 1, 3, 3, 1]], 'g')
    plt.title('{:s} green=detection'.format(test_id))


class PseudoSVM():

  def __init__(self,C=10,dim=1116):

    self.C = C
    self.coef_ = np.random.rand(dim,1)

  def decision_function(self,feats):

    return np.random.rand(len(feats))