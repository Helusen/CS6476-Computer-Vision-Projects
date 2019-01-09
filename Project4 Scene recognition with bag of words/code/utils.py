import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os.path as osp
from glob import glob
from random import shuffle

def im2single(im):
  im = im.astype(np.float32) / 255
  return im

def single2im(im):
  im *= 255
  im = im.astype(np.uint8)
  return im

def load_image(path):
  return im2single(cv2.imread(path))[:, :, ::-1]

def load_image_gray(path):
  img = load_image(path)
  return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def get_image_paths(data_path, categories, num_train_per_cat=100, fmt='jpg'):
  """
  This function returns lists containing the file path for each train
  and test image, as well as listss with the label of each train and
  test image. By default all four of these arrays will have 1500
  elements where each element is a string.
  :param data_path: path to the 'test' and 'train' directories
  :param categories: list of category names
  :param num_train_per_cat: max number of training images to use (per category)
  :param fmt: file extension of the images
  :return: lists: train_image_paths, test_image_paths, train_labels, test_labels
  """
  train_image_paths = []
  test_image_paths = []
  train_labels = []
  test_labels = []

  for cat in categories:
    # train
    pth = osp.join(data_path, 'train', cat, '*.{:s}'.format(fmt))
    pth = glob(pth)
    shuffle(pth)
    pth = pth[:num_train_per_cat]
    train_image_paths.extend(pth)
    train_labels.extend([cat]*len(pth))

    # test
    pth = osp.join(data_path, 'test', cat, '*.{:s}'.format(fmt))
    pth = glob(pth)
    shuffle(pth)
    pth = pth[:num_train_per_cat]
    test_image_paths.extend(pth)
    test_labels.extend([cat]*len(pth))

  return train_image_paths, test_image_paths, train_labels, test_labels

def show_results(train_image_paths, test_image_paths, train_labels, test_labels,
    categories, abbr_categories, predicted_categories):
  """
  shows the results
  :param train_image_paths:
  :param test_image_paths:
  :param train_labels:
  :param test_labels:
  :param categories:
  :param abbr_categories:
  :param predicted_categories:
  :return:
  """
  cat2idx = {cat: idx for idx, cat in enumerate(categories)}

  print(len(predicted_categories))

  # confusion matrix
  y_true = [cat2idx[cat] for cat in test_labels]
  y_pred = [cat2idx[cat] for cat in predicted_categories]
  print(len(y_true))
  print(len(y_pred))
  cm = confusion_matrix(y_true, y_pred)
  cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
  acc = np.mean(np.diag(cm))
  plt.figure()
  plt.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('jet'))
  plt.title('Confusion matrix. Mean of diagonal = {:4.2f}%'.format(acc*100))
  tick_marks = np.arange(len(categories))
  plt.tight_layout()
  plt.xticks(tick_marks, abbr_categories, rotation=45)
  plt.yticks(tick_marks, categories)
