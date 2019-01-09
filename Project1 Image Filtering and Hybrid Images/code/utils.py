import cv2
import numpy as np

def vis_hybrid_image(hybrid_image):
  scales = 5
  scale_factor = 0.5
  padding = 5
  original_height = hybrid_image.shape[0]
  num_colors = 1 if hybrid_image.ndim == 2 else 3

  output = np.copy(hybrid_image)
  cur_image = np.copy(hybrid_image)
  for scale in range(2, scales+1):
    # add padding
    output = np.hstack((output, np.ones((original_height, padding, num_colors),
                                        dtype=np.float32)))

    # downsample image
    cur_image = cv2.resize(cur_image, (0, 0), fx=scale_factor, fy=scale_factor)

    # pad the top to append to the output
    pad = np.ones((original_height-cur_image.shape[0], cur_image.shape[1],
                   num_colors), dtype=np.float32)
    tmp = np.vstack((pad, cur_image))
    output = np.hstack((output, tmp))

  return output

def im2single(im):
  im = im.astype(np.float32) / 255
  return im

def single2im(im):
  im *= 255
  im = im.astype(np.uint8)
  return im

def load_image(path):
  return im2single(cv2.imread(path))[:, :, ::-1]

def save_image(path, im):
  return cv2.imwrite(path, single2im(im.copy())[:, :, ::-1])
