import numpy as np

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###
  filter_size = filter.shape

  ##Create pad image
  pad_size = [int(np.floor(x/2)) for x in filter_size]
  filtered_image = np.empty(image.shape)

  pad_image = np.lib.pad(image, ((pad_size[0],),(pad_size[1],),(0,)), 'symmetric')
  #pad_image = np.lib.pad(image, ((pad_size[0],),(pad_size[1],),(0,)), 'constant', constant_values=0)

  if(image.shape[2] == 1):
    for i in range(0, filtered_image.shape[0]):
      for j in range(0, filtered_image.shape[1]):
        window = pad_image[i:i+filter_size[0], j:j+filter_size[1]]
        filtered_image[i][j] = np.sum(np.multiply(window, filter))

    return filtered_image

  else:
    filter = filter.reshape((filter_size[0],filter_size[1],1))
    for i in range(0, filtered_image.shape[0]):
      for j in range(0, filtered_image.shape[1]):
        window = pad_image[i:i+filter_size[0], j:j+filter_size[1],:]
        filtered_image[i:i+1, j:j+1, :]= np.sum(np.multiply(window, filter), axis=(0,1))
    return filtered_image

  raise NotImplementedError('`my_imfilter` function in `student_code.py` ' +
    'needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###
  filter_size = filter.shape
  low_frequencies = my_imfilter(image1, filter)
  high_frequencies = image2 - my_imfilter(image2, filter)

  img = low_frequencies+high_frequencies

  hybrid_image = np.clip(img, 0,1)

  return low_frequencies, high_frequencies, hybrid_image

  raise NotImplementedError('`create_hybrid_image` function in ' + 
    '`student_code.py` needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################


