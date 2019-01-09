import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.preprocessing import StandardScaler
import os.path as osp
import utils


def create_datasets(data_path, input_size, rgb=False):
  """
  This function creates and returns a training data loader and a
  testing/validation data loader. The dataloader should also perform some
  pre-processing steps on each of the datasets. Most of this function is
  implemented for you, you will only need to add a few additional lines.
  A data loader in pyTorch is a class inherited from the
  torch.utils.data.Dataset abstract class. In this project we will use the
  ImageFolder data loader. See
  http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder for
  details. Although you don't need to for this project, you can also create your
  own custom data loader by inheriting from the abstract class and implementing
  the __len__() and __getitem__() methods as described in
  http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
  As mentioned, the data loader should perform any necessary pre-processing
  steps on the data (images) and targets (labels). In pyTorch, this is done
  with 'transforms', which can be composed (chained together) as shown in
  http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#transforms.
  While that example implements its own transforms, for this project the
  built-in transforms in torchvision.transforms should suffice. See
  http://pytorch.org/docs/master/torchvision/transforms.html for the list of
  available built-in transforms.
  Args:
  - data_path: (string) Path to the directory that contains the 'test' and
      'train' data directories.
  - input_size: (w, h) Size of input image. The images will be resized to
      this size.
  - rgb: (boolean) Flag indicating if input images are RGB or grayscale. If
      False, images will be converted to grayscale.
  Returns:
  - train_dataloader: Dataloader for the training dataset.
  - test_dataloader: Dataloader for the testing/validation dataset.
  """
  train_data_path = osp.join(data_path, 'train')
  test_data_path = osp.join(data_path, 'test')
  # Below variables are provided for your convenience. You may or may not need
  # all of them.
  train_mean, train_std = utils.get_mean_std(train_data_path, input_size, rgb)
  test_mean, test_std = utils.get_mean_std(test_data_path, input_size, rgb)


  """ TRAIN DATA TRANSFORMS """
  train_data_tforms = []
  if not rgb:
    train_data_tforms.append(transforms.Grayscale())

  #######################################################################
  #                        TODO: YOUR CODE HERE                         #
  #######################################################################
  # TODO Add a transformation to you train_data_tforms that left-right mirrors
  # the image randomly. Which transonformation should you add?
  
  train_data_tforms.append(transforms.RandomHorizontalFlip(p=0.5)) # p: probability of the image being flipped. Default value is 0.5

  # Perform original transforms:
  train_data_tforms.append(transforms.Resize(size=max(input_size)))
  train_data_tforms.append(transforms.CenterCrop(size=input_size))

  # Do not move the position of the below line (leave it between the left-right
  # mirroring and normalization tranformations.
  train_data_tforms.append(transforms.ToTensor())

  # TODO Add a transformation to your train_data_tforms that normalizes the
  # tensor by subtracting mean and dividing by std. You may use train_mean,
  # test_mean, train_std, or test_std values that are already calculated for
  # you. Which mean and std should you use to normalize the data?
  train_data_tforms.append(transforms.Normalize(mean=train_mean, std=train_std))

  #######################################################################
  #                          END OF YOUR CODE                           #
  #######################################################################
  train_data_tforms = transforms.Compose(train_data_tforms)


  """ TEST/VALIDATION DATA TRANSFORMS """

  test_data_tforms = []
  test_data_tforms.append(transforms.Resize(size=max(input_size)))
  test_data_tforms.append(transforms.CenterCrop(size=input_size))
  if not rgb:
    test_data_tforms.append(transforms.Grayscale())
  test_data_tforms.append(transforms.ToTensor())
  #######################################################################
  #                        TODO: YOUR CODE HERE                         #
  #######################################################################
  # TODO Add a transformation to your test_data_tforms that normalizes the
  # tensor by subtracting mean and dividing by std. You may use train_mean,
  # test_mean, train_std, or test_std values that are already calculated for
  # you. Which mean and std should you use to normalize the data?


  # Normalize using mean and std from TRAIN data:
  test_data_tforms.append(transforms.Normalize(mean=train_mean, std=train_std))

  #######################################################################
  #                          END OF YOUR CODE                           #
  #######################################################################
  test_data_tforms = transforms.Compose(test_data_tforms)


  """ DATASET LOADERS """
  # Creating dataset loaders using the tranformations specified above.
  train_dset = datasets.ImageFolder(root=osp.join(data_path, 'train'),
                                    transform=train_data_tforms)
  test_dset = datasets.ImageFolder(root=osp.join(data_path, 'test'),
                                   transform=test_data_tforms)
  return train_dset, test_dset


class SimpleNet(nn.Module):
  """
  This class implements the network model needed for part 1. Network models in
  pyTorch are inherited from torch.nn.Module, only require implementing the
  __init__() and forward() methods. The backpropagation is handled automatically
  by pyTorch.
  The __init__() function defines the various operators needed for
  the forward pass e.g. conv, batch norm, fully connected, etc.
  The forward() defines how these blocks act on the input data to produce the
  network output. For hints on how to implement your network model, see the
  AlexNet example at
  https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
  """
  def __init__(self, num_classes, droprate=0.5, rgb=False, verbose=False):
    """
    This is where you set up and initialize your network. A basic network is
    already set up for you. You will need to add a few more layers to it as
    described. You can refer to https://pytorch.org/docs/stable/nn.html for
    documentation.
    Args:
    - num_classes: (int) Number of output classes.
    - droprate: (float) Droprate of the network (used for droppout).
    - rgb: (boolean) Flag indicating if input images are RGB or grayscale, used
      to set the number of input channels.
    - verbose: (boolean) If True a hook is registered to print the size of input
      to classifier everytime the forward function is called.
    """
    super(SimpleNet, self).__init__() # initialize the parent class, a must
    in_channels = 3 if rgb else 1

    """ NETWORK SETUP """
    #####################################################################
    #                       TODO: YOUR CODE HERE                        #
    #####################################################################
    # TODO modify the simple network
    # 1) add one dropout layer after the last relu layer.
    # 2) add more convolution, maxpool and relu layers.
    # 3) add one batch normalization layer after each convolution/linear layer
    #    except the last convolution/linear layer of the WHOLE model (meaning
    #    including the classifier).


    filter_nums = [15, 25] # DNN 1

    self.features = nn.Sequential(
      # shallow network:
      # nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=9,
      #   stride=1, padding=0, bias=False),
      # nn.MaxPool2d(kernel_size=7, stride=7, padding=0),
      # nn.ReLU(),

      # ------------------------------- DNN 1 mod -------------------------------------------
      nn.Conv2d(in_channels=in_channels, out_channels=filter_nums[0], kernel_size=9,
        stride=1, padding=0, bias=False),
      nn.ReLU(),  
      nn.BatchNorm2d(filter_nums[0]),
      nn.MaxPool2d(kernel_size=4, stride=2, padding=0),
        
      nn.Conv2d(in_channels=filter_nums[0], out_channels=filter_nums[1], kernel_size=5,
        stride=1, padding=0, bias=False),
      nn.ReLU(),
      nn.BatchNorm2d(filter_nums[1]),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
      nn.Dropout(p=0.5), # Added dropout layer
    )

    # Shallow network:
    # self.classifier = nn.Conv2d(in_channels=10, out_channels=num_classes,
    #   kernel_size=8, stride=1, padding=0)

    # DNN 1
    self.classifier = nn.Conv2d(in_channels=filter_nums[-1], out_channels=num_classes,
      kernel_size=11, stride=1, padding=0)

    #####################################################################
    #                         END OF YOUR CODE                          #
    #####################################################################

    """ NETWORK INITIALIZATION """
    for name, m in self.named_modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # Initializing weights with randomly sampled numbers from a normal
        # distribution.
        m.weight.data.normal_(0, 1)
        m.weight.data.mul_(1e-2)
        if m.bias is not None:
          # Initializing biases with zeros.
          nn.init.constant_(m.bias.data, 0)
      elif isinstance(m, nn.BatchNorm2d):
        #################################################################
        #                     TODO: YOUR CODE HERE                      #
        #################################################################
        nn.init.constant_(m.weight.data, 1)
        if m.bias is not None:
          # Initializing biases with zeros.
          nn.init.constant_(m.bias.data, 0)
          # print("with proper Initializing batch norm")

        #################################################################
        #                       END OF YOUR CODE                        #
        #################################################################

    if verbose:
      # Hook that prints the size of input to classifier everytime the forward
      # function is called.
      self.classifier.register_forward_hook(utils.print_input_size_hook)

  def forward(self, x):
    """
    Forward step of the network.
    Args:
    - x: input data.
    Returns:
    - x: output of the classifier.
    """
    x = self.features(x)
    x = self.classifier(x)
    return x.squeeze()

def custom_part1_trainer(model):
    # return a dict that contains your customized learning settings.
    pass
    return None
    
def create_part2_model(model, num_classes):
  """
  Take the passed in model and prepare it for finetuning by following the
  instructions.
  Args:
  - model: The model you need to prepare for finetuning. For the purposes of
    this project, you will pass in AlexNet.
  - num_classes: number of classes the model should output.
  Returns:
  - model: The model ready to be fine tuned.
  """
  # Getting all layers from the input model's classifier.
  new_classifier = list(model.classifier.children())
  new_classifier = new_classifier[:-1]
  #######################################################################
  #                        TODO: YOUR CODE HERE                         #
  #######################################################################
  # TODO modify the classifier of the model for finetuning. new_classifier is
  # now a list containing the layers of the classifier network, the last element
  # being the last layer of the classifier.
  # 1) Create a linear layer with correct in_features and out_features. What
  #    should these values be?
  # 2) Initialize the weights and the bias in the new linear layer. Look at how
  #    is the linear layer initialized in SimpleNetPart1.
  # 3) Append your new layer to your new_classifier.

  layer = nn.Linear(4096, num_classes)
  
  """ NETWORK INITIALIZATION """
  layer.weight.data.normal_(0, 1)
  layer.weight.data.mul_(1e-2)
  if layer.bias is not None:
          # Initializing biases with zeros.
    nn.init.constant_(layer.bias.data, 0)

  new_classifier.append(layer)
  
  #######################################################################
  #                          END OF YOUR CODE                           #
  #######################################################################
  # Connecting all layers to form a new classifier.
  model.classifier = nn.Sequential(*new_classifier)

  return model
  
def custom_part2_trainer(model):
    # return a dict that contains your customized learning settings.
    pass
    return None