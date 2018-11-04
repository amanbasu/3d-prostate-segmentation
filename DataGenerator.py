import glob
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

class DataGenerator(object):
    
  def __init__(self, data_dir='', image_filename='', label_filename='', transforms=None, train=False):
    self.data_dir = data_dir
    self.image_filename = image_filename
    self.label_filename = label_filename
    self.transforms = transforms
    self.train = train

  def get_dataset(self):
    image_paths = []
    label_paths = []
    for case in glob.glob(self.data_dir+'/Case*_segmentation.mhd'):
      image_paths.append(case[:-17]+'.mhd')
      label_paths.append(case)

    dataset = tf.data.Dataset.from_tensor_slices((image_paths,label_paths))

    dataset = dataset.map(lambda image_path, label_path: tuple(tf.py_func(
        self.input_parser, [image_path, label_path], [tf.float32,tf.int32])), num_parallel_calls=2)

    self.dataset = dataset
    self.data_size = len(image_paths)
    return self.dataset

  def read_image(self,path):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    return reader.Execute()

  def input_parser(self,image_path, label_path):
    # read image and label
    image = self.read_image(image_path.decode("utf-8"))
    label = self.read_image(label_path.decode("utf-8"))

    sample = {'image':image, 'label':label}

    # apply all transforms
    if self.transforms:
      for transform in self.transforms:
        sample = transform(sample)

    image_np = sitk.GetArrayFromImage(sample['image'])
    label_np = sitk.GetArrayFromImage(sample['label'])

    image_np = np.asarray(image_np,np.float32)/255
    label_np = np.asarray(label_np,np.int32)

    # setting channels and transposing volumes for more data augmentation
    if np.random.random() > 0.5 and self.train:
        image_np = np.transpose(image_np,(2,1,0))
        label_np = np.transpose(label_np,(2,1,0))
    else:
        image_np = np.transpose(image_np,(1,2,0))
        label_np = np.transpose(label_np,(1,2,0))
        
    return image_np, label_np

class Normalization(object):
  """
  Normalize an image by setting its mean to zero and variance to one
  """

  def __init__(self):
    self.name = 'Normalization'

  def __call__(self, sample):
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)
    image, label = sample['image'], sample['label']
    image = resacleFilter.Execute(image)
    
    return {'image':image, 'label':label}

class RandomFlip(object):
  """
  Randomly flipping volumes across all three axis
  """

  def __init__(self):
    self.name = 'RandomFlip'

  def __call__(self, sample):
    
    image, label = sample['image'], sample['label']
    
    # generating axis randomly
    flipaxes = np.random.random(3)>0.5
    
    flipimg = sitk.Flip(image, flipaxes.tolist())
    fliplab = sitk.Flip(label, flipaxes.tolist())
    
    return {'image':flipimg, 'label':fliplab}

class RandomSmoothing(object):
  """
  Randoml Gaussian smoothing
  """

  def __init__(self, prob):
    self.name = 'RandomSmoothing'
    self.prob = prob

  def __call__(self, sample):
    
    image, label = sample['image'], sample['label']
    
    if np.random.rand() < self.prob:
        image = sitk.RecursiveGaussian(image)
    
    return {'image':image, 'label':label}

class RandomNoise(object):
  """
  Randomly Gaussian Noise 
  """
    
  def __init__(self, prob):
    self.name = 'RandomNoise'
    self.prob = prob

  def __call__(self, sample):
    
    image, label = sample['image'], sample['label']
    
    if np.random.rand() < self.prob:
        image = sitk.AdditiveGaussianNoise(image)
    
    return {'image':image, 'label':label}


class HistogramMatching(object):
  """
  Histogram Matching with random images from training set (not applied to labels)
  """

  def __init__(self, data_dir, train_size=40, prob=0.5):
    self.name = 'Histogram Match'
    self.train_size = train_size
    self.data_dir = data_dir
    self.prob = prob

  def __call__(self, sample):

      image, label = sample['image'], sample['label']
    
      # histogram matching with random image from training set
       
      if np.random.random() <= self.prob:
            
          files = glob.glob(self.data_dir+'/Case*_segmentation.mhd')
          
          index = np.random.randint(0, self.train_size-1)
          file_name = files[index].replace('_segmentation', '')

          template = sitk.ReadImage(file_name)
          template = sitk.GetArrayFromImage(template)
          source = sitk.GetArrayFromImage(image) 
          oldshape = source.shape

          s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                  return_counts=True)
          t_values, t_counts = np.unique(template, return_counts=True)

          s_quantiles = np.cumsum(s_counts).astype(np.float64)
          s_quantiles /= s_quantiles[-1]
          t_quantiles = np.cumsum(t_counts).astype(np.float64)
          t_quantiles /= t_quantiles[-1]

          interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

          image = interp_t_values[bin_idx].reshape(oldshape)
          image = sitk.GetImageFromArray(image)

      return {'image':image, 'label':label}