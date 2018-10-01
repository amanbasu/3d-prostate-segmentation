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
      self.input_parser, [image_path, label_path], [tf.float32,tf.int32])))

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
    if self.train:
      label = self.read_image(label_path.decode("utf-8"))
    else:
      label = sitk.Image(image.GetSize(),sitk.sitkUInt32)
      label.SetOrigin(image.GetOrigin())
      label.SetSpacing(image.GetSpacing())

    sample = {'image':image, 'label':label}

    if self.transforms:
      for transform in self.transforms:
        sample = transform(sample)

    # convert sample to tf tensors
    image_np = sitk.GetArrayFromImage(sample['image'])
    label_np = sitk.GetArrayFromImage(sample['label'])

    image_np = np.asarray(image_np,np.float32)
    label_np = np.asarray(label_np,np.int32)

    # to unify matrix dimension order between SimpleITK([x,y,z]) and numpy([z,y,x])
    image_np = np.transpose(image_np,(2,1,0))
    label_np = np.transpose(label_np,(2,1,0))

    return image_np, label_np

    def hist_match(source, template):

      oldshape = source.shape

      s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                              return_counts=True)
      t_values, t_counts = np.unique(template, return_counts=True)

      s_quantiles = np.cumsum(s_counts).astype(np.float64)
      s_quantiles /= s_quantiles[-1]
      t_quantiles = np.cumsum(t_counts).astype(np.float64)
      t_quantiles /= t_quantiles[-1]

      interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

      return interp_t_values[bin_idx]

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

class RandomCrop(object):
  """
  Crop randomly the image in a sample. This is usually used for data augmentation.
	Drop ratio is implemented for randomly dropout crops with empty label. (Default to be 0.2)
	This transformation only applicable in train mode

  Args:
    output_size (tuple or int): Desired output size. If int, cubic crop is made.
  """

  def __init__(self, output_size=(128, 128, 64), drop_ratio=0.2, min_pixel=1):
    self.name = 'Random Crop'

    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size, output_size)
    else:
      assert len(output_size) == 3
      self.output_size = output_size

    assert isinstance(drop_ratio, float)
    if drop_ratio >=0 and drop_ratio<=1:
      self.drop_ratio = drop_ratio
    else:
      raise RuntimeError('Drop ratio should be between 0 and 1')

    assert isinstance(min_pixel, int)
    if min_pixel >=0 :
      self.min_pixel = min_pixel
    else:
      raise RuntimeError('Min label pixel count should be integer larger than 0')

  def __call__(self,sample):
    image, label = sample['image'], sample['label']
    size_old = image.GetSize()
    size_new = self.output_size

    contain_label = False

    roiFilter = sitk.RegionOfInterestImageFilter()
    roiFilter.SetSize([size_new[0],size_new[1],size_new[2]])

    while not contain_label: 
      # get the start crop coordinate in ijk
      if size_old[0] <= size_new[0]:
        start_i = 0
      else:
        start_i = np.random.randint(0, size_old[0]-size_new[0])

      if size_old[1] <= size_new[1]:
        start_j = 0
      else:
        start_j = np.random.randint(0, size_old[1]-size_new[1])

      if size_old[2] <= size_new[2]:
        start_k = 0
      else:
        start_k = np.random.randint(0, size_old[2]-size_new[2])

      roiFilter.SetIndex([start_i,start_j,start_k])

      label_crop = roiFilter.Execute(label)
      statFilter = sitk.StatisticsImageFilter()
      statFilter.Execute(label_crop)

      # will iterate until a sub volume containing label is extracted
      # pixel_count = seg_crop.GetHeight()*seg_crop.GetWidth()*seg_crop.GetDepth()
      # if statFilter.GetSum()/pixel_count<self.min_ratio:
      if statFilter.GetSum()<self.min_pixel:
        contain_label = self.drop(self.drop_ratio) # has some probabilty to contain patch with empty label
      else:
        contain_label = True

    image_crop = roiFilter.Execute(image)

    return {'image': image_crop, 'label': label_crop}

  def drop(self,probability):
    return np.random.random() <= probability

class RandomNoise(object):
  """
  Randomly noise to the image in a sample. This is usually used for data augmentation.
  """
  def __init__(self, probability):
    self.name = 'Random Noise'
    self.probability = probability

  def __call__(self, sample):

    image, label = sample['image'], sample['label']

    if np.random.random() <= self.probability:
      self.noiseFilter = sitk.AdditiveGaussianNoiseImageFilter()
      self.noiseFilter.SetMean(0)
      self.noiseFilter.SetStandardDeviation(0.1)

      # print("Normalizing image...")
      image = self.noiseFilter.Execute(image)

    return {'image': image, 'label': label}

class BSplineTransform(object):
    
    def __init__(self, numcontrolpoints, stdDef, probability):
        self.name = 'BSpline Transform'
        self.numcontrolpoints = numcontrolpoints
        self.stdDef = stdDef
        self.probability = probability
        
    def __call__(self, sample):

      sitkImage, sitklabel = sample['image'], sample['label']

      if np.random.random() <= self.probability:

        transfromDomainMeshSize = [self.numcontrolpoints]*sitkImage.GetDimension()

        tx = sitk.BSplineTransformInitializer(sitkImage, transfromDomainMeshSize)

        params = tx.GetParameters()

        paramsNp = np.asarray(params,dtype=float)
        paramsNp = paramsNp + np.random.randn(paramsNp.shape[0]) * self.stdDef

        paramsNp[0:int(len(params)/3)] = 0 

        params = tuple(paramsNp)
        tx.SetParameters(params)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitkImage)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(tx)

        resampler.SetDefaultPixelValue(0)
        sitkImage = resampler.Execute(sitkImage)
        sitklabel = resampler.Execute(sitklabel)

#         outimg = sitk.GetArrayFromImage(outimgsitk)
#         outimg = outimg.astype(dtype=np.float32)

#         outlbl = sitk.GetArrayFromImage(outlabsitk)
#         outlbl = (outlbl>0.5).astype(dtype=np.float32)
        
      return {'image': sitkImage, 'label': sitklabel}

class RandomTranslation(object):
    
    def __init__(self, probability):
        self.name = 'Random Translation'
        self.probability = probability
        
    def __call__(self, sample):
      sitkImage, sitklabel = sample['image'], sample['label']

      if np.random.random() <= self.probability:

        image = sitk.GetArrayFromImage(sitkImage)
        label = sitk.GetArrayFromImage(sitklabel)
        itemindex = np.where(label > 0)
        
        randTrans = (0,np.random.randint(-np.min(itemindex[1])/2,(image.shape[1]-np.max(itemindex[1]))/2),
                     np.random.randint(-np.min(itemindex[0])/2,(image.shape[0]-np.max(itemindex[0]))/2))
        translation = sitk.TranslationTransform(3, randTrans)

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(sitkImage)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(translation)

        sitkImage = resampler.Execute(sitkImage)
        sitklabel = resampler.Execute(sitklabel)

#         outimg = sitk.GetArrayFromImage(outimgsitk)
#         outimg = outimg.astype(dtype=float)

#         outlbl = sitk.GetArrayFromImage(outlabsitk) > 0
#         outlbl = outlbl.astype(dtype=float)

      return {'image': sitkImage, 'label': sitklabel}


