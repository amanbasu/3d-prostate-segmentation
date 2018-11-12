'''
The code for resizing has been taken from
https://gist.github.com/zivy/79d7ee0490faee1156c1277a78e4a4c4
'''
    
def resample(img, new_size, interpolator):
    dimension = img.GetDimension()

    # Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
    reference_physical_size = np.zeros(dimension)

    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]
    
    # Create the reference image with a zero origin, identity direction cosine matrix and dimension     
    reference_origin = np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_size = new_size
    reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

    reference_image = sitk.Image(reference_size, img.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    # Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as 
    # this takes into account size, spacing and direction cosines. For the vast majority of images the direction 
    # cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the 
    # spacing will not yield the correct coordinates resulting in a long debugging session. 
    reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))

    # Transform which maps from the reference_image to the current img with the translation mapping the image
    # origins to each other.
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    # Using the linear interpolator as these are intensity images, if there is a need to resample a ground truth 
    # segmentation then the segmentation image should be resampled using the NearestNeighbor interpolator so that 
    # no new labels are introduced.

    return sitk.Resample(img, reference_image, centered_transform, interpolator, 0.0)

new_size = [144,144,50]             
interp = sitk.sitkNearestNeighbour  # for labels
interp = sitk.sitkLinear            # for features

for file in sorted(glob.glob('train-data-orig/Case*_segmentation.mhd')):
    # uncomment when resizing features
#     file = file.replace('_segmentation', '')
    img = sitk.ReadImage(file)
    reshaped = resample(img, new_size, interp)
    sitk.WriteImage(reshaped, file.replace('-orig',''))
    print(file, end='\r')
