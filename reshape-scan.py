import glob
import numpy as np
import SimpleITK as sitk

volSize = np.array((128,128,64), np.int32)
dstRes  = np.array((1,1,1.5))
normDir = False
method  = sitk.sitkLinear

def getNumpyData(scan):

    ret = np.zeros(volSize, dtype=np.float32)

    factor = np.asarray(scan.GetSpacing()) / dstRes

    factorSize = np.asarray(scan.GetSize() * factor, dtype=np.float)

    newSize = np.max([factorSize, volSize], axis=0)
    newSize = newSize.astype(dtype=np.int32)

    T = sitk.AffineTransform(3)
    T.SetMatrix(scan.GetDirection())

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(scan)
    resampler.SetOutputSpacing(dstRes)
    resampler.SetSize(newSize.tolist())
    resampler.SetInterpolator(method)
    if normDir:
        resampler.SetTransform(T.GetInverse())

    imgResampled = resampler.Execute(scan)

    imgCentroid = np.asarray(newSize, dtype=np.float) / 2.0

    imgStartPx = (imgCentroid - np.array(volSize) / 2.0).astype(dtype=int)

    regionExtractor = sitk.RegionOfInterestImageFilter()
    regionExtractor.SetSize(volSize.astype(dtype=np.int32).tolist())
    regionExtractor.SetIndex(imgStartPx.tolist())

    imgResampledCropped = regionExtractor.Execute(imgResampled)
    
    return imgResampledCropped

for i in glob.glob('test-data/Case*.mhd'):
    itkimage = sitk.ReadImage(i)
    reshaped = getNumpyData(itkimage)
    sitk.WriteImage(reshaped, i.replace('/','-mod/'))
    print(i, end='\r')
