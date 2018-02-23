from cv_sampler import Sampler
import numpy as np
import sys
import h5py
import argparse
import scipy
import scipy.ndimage as img
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str)
parser.add_argument('-count', type=int)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
N = args.count
dim = 256
mip = 2
is_test = args.test
sampler = Sampler(dim=dim, mip=mip)
dataset = np.empty((N, 2, dim, dim))

# this might not actually be elastic
def elastic(alpha=2000, sigma=50, rng=np.random.RandomState(42), interpolation_order=1):
    """Returns a function to elastically transform multiple images."""
    # Good values for:
    #   alpha: 2000
    #   sigma: between 40 and 60
    def _elastic_transform_2D(image):
        """`images` is a numpy array of shape (K, M, N) of K images of size M*N."""
        # Take measurements
        image_shape = image.shape
        # Make random fields
        dx = rng.uniform(-1, 1, image_shape) * alpha
        dy = rng.uniform(-1, 1, image_shape) * alpha
        # Smooth dx and dy
        sdx = gaussian_filter(dx, sigma=sigma, mode='reflect')
        sdy = gaussian_filter(dy, sigma=sigma, mode='reflect')
        # Make meshgrid
        x, y = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
        # Distort meshgrid indices
        distorted_indices = (y + sdy).reshape(-1, 1), \
                            (x + sdx).reshape(-1, 1)
        
        # Map cooordinates from image to distorted index set
        transformed_image = map_coordinates(image, distorted_indices, mode='reflect',
                                            order=interpolation_order).reshape(image_shape)
        return transformed_image
    return _elastic_transform_2D

def affine(image):
    std=0.04
    stdoff=2.0
    xoff, yoff = np.random.normal(0,stdoff), np.random.normal(0,stdoff)
    d = np.array([[np.random.normal(0,std), np.random.normal(0,std)],
                  [np.random.normal(0,std), np.random.normal(0,std)]])
    matrix = np.array([[1.0,0.0],[0.0,1.0]])
    matrix += d
    return img.interpolation.affine_transform(image, matrix, (xoff,yoff))

def vertical_crack(image, d=1):
    h, w = image.shape
    assert h == w
    m = int(0.5*w)
    new_image = np.zeros((w, w))
    new_image[:,:(m-d)] = image[:,d:m]
    new_image[:,(m+d):] = image[:,m:w-d]    
    return new_image

transform = vertical_crack  # function

def get_chunk():
    chunk = None
    while chunk is None:
        chunk = sampler.random_sample(train=not is_test)
        counts = dict(zip(*np.unique(chunk, return_counts=True)))
        if 0 in counts:  # if there's an empty pixel don't use
            print '  tossing a chunk with a 0'
            chunk = None
    return chunk

for i in range(N):
    chunk = get_chunk()
    dataset[i,0,:,:] = chunk
    dataset[i,1,:,:] = transform(chunk)
    if (i+1) % 10 == 0:
        print 'generated', (i+1)

#h5f = h5py.File("data/" + args.name + '_' + ('test' if is_test else 'train') + \
#                '_mip' + str(mip) + '.h5', 'w')
h5f = h5py.File("data/vcrack/" + args.name + '.h5', 'w')
h5f.create_dataset(u'main', data=dataset)
h5f.close()
