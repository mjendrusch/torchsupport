"""
Affine transforms implemented on torch tensors, and
only requiring one interpolation

Included:
- Affine()
- AffineCompose()
- Rotation()
- Translation()
- Shear()
- Zoom()
- Flip()

"""

import math
import random
import torch

# necessary now, but should eventually not be
import scipy.ndimage as ndi
import numpy as np


def transform_matrix_offset_center(matrix, x, y):
    """Apply offset to a transform matrix so that the image is
    transformed about the center of the image. 

    NOTE: This is a fairly simple operaion, so can easily be
    moved to full torch.

    Arguments
    ---------
    matrix : 3x3 matrix/array

    x : integer
        height dimension of image to be transformed

    y : integer
        width dimension of image to be transformed
    """
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x, transform, fill_mode='nearest', fill_value=0.):
    """Applies an affine transform to a 2D array, or to each channel of a 3D array.

    NOTE: this can and certainly should be moved to full torch operations.

    Arguments
    ---------
    x : np.ndarray
        array to transform. NOTE: array should be ordered CHW
    
    transform : 3x3 affine transform matrix
        matrix to apply
    """
    x = x.astype('float32')
    transform = transform_matrix_offset_center(transform, x.shape[1], x.shape[2])
    final_affine_matrix = transform[:2, :2]
    final_offset = transform[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
            final_offset, order=0, mode=fill_mode, cval=fill_value) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    return x

class Affine(object):

    def __init__(self, 
                 rotation_range=None, 
                 translation_range=None,
                 shear_range=None, 
                 zoom_range=None, 
                 fill_mode='constant',
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0.):
        """Perform an affine transforms with various sub-transforms, using
        only one interpolation and without having to instantiate each
        sub-transform individually.

        Arguments
        ---------
        rotation_range : one integer or float
            image will be rotated between (-degrees, degrees) degrees

        translation_range : a float or a tuple/list w/ 2 floats between [0, 1)
            first value:
                image will be horizontally shifted between 
                (-height_range * height_dimension, height_range * height_dimension)
            second value:
                Image will be vertically shifted between 
                (-width_range * width_dimension, width_range * width_dimension)

        shear_range : float
            radian bounds on the shear transform

        zoom_range : list/tuple with two floats between [0, infinity).
            first float should be less than the second
            lower and upper bounds on percent zoom. 
            Anything less than 1.0 will zoom in on the image, 
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in, 
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform
            ProTip : use 'nearest' for discrete images (e.g. segmentations)
                    and use 'constant' for continuous images

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        target_fill_mode : same as fill_mode, but for target image

        target_fill_value : same as fill_value, but for target image

        """
        self.transforms = []
        if rotation_range:
            rotation_tform = Rotation(rotation_range, lazy=True)
            self.transforms.append(rotation_tform)

        if translation_range:
            translation_tform = Translation(translation_range, lazy=True)
            self.transforms.append(translation_tform)

        if shear_range:
            shear_tform = Shear(shear_range, lazy=True)
            self.transforms.append(shear_tform) 

        if zoom_range:
            zoom_tform = Translation(zoom_range, lazy=True)
            self.transforms.append(zoom_tform)

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value

    def __call__(self, x, y=None):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0](x)
        for tform in self.transforms[1:]:
            tform_matrix = np.dot(tform_matrix, tform(x)) 

        x = torch.from_numpy(apply_transform(x.numpy(), tform_matrix,
            fill_mode=self.fill_mode, fill_value=self.fill_value))

        if y:
            y = torch.from_numpy(apply_transform(y.numpy(), tform_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
            return x, y
        else:
            return x

class AffineCompose(object):

    def __init__(self, 
                 transforms, 
                 fill_mode='constant', 
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0.):
        """Apply a collection of explicit affine transforms to an input image,
        and to a target image if necessary

        Arguments
        ---------
        transforms : list or tuple
            each element in the list/tuple should be an affine transform.
            currently supported transforms:
                - Rotation()
                - Translation()
                - Shear()
                - Zoom()

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        """
        self.transforms = transforms
        # set transforms to lazy so they only return the tform matrix
        for t in self.transforms:
            t.lazy = True
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value

    def __call__(self, x, y=None):
        # collect all of the lazily returned tform matrices
        tform_matrix = self.transforms[0](x)
        for tform in self.transforms[1:]:
            tform_matrix = np.dot(tform_matrix, tform(x)) 

        x = torch.from_numpy(apply_transform(x.numpy(), tform_matrix,
            fill_mode=self.fill_mode, fill_value=self.fill_value))

        if y:
            y = torch.from_numpy(apply_transform(y.numpy(), tform_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
            return x, y
        else:
            return x


class Rotation(object):

    def __init__(self, 
                 rotation_range, 
                 fill_mode='constant', 
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        self.rotation_range = rotation_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        degree = random.uniform(-self.rotation_range, self.rotation_range)
        theta = math.pi / 180 * degree
        rotation_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                                    [math.sin(theta), math.cos(theta), 0],
                                    [0, 0, 1]])
        if self.lazy:
            return rotation_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), rotation_matrix,
                fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), rotation_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed

class Rotation4(object):

    def __init__(self,
                 fill_mode='constant', 
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly rotate an image between (-degrees, degrees). If the image
        has multiple channels, the same rotation will be applied to each channel.

        Arguments
        ---------
        rotation_range : integer or float
            image will be rotated between (-degrees, degrees) degrees

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        degree = random.choice([0, math.pi/2, math.pi, 3 * math.pi/2])
        theta = math.pi / 180 * degree
        rotation_matrix = np.array([[math.cos(theta), -math.sin(theta), 0],
                                    [math.sin(theta), math.cos(theta), 0],
                                    [0, 0, 1]])
        if self.lazy:
            return rotation_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), rotation_matrix,
                fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), rotation_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed

class Translation(object):

    def __init__(self, 
                 translation_range, 
                 fill_mode='constant',
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly translate an image some fraction of total height and/or
        some fraction of total width. If the image has multiple channels,
        the same translation will be applied to each channel.

        Arguments
        ---------
        translation_range : two floats between [0, 1) 
            first value:
                fractional bounds of total height to shift image
                image will be horizontally shifted between 
                (-height_range * height_dimension, height_range * height_dimension)
            second value:
                fractional bounds of total width to shift image 
                Image will be vertically shifted between 
                (-width_range * width_dimension, width_range * width_dimension)

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        if isinstance(translation_range, float):
            translation_range = (translation_range, translation_range)
        self.height_range = translation_range[0]
        self.width_range = translation_range[1]
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        # height shift
        if self.height_range > 0:
            tx = random.uniform(-self.height_range, self.height_range) * x.size(1)
        else:
            tx = 0
        # width shift
        if self.width_range > 0:
            ty = random.uniform(-self.width_range, self.width_range) * x.size(2)
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.lazy:
            return translation_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), 
                translation_matrix, fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), translation_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed


class Shear(object):

    def __init__(self, 
                 shear_range, 
                 fill_mode='constant', 
                 fill_value=0., 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly shear an image with radians (-shear_range, shear_range)

        Arguments
        ---------
        shear_range : float
            radian bounds on the shear transform
        
        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        self.shear_range = shear_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        shear = random.uniform(-self.shear_range, self.shear_range)
        shear_matrix = np.array([[1, -math.sin(shear), 0],
                                 [0, math.cos(shear), 0],
                                 [0, 0, 1]])
        if self.lazy:
            return shear_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), 
                shear_matrix, fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), shear_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed
      

class Zoom(object):

    def __init__(self, 
                 zoom_range, 
                 fill_mode='constant', 
                 fill_value=0, 
                 target_fill_mode='nearest', 
                 target_fill_value=0., 
                 lazy=False):
        """Randomly zoom in and/or out on an image 

        Arguments
        ---------
        zoom_range : tuple or list with 2 values, both between (0, infinity)
            lower and upper bounds on percent zoom. 
            Anything less than 1.0 will zoom in on the image, 
            anything greater than 1.0 will zoom out on the image.
            e.g. (0.7, 1.0) will only zoom in, 
                 (1.0, 1.4) will only zoom out,
                 (0.7, 1.4) will randomly zoom in or out

        fill_mode : string in {'constant', 'nearest'}
            how to fill the empty space caused by the transform

        fill_value : float
            the value to fill the empty space with if fill_mode='constant'

        lazy    : boolean
            if true, perform the transform on the tensor and return the tensor
            if false, only create the affine transform matrix and return that
        """
        if not isinstance(zoom_range, list) and not isinstance(zoom_range, tuple):
            raise ValueError('zoom_range must be tuple or list with 2 values')
        self.zoom_range = zoom_range
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.target_fill_mode = target_fill_mode
        self.target_fill_value = target_fill_value
        self.lazy = lazy

    def __call__(self, x, y=None):
        zx = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zy = random.uniform(self.zoom_range[0], self.zoom_range[1])
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if self.lazy:
            return zoom_matrix
        else:
            x_transformed = torch.from_numpy(apply_transform(x.numpy(), 
                zoom_matrix, fill_mode=self.fill_mode, fill_value=self.fill_value))
            if y:
                y_transformed = torch.from_numpy(apply_transform(y.numpy(), zoom_matrix,
                fill_mode=self.target_fill_mode, fill_value=self.target_fill_value))
                return x_transformed, y_transformed
            else:
                return x_transformed

class Normalize(object):

    def __init__(self):
      """Normalize an image by its own mean and standard deviation."""
      pass

    def __call__(self, x):
        for idx in range(x.shape[0]):
            xmean = torch.mean(x[idx, :, :])
            xstd = torch.std(x[idx, :, :])
            x[idx, :, :] = (x[idx, :, :] - xmean) / xstd
            if xstd == 0:
                x[idx, :, :] = 0.0
        return x

class Crop(object):

    def __init__(self, min_width, max_width, min_height, max_height):
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height

    def __call__(self, x):
        img_height = x.size(1)
        img_width = x.size(2)
        height = random.randrange(self.min_height, self.max_height)
        width = random.randrange(self.min_width, self.max_width)
        dh = img_height - height
        dw = img_width - width
        random_y = random.randrange(0, dh)
        random_x = random.randrange(0, dw)
        x = x[:, random_y:random_y+height, random_x:random_x+width]
        return x

class Elastic(object):
    def __init__(self, alpha=1000, sigma=30):
        self.alpha = alpha
        self.sigma = sigma

    def _elastic_transform(self, image, alpha=1000, sigma=30, spline_order=1, mode='nearest', random_state=np.random):
        """Elastic deformation of image as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.
        """
        assert image.ndim == 3
        shape = image.shape[:2]

        dx = ndi.filters.gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            sigma, mode="constant", cval=0
        ) * alpha
        dy = ndi.filters.gaussian_filter(
            (random_state.rand(*shape) * 2 - 1),
            sigma, mode="constant", cval=0
        ) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
        result = np.empty_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = ndi.interpolation.map_coordinates(
                image[:, :, i], indices, order=spline_order, mode=mode
            ).reshape(shape)
        return result

    def __call__(self, x):
        array = x.numpy().transpose(1, 2, 0)
        result = self._elastic_transform(
            array,
            np.random.uniform(*self.alpha) if isinstance(self.alpha, (list, tuple)) else self.alpha,
            np.random.uniform(*self.sigma) if isinstance(self.sigma, (list, tuple)) else self.sigma
        )
        return torch.Tensor(result.transpose(2, 0, 1))

class Perturb(object):
    def __init__(self, mean=0.0, std=0.1):
        """Perturb an image by normally distributed additive noise."""
        self.mean = mean
        self.std = std

    def __call__(self, x):
        noise = x.data.new(x.size()).normal_(
            self.mean, self.std
        ) if not isinstance(self.std, tuple) else \
        x.data.new(x.size()).normal_(
            self.mean, np.random.uniform(*self.std)
        )
        x = x + noise
        return x

class Illuminate(object):

    def __init__(self, illumination):
        """Apply an illumination mask to a given image.
        
        Arguments
        ---------
        illumination : tensor of the same size as the input tensor. Will be multiplied with the input tensor.
        """
        self.illumination = illumination
    
    def __call__(self, x):
        x = x * self.illumination
        return x

class PickChannels(object):

    def __init__(self, channels):
        """Pick a set of channels from a given image.

        Arguments
        ---------
        channels : list of channels to pick from the image.
        """
        self.channels = torch.LongTensor(channels)

    def __call__(self, x):
        return x.index_select(0, self.channels)

class Compose(object):
    def __init__(self, transforms, subset=2):
        self.transforms = transforms
        self.subset = min(subset, len(self.transforms))

    def __call__(self, x):
        out = x
        for transform in self.transforms:
            out = transform(out)
        return out
