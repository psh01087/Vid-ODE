import collections
import math
import torch
import random
import numpy as np
import numbers
import cv2
import PIL
from PIL import Image
import torchvision.transforms.functional as F
import skimage

def resize(video, size, interpolation):
    if interpolation == 'bilinear':
        inter = cv2.INTER_LINEAR
    elif interpolation == 'nearest':
        inter = cv2.INTER_NEAREST
    else:
        raise NotImplementedError
    
    shape = video.shape[:-3]
    video = video.reshape((-1, *video.shape[-3:]))
    resized_video = np.zeros((video.shape[0], size[1], size[0], video.shape[-1]))
    for i in range(video.shape[0]):
        img = cv2.resize(video[i], size, inter)
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
        resized_video[i] = img
    return resized_video.reshape((*shape, size[1], size[0], video.shape[-1]))


class ToTensor(object):
    """Converts a numpy.ndarray (... x H x W x C) in the range
  [0, 255] to a torch.FloatTensor of shape (... x C x H x W) in the range [0.0, 1.0].
  """
    
    def __init__(self, scale=True):
        self.scale = scale
    
    def __call__(self, arr):
        if isinstance(arr, np.ndarray):
            video = torch.from_numpy(np.rollaxis(arr, axis=-1, start=-3))
            # print(f"type(video):{type(video)}")
            if self.scale:
                return video.float().div(255)
            else:
                return video.float()
        
        else:
            raise NotImplementedError


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
  will normalize each channel of the torch.*Tensor, i.e.
  channel = (channel - mean) / std
  """
    
    def __init__(self, mean, std):
        if not isinstance(mean, list):
            mean = [mean]
        if not isinstance(std, list):
            std = [std]
        
        self.mean = torch.FloatTensor(mean).unsqueeze(1).unsqueeze(2)
        self.std = torch.FloatTensor(std).unsqueeze(1).unsqueeze(2)
    
    def __call__(self, tensor):
        return tensor.sub_(self.mean).div_(self.std)


class Scale(object):
    """Rescale the input numpy.ndarray to the given size.
  Args:
      size (sequence or int): Desired output size. If size is a sequence like
          (w, h), output size will be matched to this. If size is an int,
          smaller edge of the image will be matched to this number.
          i.e, if height > width, then image will be rescaled to
          (size * height / width, size)
      interpolation (int, optional): Desired interpolation. Default is
          ``bilinear``
  """
    
    def __init__(self, size, interpolation='bilinear'):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, video):
        """
    Args:
        video (numpy.ndarray): Video to be scaled.
    Returns:
        numpy.ndarray: Rescaled video.
    """
        if isinstance(self.size, int):
            w, h = video.shape[-2], video.shape[-3]
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return video
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return resize(video, (ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return resize(video, (ow, oh), self.interpolation)
        else:
            return resize(video, self.size, self.interpolation)


class CenterCrop(object):
    """Crops the given numpy.ndarray at the center to have a region of
  the given size. size can be a tuple (target_height, target_width)
  or an integer, in which case the target will be of a square shape (size, size)
  """
    
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
    
    def __call__(self, video):
        h, w = video.shape[-3:-1]
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        
        return video[..., y1:y1 + th, x1:x1 + tw, :]

class Cutout(object):
    """Cutout the given np.ndarray with rectangle mask.
    Args:
        mask_size (int): size of mask.
        mask_color: color of mask.
        centered (bool): if True, cutout occur at the center of the image
    """
    def __init__(self, mask_size, mask_color=(0, 0, 0), centered=True):
        if isinstance(mask_size, tuple):
            self.half_mask_size_x, self.half_mask_size_y = mask_size[0] // 2, mask_size[1] // 2
            self.offset_x = 1 if mask_size[0] % 2 == 0 else 0
            self.offset_y = 1 if mask_size[1] % 2 == 0 else 0
        else:
            self.half_mask_size_x, self.half_mask_size_y = mask_size // 2, mask_size // 2
            self.offset_x, self.offset_y = (1, 1) if mask_size % 2 == 0 else (0, 0)
            
        self.mask_color = mask_color
        self.centered = centered
        
    def __call__(self, video):
        
        h, w = video.shape[-3:-1]
        
        if self.centered:
            cx = w // 2
            cy = h // 2
        else:
            cxmin, cxmax = self.half_mask_size_x, w + self.offset_x - self.half_mask_size_x
            cymin, cymax = self.half_mask_size_y, h + self.offset_y - self.half_mask_size_y
            cx = np.random.randint(cxmin, cxmax)
            cy = np.random.randint(cymin, cymax)
        
        xmin, xmax = cx - self.half_mask_size_x, cx + self.half_mask_size_x
        ymin, ymax = cy - self.half_mask_size_y, cy + self.half_mask_size_y
        # print(f"xmin:{xmin}, xmax:{xmax}")
        # print(f"ymin:{ymin}, ymax:{ymax}")

        # prevent overflow
        xmin, ymin = max(0, xmin), max(0, ymin)
        xmax, ymax = min(w, xmax), min(h, ymax)
        # print(f"xmin:{xmin}, xmax:{xmax}")
        # print(f"ymin:{ymin}, ymax:{ymax}")

        # insert color
        video[..., ymin:ymax, xmin:xmax, :] = self.mask_color
        
        return video

class Pad(object):
    """Pad the given np.ndarray on all sides with the given "pad" value.
    Args:
      padding (int or sequence): Padding on each border. If a sequence of
          length 4, it is used to pad left, top, right and bottom borders respectively.
      fill: Pixel fill value. Default is 0.
    """
    
    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number) or isinstance(padding, tuple)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill
    
    def __call__(self, video):
        """
    Args:
        video (np.ndarray): Video to be padded.
    Returns:
        np.ndarray: Padded video.
    """
        # pad_width = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        # Note pad to last two dimension
        if isinstance(self.padding, tuple):
            # pad_width = ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1]))
            pad_width = ((0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1]), (0, 0))
        else:
            pad_width = ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        return np.pad(video, pad_width=pad_width, mode='constant', constant_values=self.fill)


class RandomCrop(object):
    """Crop the given numpy.ndarray at a random location.
  Args:
      size (sequence or int): Desired output size of the crop. If size is an
          int instead of sequence like (h, w), a square crop (size, size) is
          made.
      padding (int or sequence, optional): Optional padding on each border
          of the image. Default is 0, i.e no padding. If a sequence of length
          4 is provided, it is used to pad left, top, right, bottom borders
          respectively.
  """
    
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
    
    def __call__(self, video):
        """
    Args:
        video (np.ndarray): Video to be cropped.
    Returns:
        np.ndarray: Cropped video.
    """
        if self.padding > 0:
            pad = Pad(self.padding, 0)
            video = pad(video)
        
        w, h = video.shape[-2], video.shape[-3]
        th, tw = self.size
        if w == tw and h == th:
            return video
        
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return video[..., y1:y1 + th, x1:x1 + tw, :]


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy.ndarray with a probability of 0.5
  """
    
    def __call__(self, video):
        if random.random() < 0.5:
            return video[..., ::-1, :].copy()
        return video


class RandomSizedCrop(object):
    """Crop the given np.ndarray to random size and aspect ratio.
  A crop of random size of (0.08 to 1.0) of the original size and a random
  aspect ratio of 3/4 to 4/3 of the original aspect ratio is made. This crop
  is finally resized to given size.
  This is popularly used to train the Inception networks.
  Args:
      size: size of the smaller edge
      interpolation: Default: 'bilinear'
  """
    
    def __init__(self, size, interpolation='bilinear'):
        self.size = size
        self.interpolation = interpolation
    
    def __call__(self, video):
        for attempt in range(10):
            area = video.shape[-3] * video.shape[-2]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)
            
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))
            
            if random.random() < 0.5:
                w, h = h, w
            
            if w <= video.shape[-2] and h <= video.shape[-3]:
                x1 = random.randint(0, video.shape[-2] - w)
                y1 = random.randint(0, video.shape[-3] - h)
                
                video = video[..., y1:y1 + h, x1:x1 + w, :]
                
                return resize(video, (self.size, self.size), self.interpolation)
        
        # Fallback
        scale = Scale(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(video))

class RandomRotation(object):
    """Rotate entire clip randomly by a random angle within
    given bounds
    Args:
    degrees (sequence or int): Range of degrees to select from
    If degrees is a number instead of sequence like (min, max),
    the range of degrees, will be (-degrees, +degrees).
    """

    def __init__(self, degrees=10):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """
        
        angle = random.uniform(self.degrees[0], self.degrees[1])
        if isinstance(clip[0], np.ndarray):
            rotated = [skimage.transform.rotate(img, angle, preserve_range=True) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        
        rotated = np.array(rotated)
        return rotated


class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.
  Args:
      brightness (float): How much to jitter brightness. brightness_factor
          is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
      contrast (float): How much to jitter contrast. contrast_factor
          is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
      saturation (float): How much to jitter saturation. saturation_factor
          is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
      hue(float): How much to jitter hue. hue_factor is chosen uniformly from
          [-hue, hue]. Should be >=0 and <= 0.5.
  """
    
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
    Arguments are same as that of __init__.
    Returns:
        Transform which randomly adjusts brightness, contrast and
        saturation in a random order.
    """
        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
            transforms.append(lambda img: F.adjust_brightness(img, brightness_factor))
        
        if contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
            transforms.append(lambda img: F.adjust_contrast(img, contrast_factor))
        
        if saturation > 0:
            saturation_factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
            transforms.append(lambda img: F.adjust_saturation(img, saturation_factor))
        
        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(lambda img: F.adjust_hue(img, hue_factor))
        
        random.shuffle(transforms)
        
        return transforms
    
    def __call__(self, video):
        """
    Args:
        img (numpy array): Input image, shape (... x H x W x C), dtype uint8.
    Returns:
        PIL Image: Color jittered image.
    """
        transforms = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        reshaped_video = video.reshape((-1, *video.shape[-3:]))
        n_channels = video.shape[-1]
        for i in range(reshaped_video.shape[0]):
            img = reshaped_video[i]
            if n_channels == 1:
                img = img.squeeze(axis=2)
            img = Image.fromarray(img)
            for t in transforms:
                img = t(img)
            img = np.array(img)
            if n_channels == 1:
                img = img[..., np.newaxis]
            reshaped_video[i] = img
        video = reshaped_video.reshape(video.shape)
        return video
