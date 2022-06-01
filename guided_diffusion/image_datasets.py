import math
import random
import torch as th
from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import imgaug.augmenters as iaa
from basicsr.data import degradations as degradations
import cv2
import math
import random
seed = np.random.RandomState(112311)
def load_data(
    *,
    data_dir,
    gt_dir,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        gt_dir,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class RandomCrop(object):

    def __init__(self, crop_size=[256,256]):
        """Set the height and weight before and after cropping"""
        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, inputs, target):
        input_size_h, input_size_w, _ = inputs.shape
        try:
            x_start = random.randint(0, input_size_w - self.crop_size_w)
            y_start = random.randint(0, input_size_h - self.crop_size_h)
            inputs = inputs[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] 
            target = target[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] 
        except:
            inputs=cv2.resize(inputs,(256,256))
            target=cv2.resize(target,(256,256))

        return inputs,target

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        gt_paths,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = True #random_crop
        self.random_flip = random_flip
        self.gt_paths=gt_paths
        # train_list=train_list[:10000]

        self.deformation = iaa.ElasticTransformation(alpha=[0, 50.], sigma=[4., 5.])

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        

        pil_image = cv2.imread(path)      ## Clean image RGB
        
        pil_image = cv2.cvtColor(pil_image, cv2.COLOR_BGR2GRAY)
        pil_image = np.repeat(pil_image[:,:,np.newaxis],3, axis=2)
        
        

        im1 = ((np.float32(pil_image)+1.0)/256.0)**2
        gamma_noise = seed.gamma(size=im1.shape, shape=1.0, scale=1.0).astype(im1.dtype)
        syn_sar = np.sqrt(im1 * gamma_noise)
        pil_image1 = syn_sar * 256-1   ## Noisy image

        

        
        arr1=np.array(pil_image)
        arr2=np.array(pil_image1)
        
        

        arr1 = cv2.resize(arr1, (256,256), interpolation=cv2.INTER_LINEAR)
        arr2= cv2.resize(arr2, (256,256), interpolation=cv2.INTER_LINEAR)
        

        

        arr1 = arr1.astype(np.float32) / 127.5 - 1
        arr2 = arr2.astype(np.float32) / 127.5 - 1
        
        

        out_dict = {}
        

        
        arr2 = np.transpose(arr2, [2, 0, 1])
        arr1 = np.transpose(arr1, [2, 0, 1])
        
        out_dict["SR"]=arr2
        out_dict["HR"]=arr1

        

        return arr1, out_dict  
        



def center_crop_arr(pil_image, pil_image1, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    while min(*pil_image1.size) >= 2 * image_size:
        pil_image1 = pil_image1.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image1.size)
    pil_image1 = pil_image1.resize(
        tuple(round(x * scale) for x in pil_image1.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    arr1 = np.array(pil_image1)

    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size],arr1[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, pil_image1, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    while min(*pil_image1.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image1.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image1.size)
    pil_image1 = pil_image1.resize(
        tuple(round(x * scale) for x in pil_image1.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    arr1 = np.array(pil_image1)

    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size],arr1[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
