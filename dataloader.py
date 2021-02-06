import numpy as np
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

import video_transforms as vtransforms
import utils


class Dataset_base(Dataset):
    def __init__(self, opt, train=True):
        
        # Get options
        self.opt = opt
        self.window_size = opt.window_size
        self.sample_size = opt.sample_size
        self.irregular = opt.irregular
        self.train = train
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Print out Dataset setting
        regularity = "irregular" if self.opt.irregular else "regular"
        task = "extrapolation" if self.opt.extrap else "interpolation"
        print(f"[Info] Dataset:{self.opt.dataset} / regularity:{regularity} / task:{task}")
    
    def sample_regular_interp(self, images):
        seq_len = images.shape[0]
        assert self.sample_size <= seq_len, "[Error] sample_size > seq_len"
        
        win_start = np.random.randint(0, seq_len - self.sample_size + 1) if self.train else 0
        

        if self.opt.phase == 'train':
            input_images = images[np.arange(win_start, win_start + self.sample_size, 2), ...]
            mask = torch.ones((self.sample_size // 2, 1))
        else:
            input_images = images[win_start: win_start + self.sample_size]
            mask = torch.zeros((self.sample_size, 1))
            mask[np.arange(0, self.sample_size, 2), :] = 1

        mask = mask.type(torch.FloatTensor).to(self.device)
        return input_images, mask
    
    def sample_regular_extrap(self, images):
        """ Same as sample_regular_interp, may be different when utils.sampling """

        seq_len = images.shape[0]
        assert self.sample_size <= seq_len, "[Error] sample_size > seq_len"
        
        # win_start = random.randint(0, seq_len - self.sample_size - 1) if self.train else 0
        win_start = random.randint(0, seq_len - self.sample_size) if self.train else 0
        
        input_images = images[win_start: win_start + self.sample_size]
        mask = torch.ones((self.sample_size, 1))
        mask = mask.type(torch.FloatTensor).to(self.device)
        
        return input_images, mask
    
    def sample_irregular_interp(self, images):
        
        seq_len = images.shape[0]
        
        if seq_len <= self.window_size:
            
            assert self.sample_size <= seq_len, "[Error] sample_size > seq_len"
            win_start = 0
            
            rand_idx = sorted(np.random.choice(list(range(win_start + 1, seq_len - 1)), size=self.sample_size - 2, replace=False))
            rand_idx = [win_start] + rand_idx + [seq_len - 1]
        
        elif seq_len > self.window_size:
            
            win_start = random.randint(0, seq_len - self.window_size - 1) if self.train else 0
            
            rand_idx = sorted(np.random.choice(list(range(win_start + 1, win_start + self.window_size - 1)), size=self.sample_size - 2, replace=False))
            rand_idx = [win_start] + rand_idx + [win_start + self.window_size - 1]

        # [Caution]: Irregular setting return window-sized images and it is filtered out
        # Sample images
        input_idx = list(range(win_start, win_start + self.window_size))
        input_images = images[input_idx]
        
        # Sample masks
        mask = torch.zeros((self.window_size, 1))
        mask_idx = [r - win_start for r in rand_idx]
        mask[mask_idx, :] = 1
        mask = mask.type(torch.FloatTensor).to(self.device)
        
        return input_images, mask
    
    def sample_irregular_extrap(self, images):
        
        seq_len = images.shape[0]
        assert self.window_size % 2 == 0, "[Error] window_size should be even number"
        assert self.sample_size % 2 == 0, "[Error] sample_size should be even number"
        half_window_size = self.window_size // 2
        half_sample_size = self.sample_size // 2
        
        if seq_len <= self.window_size:
            
            assert self.sample_size <= seq_len, "[Error] sample_size > seq_len"
            win_start = 0  # + 1
            half_window_size = seq_len // 2
            
            rand_idx_in = sorted(np.random.choice(list(range(win_start + 1, win_start + half_window_size)), size=half_sample_size - 1, replace=False))
            rand_idx_out = sorted(np.random.choice(list(range(win_start + half_window_size, win_start + seq_len - 1)), size=half_sample_size - 1, replace=False))
            rand_idx = [win_start] + rand_idx_in + rand_idx_out + [win_start + seq_len - 1]
        
        elif seq_len > self.window_size:
            win_start = random.randint(0, seq_len - self.window_size - 1) if self.train else 0  # + 1
            
            rand_idx_in = sorted(np.random.choice(list(range(win_start + 1, win_start + half_window_size)), size=half_sample_size - 1, replace=False))
            rand_idx_out = sorted(np.random.choice(list(range(win_start + half_window_size, win_start + self.window_size - 1)), size=half_sample_size - 1, replace=False))
            rand_idx = [win_start] + rand_idx_in + rand_idx_out + [win_start + self.window_size - 1]
        
        # [Caution]: Irregular setting return window-sized images and it is filtered out
        # Sample images
        input_idx = list(range(win_start, win_start + self.window_size))
        input_images = images[input_idx]
        
        # Sample mask
        mask = torch.zeros((self.window_size, 1))
        mask_idx = [r - win_start for r in rand_idx]
        mask[mask_idx, :] = 1
        
        return input_images, mask
    
    def sampling(self, images):
        
        # Sampling
        if not self.irregular and not self.opt.extrap:
            input_images, mask = self.sample_regular_interp(images=images)
        elif not self.irregular and self.opt.extrap:
            input_images, mask = self.sample_regular_extrap(images=images)
        elif self.irregular and not self.opt.extrap:
            input_images, mask = self.sample_irregular_interp(images=images)
        else:
            input_images, mask = self.sample_irregular_extrap(images=images)
        
        return input_images, mask
    
def remove_files_under_sample_size(image_path, threshold):

    temp_image_list = [x for x in os.listdir(image_path)]
    image_list = []
    remove_count = 0

    for i, file in enumerate(temp_image_list):
        _image = np.load(os.path.join(image_path, file))

        if _image.shape[0] >= threshold:
            image_list.append(file)
        else:
            remove_count += 1

    if remove_count > 0:
        print(f"Remove {remove_count:03d} shorter than than sample_size...")

    return image_list


class HurricaneVideoDataset(Dataset_base):
    def __init__(self, opt, train=True):
        super(HurricaneVideoDataset, self).__init__(opt, train=train)
        
        self.nc = 3 if self.opt.dataset == "hurricane" else 6
        
        if self.train:
            self.image_path = os.path.join('./dataset/Hurricane/', 'train')
        else:
            self.image_path = os.path.join('./dataset/Hurricane/', 'test')
        
        threshold = self.window_size if opt.irregular else self.sample_size
        self.image_list = remove_files_under_sample_size(image_path=self.image_path, threshold=threshold)
        self.image_list = sorted(self.image_list)
        
        vtrans = [vtransforms.Pad(padding=(1, 0), fill=0)]

        if self.train:
            # vtrans += [vtransforms.RandomHorizontalFlip()]
            # vtrans += [vtransforms.RandomRotation()]
            pass
        
        vtrans += [vtransforms.ToTensor(scale=False)]
        vtrans += [vtransforms.Normalize(0.5, 0.5)] if opt.input_norm else []
        self.vtrans = T.Compose(vtrans)
        
    def __getitem__(self, index):
        
        assert self.sample_size <= self.window_size, "[Error] sample_size > window_size"
        
        images = np.load(os.path.join(self.image_path, self.image_list[index]))
        images = images[..., :self.nc]
        
        # Sampling
        input_images, mask = self.sampling(images=images)
        
        # Transform
        input_images = self.vtrans(input_images)  # return (b, c, h, w)
        
        return input_images, mask
    
    def __len__(self):
        return len(self.image_list)


class VideoDataset(Dataset_base):
    
    def __init__(self, opt, train=True):
        super(VideoDataset, self).__init__(opt, train=train)
        
        # Dataroot & Transform
        if opt.dataset == 'mgif':
            data_root = './dataset/moving-gif'
            vtrans = [vtransforms.Scale(size=128)]
        elif opt.dataset == 'kth':
            data_root = './dataset/kth_action/'
            vtrans = [vtransforms.CenterCrop(size=120), vtransforms.Scale(size=128)]
        elif opt.dataset == 'penn':
            data_root = './dataset/penn_action/'
            vtrans = [vtransforms.Scale(size=128)]
        
        if self.train:
            vtrans += [vtransforms.RandomHorizontalFlip()]
            vtrans += [vtransforms.RandomRotation()]

        vtrans += [vtransforms.ToTensor(scale=True)]
        vtrans += [vtransforms.Normalize(0.5, 0.5)] if opt.input_norm else []
        self.vtrans = T.Compose(vtrans)
        
        if self.train:
            self.image_path = os.path.join(data_root, 'train')
        else:
            self.image_path = os.path.join(data_root, 'test')
        
        threshold = self.window_size if opt.irregular else self.sample_size
        if opt.dataset in ['kth', 'sintel', 'ucf101', 'penn']:
            self.image_list = os.listdir(self.image_path)
        elif opt.dataset in ['mgif', 'stickman']:
            self.image_list = remove_files_under_sample_size(image_path=self.image_path, threshold=threshold)
        self.image_list = sorted(self.image_list)
    
    def __getitem__(self, index):
        
        assert self.sample_size <= self.window_size, "[Error] sample_size > window_size"
        
        images = np.load(os.path.join(self.image_path, self.image_list[index]))
        
        # Sampling
        input_images, mask = self.sampling(images=images)

        # Transform
        input_images = self.vtrans(input_images)  # return (b, c, h, w)

        return input_images, mask
    
    def __len__(self):
        return len(self.image_list)


def parse_datasets(opt, device):
    def video_collate_fn(batch, time_steps, opt=opt, data_type="train"):
        
        images = torch.stack([b[0] for b in batch])
        mask = torch.stack([b[1] for b in batch])
        
        data_dict = {"data": images, "time_steps": time_steps, "mask": mask}
        
        data_dict = utils.split_and_subsample_batch(data_dict, opt, data_type=data_type)
        data_dict['mode'] = data_type
        return data_dict
    
    if opt.irregular:
        time_steps = np.arange(0, opt.window_size) / opt.window_size
    else:
        if opt.extrap:
            time_steps = np.arange(0, opt.sample_size) / opt.sample_size
        else:
            time_steps = np.arange(0, opt.sample_size // 2) / (opt.sample_size // 2)
            # time_steps = np.arange(0, opt.sample_size) / opt.sample_size
    
    time_steps = torch.from_numpy(time_steps).type(torch.FloatTensor).to(device)
    
    if opt.dataset in ['hurricane']:
        train_dataloader = DataLoader(HurricaneVideoDataset(opt, train=True),
                                      batch_size=opt.batch_size,
                                      shuffle=True,
                                      collate_fn=lambda batch: video_collate_fn(batch, time_steps, data_type="train"))
        test_dataloader = DataLoader(HurricaneVideoDataset(opt, train=False),
                                     batch_size=opt.batch_size,
                                     shuffle=False,
                                     collate_fn=lambda batch: video_collate_fn(batch, time_steps, data_type="test"))
    elif opt.dataset in ['mgif', 'kth', 'penn']:
        train_dataloader = DataLoader(VideoDataset(opt, train=True),
                                      batch_size=opt.batch_size,
                                      shuffle=True,
                                      collate_fn=lambda batch: video_collate_fn(batch, time_steps, data_type="train"))
        test_dataloader = DataLoader(VideoDataset(opt, train=False),
                                     batch_size=opt.batch_size,
                                     shuffle=False,
                                     collate_fn=lambda batch: video_collate_fn(batch, time_steps, data_type="test"))
    else:
        raise NotImplementedError(f"There is no dataset named {opt.dataset}")
    
    data_objects = {"train_dataloader": utils.inf_generator(train_dataloader),
                    "test_dataloader": utils.inf_generator(test_dataloader),
                    "n_train_batches": len(train_dataloader),
                    "n_test_batches": len(test_dataloader)}
    
    return data_objects


if __name__ == "__main__":
    pass
