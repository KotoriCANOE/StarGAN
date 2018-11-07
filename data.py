from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
from skimage import io
from scipy import ndimage
import os
import random
from utils import bool_argument, eprint, listdir_files

# ======
# base class

class DataBase:
    __metaclass__ = ABCMeta

    def __init__(self, config):
        self.dataset = None
        self.num_epochs = None
        self.max_steps = None
        self.batch_size = None
        self.val_size = None
        self.packed = None
        self.processes = None
        self.threads = None
        self.prefetch = None
        self.buffer_size = None
        self.shuffle = None
        self.num_domains = None
        # copy all the properties from config object
        self.config = config
        self.__dict__.update(config.__dict__)
        # initialize
        self.get_files()

    @staticmethod
    def add_arguments(argp, test=False):
        # base parameters
        bool_argument(argp, 'packed', False)
        bool_argument(argp, 'test', test)
        # pre-processing parameters
        argp.add_argument('--processes', type=int, default=2)
        argp.add_argument('--threads', type=int, default=1)
        argp.add_argument('--prefetch', type=int, default=64)
        argp.add_argument('--buffer-size', type=int, default=256)
        bool_argument(argp, 'shuffle', True)
        argp.add_argument('--alignment', type=int, default=32)

    @staticmethod
    def parse_arguments(args):
        def argdefault(name, value):
            if args.__getattribute__(name) is None:
                args.__setattr__(name, value)
        def argchoose(name, cond, tv, fv):
            argdefault(name, tv if cond else fv)
        argchoose('batch_size', args.test, 12, 12)

    def get_files_packed(self):
        data_list = os.listdir(self.dataset)
        data_list = [os.path.join(self.dataset, i) for i in data_list]
        # val set
        if self.val_size is not None:
            self.val_steps = self.val_size // self.batch_size
            assert self.val_steps < len(data_list)
            self.val_size = self.val_steps * self.batch_size
            self.val_set = data_list[:self.val_steps]
            data_list = data_list[self.val_steps:]
            eprint('validation set: {}'.format(self.val_size))
        # main set
        self.epoch_steps = len(data_list)
        self.epoch_size = self.epoch_steps * self.batch_size
        if self.max_steps is None:
            self.max_steps = self.epoch_steps * self.num_epochs
        else:
            self.num_epochs = (self.max_steps + self.epoch_steps - 1) // self.epoch_steps
        self.main_set = data_list

    @abstractmethod
    def get_files_origin(self):
        pass

    def get_files(self):
        if self.packed: # packed dataset
            self.get_files_packed()
        else: # non-packed dataset
            data_list = self.get_files_origin()
            # val set
            if self.val_size is not None:
                assert self.val_size < len(data_list)
                self.val_steps = self.val_size // self.batch_size
                self.val_size = self.val_steps * self.batch_size
                self.val_set = data_list.iloc[:self.val_size]
                data_list = data_list.iloc[self.val_size:]
                eprint('validation set: {}'.format(self.val_size))
            # main set
            assert self.batch_size <= len(data_list)
            self.epoch_steps = len(data_list) // self.batch_size
            self.epoch_size = self.epoch_steps * self.batch_size
            if self.max_steps is None:
                self.max_steps = self.epoch_steps * self.num_epochs
            else:
                self.num_epochs = (self.max_steps + self.epoch_steps - 1) // self.epoch_steps
            self.main_set = data_list.iloc[:self.epoch_size]
        # print
        eprint('main set: {}\nepoch steps: {}\nnum epochs: {}\nmax steps: {}\n'
            .format(self.epoch_size, self.epoch_steps, self.num_epochs, self.max_steps))

    @staticmethod
    def process_sample(file, labels, config):
        # read from file
        data = io.imread(file)
        # convert from HWC to CHW
        if len(data.shape) < 3:
            data = np.expand_dims(data, 0)
        else:
            data = np.transpose(data, (2, 0, 1))
        # alignment
        sw = data.shape[-1]
        sh = data.shape[-2]
        tw = sw // config.alignment * config.alignment
        th = sh // config.alignment * config.alignment
        crop_l = (sw - tw) // 2
        crop_r = sw - tw - crop_l
        crop_r = -crop_r if crop_r > 0 else None
        crop_t = (sh - th) // 2
        crop_b = sh - th - crop_t
        crop_b = -crop_b if crop_b > 0 else None
        data = data[:, crop_t:crop_b, crop_l:crop_r]
        # random data manipulation
        # data = DataPP.process(data, config)
        # label format
        labels = np.maximum(0, labels)
        # return
        return data, labels

    @classmethod
    def extract_batch(cls, batch_set, config):
        from concurrent.futures import ThreadPoolExecutor
        # initialize
        inputs = []
        labels = []
        # load all the data
        with ThreadPoolExecutor(config.threads) as executor:
            futures = []
            for _file, _labels in batch_set.iterrows():
                futures.append(executor.submit(cls.process_sample, _file, _labels.values, config))
            # final data
            while len(futures) > 0:
                _input, _label = futures.pop(0).result()
                inputs.append(_input)
                labels.append(_label)
        # stack data to form a batch (NCHW)
        inputs = np.stack(inputs)
        labels = np.array(labels)
        targets = np.random.randint(0, 2, labels.shape)
        # return
        return inputs, labels, targets

    @classmethod
    def extract_batch_packed(cls, batch_set):
        npz = np.load(batch_set)
        inputs = npz['inputs']
        labels = npz['labels']
        targets = npz['targets']
        return inputs, labels, targets

    def _gen_batches_packed(self, dataset, epoch_steps, num_epochs=1, start=0):
        max_steps = epoch_steps * num_epochs
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(self.processes) as executor:
            futures = []
            # loop over epochs
            for epoch in range(start // epoch_steps, num_epochs):
                step_offset = epoch_steps * epoch
                step_start = max(0, start - step_offset)
                step_stop = min(epoch_steps, max_steps - step_offset)
                # loop over steps within an epoch
                for step in range(step_start, step_stop):
                    batch_set = dataset[step]
                    futures.append(executor.submit(self.extract_batch_packed,
                        batch_set))
                    # yield the data beyond prefetch range
                    while len(futures) >= self.prefetch:
                        yield futures.pop(0).result()
            # yield the remaining data
            for future in futures:
                yield future.result()

    def _gen_batches_origin(self, dataset, epoch_steps, num_epochs=1, start=0,
        shuffle=False):
        _dataset = dataset
        max_steps = epoch_steps * num_epochs
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(self.processes) as executor:
            futures = []
            # loop over epochs
            for epoch in range(start // epoch_steps, num_epochs):
                step_offset = epoch_steps * epoch
                step_start = max(0, start - step_offset)
                step_stop = min(epoch_steps, max_steps - step_offset)
                # random shuffle
                if shuffle and epoch > 0:
                    _dataset = dataset.sample(frac=1)
                # loop over steps within an epoch
                for step in range(step_start, step_stop):
                    offset = step * self.batch_size
                    upper = min(len(_dataset), offset + self.batch_size)
                    batch_set = _dataset.iloc[offset : upper]
                    futures.append(executor.submit(self.extract_batch,
                        batch_set, self.config))
                    # yield the data beyond prefetch range
                    while len(futures) >= self.prefetch:
                        yield futures.pop(0).result()
            # yield the remaining data
            for future in futures:
                yield future.result()

    def _gen_batches(self, dataset, epoch_steps, num_epochs=1, start=0,
        shuffle=False):
        # packed dataset
        if self.packed:
            return self._gen_batches_packed(dataset, epoch_steps, num_epochs, start)
        else:
            return self._gen_batches_origin(dataset, epoch_steps, num_epochs, start, shuffle)

    def gen_main(self, start=0):
        return self._gen_batches(self.main_set, self.epoch_steps, self.num_epochs,
            start, self.shuffle)

    def gen_val(self, start=0):
        return self._gen_batches(self.val_set, self.val_steps, 1,
            start, False)

class DataPP:
    @classmethod
    def process(cls, data, config):
        # smoothing
        smooth_prob = config.pp_smooth
        smooth_std = 0.75
        if cls.active_prob(smooth_prob):
            smooth_scale = cls.truncate_normal(smooth_std)
            data = ndimage.gaussian_filter1d(data, smooth_scale, truncate=2.0)
        # add noise
        noise_prob = config.pp_noise
        noise_std = 0.01
        noise_smooth_prob = 0.8
        noise_smooth_std = 1.5
        while cls.active_prob(noise_prob):
            # Gaussian noise
            noise_scale = cls.truncate_normal(noise_std)
            noise = np.random.normal(0.0, noise_scale, data.shape)
            # noise smoothing
            if cls.active_prob(noise_smooth_prob):
                smooth_scale = cls.truncate_normal(noise_smooth_std)
                noise = ndimage.gaussian_filter1d(noise, smooth_scale, truncate=2.0)
            # add noise
            data += noise
        # random amplitude
        amplitude = config.pp_amplitude / 10
        if amplitude > 0:
            data *= 0.1 ** np.random.uniform(0, amplitude) # 0~-20 dB
        # return
        return data

    @staticmethod
    def active_prob(prob):
        return np.random.uniform(0, 1) < prob

    @staticmethod
    def truncate_normal(std, mean=0.0, max_rate=4.0):
        max_scale = std * max_rate
        scale = max_scale + 1.0
        while scale > max_scale:
            scale = np.abs(np.random.normal(0.0, std))
        scale += mean
        return scale

# ======
# derived classes

class DataCelebA(DataBase):
    def get_files_origin(self):
        # get file ids
        base_dir = self.dataset
        attr_file = os.path.join(base_dir, 'Anno/list_attr_celeba.txt')
        img_dir = os.path.join(base_dir, 'Img/img_align_celeba_png')
        # read attr
        attr = pd.read_csv(attr_file, r'\s+', header=1)
        attr = attr.rename(index=lambda f: os.path.join(img_dir, f.replace('.jpg', '.png')))
        # select partial attr
        columns = ['Bald', 'Eyeglasses', 'Goatee', 'Male', 'Mustache', 'Pale_Skin']
        attr = attr[columns]
        self.num_domains = len(columns)
        self.config.num_domains = len(columns)
        # return
        if self.shuffle:
            attr = attr.sample(frac=1)
        return attr
