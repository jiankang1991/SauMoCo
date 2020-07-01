
import os
import numpy as np
import random
import glob
import torch
import lmdb
import pyarrow as pa
from skimage import io
from collections import defaultdict
from skimage.transform import resize


def interp_band(bands, img10_shape=[120,120]):
    """ 
    https://github.com/lanha/DSen2/blob/master/utils/patches.py
    """
    bands_interp = np.zeros(img10_shape + [bands.shape[-1]]).astype(np.float32)
    
    for i in range(bands.shape[-1]):
        bands_interp[...,i] = resize(bands[...,i] / 30000, img10_shape, mode='reflect') * 30000

    return bands_interp

def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)

BANDS = ['02', '03', '04', '08', '05', '06', '07', '8A', '11', '12', '01','09', '10']
EUROSAT_BANDS = ['01', '02', '03', '04', '05', '06', '07', '08', '8A', '09', '10', '11', '12']

eurosatbandsIdx = []
for b in EUROSAT_BANDS:
    eurosatbandsIdx.append(BANDS.index(b))


def sample_neighbor(img_shape_60, anchor_points_bands, neighborhood, tile_radius):

    w_padded, h_padded, c = img_shape_60
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius
    
    xn_60 = np.random.randint(max(anchor_points_bands['xa_60']-neighborhood, tile_radius),
                            min(anchor_points_bands['xa_60']+neighborhood, w+tile_radius))
    yn_60 = np.random.randint(max(anchor_points_bands['ya_60']-neighborhood, tile_radius),
                            min(anchor_points_bands['ya_60']+neighborhood, h+tile_radius))
    
    xn_20 = xn_60 * 3
    yn_20 = yn_60 * 3
    
    xn_10 = xn_60 * 6
    yn_10 = yn_60 * 6

    neighbor_points_bands = {'xn_60':xn_60, 'yn_60':yn_60, 'xn_20':xn_20, 'yn_20':yn_20, 'xn_10':xn_10, 'yn_10':yn_10}

    return neighbor_points_bands


def extract_tile(img_padded, x0, y0, tile_radius):
    """
    Extracts a tile from a (padded) image given the row and column of
    the center pixel and the tile size. E.g., if the tile
    size is 15 pixels per side, then the tile radius should be 7.
    """
    w_padded, h_padded, c = img_padded.shape
    row_min = x0 - tile_radius
    row_max = x0 + tile_radius
    col_min = y0 - tile_radius
    col_max = y0 + tile_radius
    assert row_min >= 0, 'Row min: {}'.format(row_min)
    assert row_max <= w_padded, 'Row max: {}'.format(row_max)
    assert col_min >= 0, 'Col min: {}'.format(col_min)
    assert col_max <= h_padded, 'Col max: {}'.format(col_max)
    # tile = img_padded[row_min:row_max+1, col_min:col_max+1, :]
    tile = img_padded[row_min:row_max, col_min:col_max, :]
    return tile



class Sen2dataGenNPY:

    def __init__(self, img_dir=None, imgTransform=None, tile_size_10=64, num_tile=100000, neighborhood=100):

        self.img_dir = img_dir
        self.imgTransform = imgTransform
        self.tile_size_10 = tile_size_10
        self.num_tile = num_tile
        self.neighborhood = neighborhood
        self.tile_radius_10 = self.tile_size_10 // 2

    def __len__(self):

        return self.num_tile
    
    def __getitem__(self, idx):

        return self._getData(idx)
    
    def _getData(self, idx):

        tile_pth = os.path.join(self.img_dir, f"{idx}.npy")
        img = np.load(tile_pth, allow_pickle=True).item()

        bands10 = img['bands_10']
        bands20 = img['bands_20']
        bands60 = img['bands_60']

        bands10_padded = np.pad(bands10, pad_width=[(self.tile_radius_10, self.tile_radius_10),
                (self.tile_radius_10, self.tile_radius_10), (0,0)], mode='reflect')
        bands20_padded = np.pad(bands20, pad_width=[(self.tile_radius_10//2, self.tile_radius_10//2),
            (self.tile_radius_10//2, self.tile_radius_10//2), (0,0)], mode='reflect')
        bands60_padded = np.pad(bands60, pad_width=[(self.tile_radius_10//6, self.tile_radius_10//6),
            (self.tile_radius_10//6, self.tile_radius_10//6), (0,0)], mode='reflect')

        img_shape_60 = bands60_padded.shape

        xa_60 = img_shape_60[0] // 2
        ya_60 = img_shape_60[1] // 2

        xa_20 = xa_60 * 3
        ya_20 = ya_60 * 3

        xa_10 = xa_60 * 6
        ya_10 = ya_60 * 6

        anchor_points_bands = {'xa_60':xa_60, 'ya_60':ya_60, 'xa_20':xa_20, 'ya_20':ya_20, 'xa_10':xa_10, 'ya_10':ya_10}

        tile_mid_60 = extract_tile(bands60_padded, anchor_points_bands['xa_60'], anchor_points_bands['ya_60'], self.tile_radius_10//6)
        tile_mid_20 = extract_tile(bands20_padded, anchor_points_bands['xa_20'], anchor_points_bands['ya_20'], self.tile_radius_10//2)
        tile_mid_10 = extract_tile(bands10_padded, anchor_points_bands['xa_10'], anchor_points_bands['ya_10'], self.tile_radius_10)

        neighbor_points_bands = sample_neighbor(img_shape_60, anchor_points_bands, self.neighborhood//6, self.tile_radius_10//6)

        tile_nbr_60 = extract_tile(bands60_padded, neighbor_points_bands['xn_60'], neighbor_points_bands['yn_60'], self.tile_radius_10//6)
        tile_nbr_20 = extract_tile(bands20_padded, neighbor_points_bands['xn_20'], neighbor_points_bands['yn_20'], self.tile_radius_10//2)
        tile_nbr_10 = extract_tile(bands10_padded, neighbor_points_bands['xn_10'], neighbor_points_bands['yn_10'], self.tile_radius_10)


        up_tile_mid_60 = interp_band(tile_mid_60, img10_shape=[self.tile_size_10, self.tile_size_10])
        up_tile_mid_20 = interp_band(tile_mid_20, img10_shape=[self.tile_size_10, self.tile_size_10])

        up_tile_nbr_60 = interp_band(tile_nbr_60, img10_shape=[self.tile_size_10, self.tile_size_10])
        up_tile_nbr_20 = interp_band(tile_nbr_20, img10_shape=[self.tile_size_10, self.tile_size_10])

        tile_mid = np.concatenate((tile_mid_10, up_tile_mid_20, up_tile_mid_60), axis=-1)
        tile_nbr = np.concatenate((tile_nbr_10, up_tile_nbr_20, up_tile_nbr_60), axis=-1)
        
        tile_mid = tile_mid[...,eurosatbandsIdx]
        tile_nbr = tile_nbr[...,eurosatbandsIdx]

        tile_mid = np.moveaxis(tile_mid, -1, 0)
        tile_nbr = np.moveaxis(tile_nbr, -1, 0)

        sample = {'anchor':tile_mid, 'neighbor':tile_nbr, 'idx':idx}

        if self.imgTransform is not None:
            sample = self.imgTransform(sample)
        
        return sample


class Sen2dataGenLMDB:
    def __init__(self, lmdb_pth=None, imgTransform=None, tile_size_10=64, num_tile=100000, neighborhood=100):
        
        self.env = lmdb.open(lmdb_pth, readonly=True, lock=False, readahead=False, meminit=False)
        self.imgTransform = imgTransform
        self.tile_size_10 = tile_size_10
        self.num_tile = num_tile
        self.neighborhood = neighborhood
        self.tile_radius_10 = self.tile_size_10 // 2

    def __len__(self):

        return self.num_tile
    
    def __getitem__(self, idx):

        return self._getData(idx)
    
    def _getData(self, idx):

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(u'{}'.format(idx).encode())

        bands10, bands20, bands60 = loads_pyarrow(byteflow)

        bands10_padded = np.pad(bands10, pad_width=[(self.tile_radius_10, self.tile_radius_10),
                (self.tile_radius_10, self.tile_radius_10), (0,0)], mode='reflect')
        bands20_padded = np.pad(bands20, pad_width=[(self.tile_radius_10//2, self.tile_radius_10//2),
            (self.tile_radius_10//2, self.tile_radius_10//2), (0,0)], mode='reflect')
        bands60_padded = np.pad(bands60, pad_width=[(self.tile_radius_10//6, self.tile_radius_10//6),
            (self.tile_radius_10//6, self.tile_radius_10//6), (0,0)], mode='reflect')

        img_shape_60 = bands60_padded.shape

        xa_60 = img_shape_60[0] // 2
        ya_60 = img_shape_60[1] // 2

        xa_20 = xa_60 * 3
        ya_20 = ya_60 * 3

        xa_10 = xa_60 * 6
        ya_10 = ya_60 * 6

        anchor_points_bands = {'xa_60':xa_60, 'ya_60':ya_60, 'xa_20':xa_20, 'ya_20':ya_20, 'xa_10':xa_10, 'ya_10':ya_10}

        tile_mid_60 = extract_tile(bands60_padded, anchor_points_bands['xa_60'], anchor_points_bands['ya_60'], self.tile_radius_10//6)
        tile_mid_20 = extract_tile(bands20_padded, anchor_points_bands['xa_20'], anchor_points_bands['ya_20'], self.tile_radius_10//2)
        tile_mid_10 = extract_tile(bands10_padded, anchor_points_bands['xa_10'], anchor_points_bands['ya_10'], self.tile_radius_10)

        neighbor_points_bands = sample_neighbor(img_shape_60, anchor_points_bands, self.neighborhood//6, self.tile_radius_10//6)

        tile_nbr_60 = extract_tile(bands60_padded, neighbor_points_bands['xn_60'], neighbor_points_bands['yn_60'], self.tile_radius_10//6)
        tile_nbr_20 = extract_tile(bands20_padded, neighbor_points_bands['xn_20'], neighbor_points_bands['yn_20'], self.tile_radius_10//2)
        tile_nbr_10 = extract_tile(bands10_padded, neighbor_points_bands['xn_10'], neighbor_points_bands['yn_10'], self.tile_radius_10)


        up_tile_mid_60 = interp_band(tile_mid_60, img10_shape=[self.tile_size_10, self.tile_size_10])
        up_tile_mid_20 = interp_band(tile_mid_20, img10_shape=[self.tile_size_10, self.tile_size_10])

        up_tile_nbr_60 = interp_band(tile_nbr_60, img10_shape=[self.tile_size_10, self.tile_size_10])
        up_tile_nbr_20 = interp_band(tile_nbr_20, img10_shape=[self.tile_size_10, self.tile_size_10])

        tile_mid = np.concatenate((tile_mid_10, up_tile_mid_20, up_tile_mid_60), axis=-1)
        tile_nbr = np.concatenate((tile_nbr_10, up_tile_nbr_20, up_tile_nbr_60), axis=-1)
        
        tile_mid = tile_mid[...,eurosatbandsIdx]
        tile_nbr = tile_nbr[...,eurosatbandsIdx]

        tile_mid = np.moveaxis(tile_mid, -1, 0)
        tile_nbr = np.moveaxis(tile_nbr, -1, 0)

        sample = {'anchor':tile_mid, 'neighbor':tile_nbr, 'idx':idx}

        if self.imgTransform is not None:
            sample = self.imgTransform(sample)
        
        return sample


class RandomFlipAndRotate(object):
    """
    Does data augmentation during training by randomly flipping (horizontal
    and vertical) and randomly rotating (0, 90, 180, 270 degrees). Keep in mind
    that pytorch samples are CxWxH.
    """
    def __call__(self, sample):
        a, n, idx = (sample['anchor'], sample['neighbor'], sample['idx'])
        # Randomly horizontal flip
        if np.random.rand() < 0.5: a = np.flip(a, axis=2).copy()
        if np.random.rand() < 0.5: n = np.flip(n, axis=2).copy()
        # if np.random.rand() < 0.5: d = np.flip(d, axis=2).copy()
        # Randomly vertical flip
        if np.random.rand() < 0.5: a = np.flip(a, axis=1).copy()
        if np.random.rand() < 0.5: n = np.flip(n, axis=1).copy()
        # if np.random.rand() < 0.5: d = np.flip(d, axis=1).copy()
        # Randomly rotate
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: a = np.rot90(a, k=rotations, axes=(1,2)).copy()
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: n = np.rot90(n, k=rotations, axes=(1,2)).copy()
        # rotations = np.random.choice([0, 1, 2, 3])
        # if rotations > 0: d = np.rot90(d, k=rotations, axes=(1,2)).copy()
        sample = {'anchor': a, 'neighbor': n, 'idx':idx}
        return sample



class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, sample):
        a, n, idx = (torch.from_numpy(sample['anchor']).float(),
            torch.from_numpy(sample['neighbor']).float(),
            sample['idx'])

        sample = {'anchor': a, 'neighbor': n, 'idx':idx}
        return sample



def eurosat_loader(path):
    return io.imread(path)


class DataGeneratorSplitting:
    """
    generate train and val dataset based on the following data structure:
    Data structure:
    └── SeaLake
        ├── SeaLake_1000.jpg
        ├── SeaLake_1001.jpg
        ├── SeaLake_1002.jpg
        ├── SeaLake_1003.jpg
        ├── SeaLake_1004.jpg
        ├── SeaLake_1005.jpg
        ├── SeaLake_1006.jpg
        ├── SeaLake_1007.jpg
        ├── SeaLake_1008.jpg
        ├── SeaLake_1009.jpg
        ├── SeaLake_100.jpg
        ├── SeaLake_1010.jpg
        ├── SeaLake_1011.jpg
    """

    def __init__(self, data, dataset, imgExt='jpg', imgTransform=None, phase='train'):

        self.dataset = dataset
        self.datadir = os.path.join(data, dataset)
        self.sceneList = [os.path.join(self.datadir, x) for x in sorted(os.listdir(self.datadir)) if os.path.isdir(os.path.join(self.datadir, x))]
        self.sceneFilesNum = defaultdict()
        
        self.train_idx2fileDict = defaultdict()
        self.test_idx2fileDict = defaultdict()
        self.val_idx2fileDict = defaultdict()

        self.scene2Label = defaultdict()
        self.imgTransform = imgTransform
        self.imgExt = imgExt
        self.phase = phase
        self.CreateIdx2fileDict()


    def CreateIdx2fileDict(self):
        # import random
        # random.seed(42)

        self.train_numImgs = 0
        self.test_numImgs = 0
        # self.val_numImgs = 0

        train_count = 0
        test_count = 0
        # val_count = 0

        for label, scenePth in enumerate(self.sceneList):
            self.scene2Label[os.path.basename(scenePth)] = label

            subdirImgPth = sorted(glob.glob(os.path.join(scenePth, '*.'+self.imgExt)))
            random.seed(42)
            random.shuffle(subdirImgPth)

            train_subdirImgPth = subdirImgPth[:int(0.8*len(subdirImgPth))]
            # val_subdirImgPth = subdirImgPth[int(0.7*len(subdirImgPth)):int(0.8*len(subdirImgPth))]
            test_subdirImgPth = subdirImgPth[int(0.8*len(subdirImgPth)):]
            # self.sceneFilesNum[os.path.basename(scenePth)] = len(subdirImgPth)
            self.train_numImgs += len(train_subdirImgPth)
            self.test_numImgs += len(test_subdirImgPth)
            # self.val_numImgs += len(val_subdirImgPth)

            for imgPth in train_subdirImgPth:
                self.train_idx2fileDict[train_count] = (imgPth, label)
                train_count += 1
            
            for imgPth in test_subdirImgPth:
                self.test_idx2fileDict[test_count] = (imgPth, label)
                test_count += 1
            
            # for imgPth in val_subdirImgPth:
            #     self.val_idx2fileDict[val_count] = (imgPth, label)
            #     val_count += 1
        
        print("total number of classes: {}".format(len(self.sceneList)))
        print("total number of train images: {}".format(self.train_numImgs))
        # print("total number of val images: {}".format(self.val_numImgs))
        print("total number of test images: {}".format(self.test_numImgs))

        # self.totalDataIndex = list(range(self.numImgs))
        self.trainDataIndex = list(range(self.train_numImgs))
        self.testDataIndex = list(range(self.test_numImgs))
        # self.valDataIndex = list(range(self.val_numImgs))

    def __getitem__(self, index):

        if self.phase == 'train':
            idx = self.trainDataIndex[index]
        # elif self.phase == 'val':
        #     idx = self.valDataIndex[index]
        else:
            idx = self.testDataIndex[index]
        
        return self.__data_generation(idx)

            
    def __data_generation(self, idx):
        
        # imgPth, imgLb = self.idx2fileDict[idx]

        if self.phase == 'train':
            imgPth, imgLb = self.train_idx2fileDict[idx]
        # elif self.phase == 'val':
        #     imgPth, imgLb = self.val_idx2fileDict[idx]
        else:
            imgPth, imgLb = self.test_idx2fileDict[idx]

        if self.dataset == 'eurosat':
            img = eurosat_loader(imgPth).astype(np.float32)
        else:
            img = default_loader(imgPth)

        if self.imgTransform is not None:
            img = self.imgTransform(img)
        
        # print(img.shape)

        return {'img': img, 'label': imgLb, 'idx':idx}
        # one hot encoding
        # oneHotVec = one_hot_encode(imgLb, len(self.sceneList))

        # return {'img': img, 'label': imgLb}

    def __len__(self):
        
        if self.phase == 'train':
            return len(self.trainDataIndex)
        # elif self.phase == 'val':
        #     return len(self.valDataIndex)
        else:
            return len(self.testDataIndex)


class Normalize(object):
    def __init__(self, eurosat_norm):
        
        self.mean = eurosat_norm['mean']
        self.std = eurosat_norm['std']

    def __call__(self, sample):
        
        a, n, idx = sample['anchor'], sample['neighbor'], sample['idx']


        for t, m, s in zip(a, self.mean, self.std):
            t.sub_(m).div_(s)
        
        for t, m, s in zip(n, self.mean, self.std):
            t.sub_(m).div_(s)

        return {'anchor': a, 'neighbor': n, 'idx':idx}
