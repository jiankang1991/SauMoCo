
import os
import numpy as np
import lmdb
import pyarrow as pa
import h5py
import torch

import gdal

from sklearn.preprocessing import LabelEncoder

def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def sample_neighbor(img_shape, xa, ya, neighborhood, tile_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius
    
    xn = np.random.randint(max(xa-neighborhood, tile_radius),
                           min(xa+neighborhood, w+tile_radius))
    yn = np.random.randint(max(ya-neighborhood, tile_radius),
                           min(ya+neighborhood, h+tile_radius))
    return xn, yn

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

def load_NAIP_img(img_file, val_type='uint8', bands_only=False, num_bands=4):
    """
    Loads an image using gdal, returns it as an array.
    """
    obj = gdal.Open(img_file)
    if val_type == 'uint8':
        img = obj.ReadAsArray().astype(np.uint8)
    elif val_type == 'float32':
        img = obj.ReadAsArray().astype(np.float32)
    else:
        raise ValueError('Invalid val_type for image values. Try uint8 or float32.')
    img = np.moveaxis(img, 0, -1)
    if bands_only: img = img[:,:,:num_bands]
    return img

class dataGenNAIP:
    """ 
    data generation for NAIP dataset, patch, patch_aug
    """
    def __init__(self, NAIP_LMDB=None, imgTransform=None, tile_size=50, num_tile=100000, neighborhood=100):

        self.env = lmdb.open(NAIP_LMDB, readonly=True, lock=False, readahead=False, meminit=False)
        self.imgTransform = imgTransform
        # self.state = state
        self.tile_size = tile_size
        self.num_tile = num_tile
        self.neighborhood = neighborhood
        self.tile_radius = self.tile_size // 2

    def __len__(self):

        return self.num_tile

    def __getitem__(self, idx):

        return self._getData(idx)

    def _getData(self, idx):

        with self.env.begin(write=False) as txn:
            byteflow = txn.get(f'{idx}'.encode())

        img = loads_pyarrow(byteflow)

        img_padded = np.pad(img, pad_width=[(self.tile_radius, self.tile_radius),
                                            (self.tile_radius, self.tile_radius), (0,0)],
                            mode='reflect')

        img_shape = img_padded.shape

        row_mid = img_shape[0] // 2
        col_mid = img_shape[1] // 2

        mid_tile = extract_tile(img_padded, row_mid, col_mid, self.tile_radius)
        xn, yn = sample_neighbor(img_shape, row_mid, col_mid, neighborhood=self.neighborhood, tile_radius=self.tile_radius)
        ngb_tile = extract_tile(img_padded, xn, yn, self.tile_radius)

        mid_tile = np.moveaxis(mid_tile, -1, 0)
        ngb_tile = np.moveaxis(ngb_tile, -1, 0)

        sample = {'anchor':mid_tile, 'neighbor':ngb_tile, 'idx':idx}

        if self.imgTransform is not None:
            sample = self.imgTransform(sample)
        
        return sample

class dataGenNAIP_HD5:
    
    def __init__(self, H5Pth=None, imgTransform=None, tile_size=50, num_tile=100000, neighborhood=100):

        self.H5Pth = H5Pth
        self.imgTransform = imgTransform
        # self.state = state
        self.tile_size = tile_size
        self.num_tile = num_tile
        self.neighborhood = neighborhood
        self.tile_radius = self.tile_size // 2
    

    def __len__(self):

        return self.num_tile

    def __getitem__(self, idx):

        return self._getData(idx)
    
    def _getData(self, idx):

        with h5py.File(self.H5Pth, "r") as data:

            img = data[f"{idx}"][:].astype(np.float32)

        img_padded = np.pad(img, pad_width=[(self.tile_radius, self.tile_radius),
                                            (self.tile_radius, self.tile_radius), (0,0)],
                            mode='reflect')

        img_shape = img_padded.shape

        row_mid = img_shape[0] // 2
        col_mid = img_shape[1] // 2

        mid_tile = extract_tile(img_padded, row_mid, col_mid, self.tile_radius)
        xn, yn = sample_neighbor(img_shape, row_mid, col_mid, neighborhood=self.neighborhood, tile_radius=self.tile_radius)
        ngb_tile = extract_tile(img_padded, xn, yn, self.tile_radius)

        mid_tile = np.moveaxis(mid_tile, -1, 0)
        ngb_tile = np.moveaxis(ngb_tile, -1, 0)

        sample = {'anchor':mid_tile, 'neighbor':ngb_tile, 'idx':idx}

        if self.imgTransform is not None:
            sample = self.imgTransform(sample)
        
        return sample

class dataGenNPY:
    def __init__(self, img_dir=None, imgTransform=None, tile_size=50, num_tile=100000, neighborhood=100):
        self.img_dir = img_dir
        self.imgTransform = imgTransform
        self.tile_size = tile_size
        self.num_tile = num_tile
        self.neighborhood = neighborhood
        self.tile_radius = self.tile_size // 2

    def __len__(self):

        return self.num_tile

    def __getitem__(self, idx):

        return self._getData(idx)

    def _getData(self, idx):

        tile_pth = os.path.join(self.img_dir, f"{idx}.npy")
        img = np.load(tile_pth)

        img_padded = np.pad(img, pad_width=[(self.tile_radius, self.tile_radius),
                                            (self.tile_radius, self.tile_radius), (0,0)],
                            mode='reflect')

        img_shape = img_padded.shape

        row_mid = img_shape[0] // 2
        col_mid = img_shape[1] // 2

        mid_tile = extract_tile(img_padded, row_mid, col_mid, self.tile_radius)
        xn, yn = sample_neighbor(img_shape, row_mid, col_mid, neighborhood=self.neighborhood, tile_radius=self.tile_radius)
        ngb_tile = extract_tile(img_padded, xn, yn, self.tile_radius)

        mid_tile = np.moveaxis(mid_tile, -1, 0)
        ngb_tile = np.moveaxis(ngb_tile, -1, 0)

        sample = {'anchor':mid_tile, 'neighbor':ngb_tile, 'idx':idx}

        if self.imgTransform is not None:
            sample = self.imgTransform(sample)
        
        return sample


class dataGenNPY_v2:
    def __init__(self, img_dir=None, imgTransform=None, tile_size=50, num_tile=100000, neighborhood=100):
        self.img_dir = img_dir
        self.imgTransform = imgTransform
        self.tile_size = tile_size
        self.num_tile = num_tile
        self.neighborhood = neighborhood
        self.tile_radius = self.tile_size // 2

        img_shape = (self.tile_radius+250, self.tile_radius+250, 4)
        self.row_mid = img_shape[0] // 2
        self.col_mid = img_shape[1] // 2

        self.neighors_xn_yn = []
        for _ in range(self.num_tile):
            self.neighors_xn_yn.append(sample_neighbor(img_shape, self.row_mid, self.col_mid, neighborhood=self.neighborhood, tile_radius=self.tile_radius))

    def __len__(self):

        return self.num_tile

    def __getitem__(self, idx):

        return self._getData(idx)

    def _getData(self, idx):

        tile_pth = os.path.join(self.img_dir, f"{idx}.npy")
        img = np.load(tile_pth)

        img_padded = np.pad(img, pad_width=[(self.tile_radius, self.tile_radius),
                                            (self.tile_radius, self.tile_radius), (0,0)],
                            mode='reflect')

        mid_tile = extract_tile(img_padded, self.row_mid, self.col_mid, self.tile_radius)
        xn, yn = self.neighors_xn_yn[idx]
        ngb_tile = extract_tile(img_padded, xn, yn, self.tile_radius)

        mid_tile = np.moveaxis(mid_tile, -1, 0)
        ngb_tile = np.moveaxis(ngb_tile, -1, 0)

        sample = {'anchor':mid_tile, 'neighbor':ngb_tile, 'idx':idx}

        if self.imgTransform is not None:
            sample = self.imgTransform(sample)
        
        return sample


class dataGenTestNAIP:
    """
    test data generation for NAIP
    """
    def __init__(self, test_dir=None, imgTransform=None, num_tiles=1000):

        self.test_dir = test_dir
        self.imgTransform = imgTransform
        self.num_tiles = num_tiles

        y = np.load(os.path.join(self.test_dir, 'y.npy'))
        # Reindex CDL classes
        self.y = LabelEncoder().fit_transform(y)

    def __len__(self):
        return self.num_tiles

    def __getitem__(self, idx):

        tile = np.load(os.path.join(self.test_dir, f'{idx+1}tile.npy'))
        tile = tile[...,:4]
        tile = np.moveaxis(tile, -1, 0)
        tile = tile / 255
        label = self.y[idx]

        sample = {'img':tile, 'label':label}

        if self.imgTransform is not None:
            sample = self.imgTransform(sample)
        
        return sample


class dataGenTestNAIP_RGB:
    """
    test data generation for NAIP
    """
    def __init__(self, test_dir=None, imgTransform=None, num_tiles=1000):

        self.test_dir = test_dir
        self.imgTransform = imgTransform
        self.num_tiles = num_tiles

        y = np.load(os.path.join(self.test_dir, 'y.npy'))
        # Reindex CDL classes
        self.y = LabelEncoder().fit_transform(y)

    def __len__(self):
        return self.num_tiles

    def __getitem__(self, idx):

        tile = np.load(os.path.join(self.test_dir, f'{idx+1}tile.npy'))
        tile = tile[...,:3]
        # tile = np.moveaxis(tile, -1, 0)
        tile = (tile / 255).astype(np.float32)
        label = self.y[idx]

        # sample = {'img':tile, 'label':label}

        if self.imgTransform is not None:
            tile = self.imgTransform(tile)
        
        sample = {'img':tile, 'label':label}

        return sample


class dataGenNAIPTile:
    """ 
    create 50x50 patches from one NAIP tile
    """
    def __init__(self, tile_pth, patch_sz=50, imgTransform=None):

        self.tile_img = load_NAIP_img(tile_pth)
        self.img_shape = self.tile_img.shape
        self.row_max = self.img_shape[0] // patch_sz
        self.col_max = self.img_shape[1] // patch_sz
        self.total_patch = self.row_max * self.col_max
        self.patch_sz = patch_sz
        self.imgTransform = imgTransform

    def __len__(self):
        return self.total_patch
    
    def __getitem__(self, idx):
        
        row_idx, col_idx = np.unravel_index(idx, (self.row_max, self.col_max))
        patch_img = self.tile_img[row_idx*self.patch_sz : (row_idx+1)*self.patch_sz, col_idx*self.patch_sz : (col_idx+1)*self.patch_sz, :]

        patch_img = (patch_img / 255).astype(np.float32)

        if self.imgTransform is not None:
            patch_img = self.imgTransform(patch_img)

        return {'img':patch_img}


class dataGenNAIPTile_RGB:
    """ 
    create 50x50 patches from one NAIP tile
    """
    def __init__(self, tile_pth, patch_sz=50, imgTransform=None):

        self.tile_img = load_NAIP_img(tile_pth)[...,:3]
        self.img_shape = self.tile_img.shape
        self.row_max = self.img_shape[0] // patch_sz
        self.col_max = self.img_shape[1] // patch_sz
        self.total_patch = self.row_max * self.col_max
        self.patch_sz = patch_sz
        self.imgTransform = imgTransform

    def __len__(self):
        return self.total_patch
    
    def __getitem__(self, idx):
        
        row_idx, col_idx = np.unravel_index(idx, (self.row_max, self.col_max))
        patch_img = self.tile_img[row_idx*self.patch_sz : (row_idx+1)*self.patch_sz, col_idx*self.patch_sz : (col_idx+1)*self.patch_sz, :]

        patch_img = (patch_img / 255).astype(np.float32)

        if self.imgTransform is not None:
            patch_img = self.imgTransform(patch_img)

        return {'img':patch_img}

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


def clip_and_scale_image(img, img_type='naip', clip_min=0, clip_max=10000):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    if img_type in ['naip', 'rgb']:
        return img / 255
    elif img_type == 'landsat':
        return np.clip(img, clip_min, clip_max) / (clip_max - clip_min)

class ClipAndScale(object):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    def __init__(self, img_type):
        assert img_type in ['naip', 'rgb', 'landsat']
        self.img_type = img_type

    def __call__(self, sample):
        a, n, idx= (clip_and_scale_image(sample['anchor'], self.img_type),
                   clip_and_scale_image(sample['neighbor'], self.img_type),
                   sample['idx'])
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



class TestToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, sample):
        img, label = (torch.from_numpy(sample['img']).float(),
            sample['label'])

        sample = {'img': img, 'label': label}
        return sample










