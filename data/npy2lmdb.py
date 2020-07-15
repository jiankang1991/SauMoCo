""" 
save the npy samples into a big lmdb file
"""
import numpy as np
import os
from tqdm import tqdm
import pyarrow as pa
import lmdb

def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()

tile_pth = os.path.join(f"./eurosat/train/0.npy")
img = np.load(tile_pth, allow_pickle=True).item()
num_tile=100000
map_size_ = (img['bands_10'].nbytes + img['bands_20'].nbytes + img['bands_60'].nbytes)*10*num_tile

db = lmdb.open('./eurosat/sen2_L1C_patches.lmdb', map_size=map_size_)

txn = db.begin(write=True)

for i in range(num_tile):
    
    tile_pth = os.path.join(f"./eurosat/train/{i}.npy")
    img = np.load(tile_pth, allow_pickle=True).item()

    txn.put(u'{}'.format(i).encode('ascii'), dumps_pyarrow((img['bands_10'], img['bands_20'], img['bands_60'])))

    if i % 10000 == 0:
        print("[%d/%d]" % (i, num_tile))
        txn.commit()
        txn = db.begin(write=True)

txn.commit()
with db.begin(write=True) as txn:
    # txn.put(b'__keys__', dumps_pyarrow(keys))
    txn.put(b'__len__', dumps_pyarrow(num_tile))

print("Flushing database ...")
db.sync()
db.close()









