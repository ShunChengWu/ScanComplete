import tensorflow as tf
import numpy as np

from dataset_volume import Dataset
import os 
from tqdm import tqdm 
import util
import math
import subprocess, os, sys, time
import multiprocessing as mp

_RESOLUTIONS = ['5cm', '9cm', '19cm']
# _RESOLUTIONS = ['19cm']
_INPUT_FEATURE = 'input_sdf'
_TARGET_FEATURE = 'target_df'
_TARGET_SEM_FEATURE = 'target_sem'
TRUNCATION = 3
p_norm = 1
num_quant_levels = 256
DIM = 16
debug=False

# stored_dim_block=64
# stored_height_block=64
def create_dfs_from_occupancy(input_ofu):
    X=input_ofu.shape[0]
    Y=input_ofu.shape[1]
    Z=input_ofu.shape[2]
    if X != DIM or Y != DIM or Z != DIM:
        raise RuntimeError('Dims should be ', DIM, 'got ', X)
        
    output_df = np.ones_like(input_ofu, dtype=np.float)
    nearest_distance = TRUNCATION
    output_df.fill(nearest_distance)
    #Find nearest occupied voxel in within truncation margin
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                if input_ofu[x,y,z] > 0: # only when current voxel is empty
                    output_df[x,y,z]=0
                    continue;
                for x_ in range(x-TRUNCATION,x+TRUNCATION+1):
                    if x_ < 0 or x_ >= X or x_ == x:
                        continue
                    for y_ in range(y-TRUNCATION,y+TRUNCATION+1):
                        if y_ < 0 or y_ >= Y or y_ == y:
                            continue
                        for z_ in range(z-TRUNCATION,z+TRUNCATION+1):
                            if z_ < 0 or z_ >= Z or z_ == z:
                                continue
                            # find closet
                            if input_ofu[x_,y_,z_] > 0:
                                distance = math.sqrt(x_*x_+y_+y_+z_*z_)
                                if distance < output_df[x,y,z]:
                                    output_df[x,y,z] = distance                                    
    return output_df                       

def create_dfs_from_output(input_sdf, output_df, target_scan):
  """Rescales model output to distance fields (in voxel units)."""
  input_sdf = (input_sdf[0, :, :, :, 0].astype(np.float32) + 1
              ) * 0.5 * TRUNCATION
  if p_norm > 0:
    factor = 0.5 if target_scan is not None else 1.0
    output_df = factor * TRUNCATION * (
        output_df[0, :, :, :, 0] + 1)
  else:
    output_df = (output_df[0, :, :, :, 0] + 1) * 0.5 * (
        num_quant_levels - 1)
    output_df = util.dequantize(output_df, num_quant_levels,
                                TRUNCATION)
  return input_sdf, output_df

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    
def Feature(sdf, gt, mask=None, hierarchy_level=0):
    key_input = _RESOLUTIONS[hierarchy_level - 1] + '_' + _INPUT_FEATURE
    key_target = _RESOLUTIONS[hierarchy_level - 1] + '_' + _TARGET_FEATURE
    key_target_sem = _RESOLUTIONS[hierarchy_level - 1] + '_' + _TARGET_SEM_FEATURE
    
    sdf = sdf.astype(float)
    # gt = gt.astype(float)
    gt = np.uint8(gt)
    comp_gt = (gt > 0).astype(float)
    
    # print("Length of one array element in bytes: ", gt.itemsize)
    # print("Total bytes consumed by the elements of the array: ", gt.nbytes) 
    # gt = np.uint8(gt)
    # print("Length of one array element in bytes: ", gt.itemsize)
    # print("Total bytes consumed by the elements of the array: ", gt.nbytes)
    
    comp_gt = create_dfs_from_occupancy(comp_gt)
    
    sdf = sdf * scale*TRUNCATION
    comp_gt = comp_gt * scale
    
    
    
    serialization = {
        key_input: _float_feature(sdf.ravel()),#tf.convert_to_tensor(sdf),
        key_target: _float_feature(comp_gt.ravel()),# tf.convert_to_tensor(gt),
        key_target_sem: _bytes_feature(gt.tobytes()), #tf.convert_to_tensor(gt),
    }
    feature = tf.train.Example(features=tf.train.Features(feature=serialization))
    return feature.SerializeToString()
    
    
def LoadSequencePairToTFRecord(sdf_path,gt_path,mask_path,record_file, scale=1.0, level = 1):
    dataset = Dataset(sdf_path,gt_path,mask_path)
    with tf.io.TFRecordWriter(record_file) as writer:
        for i in tqdm(range(len(dataset))): 
            sdf, gt, mask = dataset.__getitem__(i)
            feature = Feature(sdf, gt, mask, hierarchy_level=level)
            writer.write(feature)
            if debug and i > 50:
                break
            
if __name__ is '__main__':
    scale=18.8
    level = 3
    input_base_folder = '/media/sc/BackupDesk/TrainingData_TSDF/SceneNet_train_188'
    output_folder = '/home/sc/research/ScanComplete/train_SceneNetRGBD'
    input_folder_names = sorted(os.listdir(os.path.join(input_base_folder, 'train')))
    createFolder(output_folder)
    
    pool = mp.Pool(4)
    pool.daemon = True
    results=[]
   
    
    import re
    for input_file_name in input_folder_names:
        number = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",input_file_name)      
        output_file_name = 'train_{}.tfrecords'.format(number[0])
        
        input_path = os.path.join(input_base_folder, 'train',input_file_name)
        gt_path = os.path.join(input_base_folder, 'gt',input_file_name)
        mask_path = os.path.join(input_base_folder, 'mask',input_file_name)
        output_path = os.path.join(output_folder, output_file_name)
        print(input_path)
        print(output_path)
        print('')
      
        if debug:
            LoadSequencePairToTFRecord(input_path,gt_path,mask_path,output_path, scale=scale, level = level)
            break
        else:
            results.append(
                pool.apply_async(LoadSequencePairToTFRecord, 
                                 (input_path,gt_path,mask_path,output_path, scale,level))
                )
            

    pool.close()
    pool.join()
    results = [r.get() for r in results]
    for r in results:
         print(r)