# Read numpy to TFRecord
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
_INPUT_FEATURE = 'input_sdf'
_TARGET_FEATURE = 'target_df'
_TARGET_SEM_FEATURE = 'target_sem'
num_quant_levels = 256
threads = 17
debug=True
TRUNCATION = 3
p_norm = 1

#baseFolder='/media/sc/BackupDesk/TrainingData_TSDF_0220/'

# for training
if 1:
    for_eval = True
    baseFolder = '/home/sc/research/scslam/cmake-build-debug/App/TrainingDataGenerator/tmp/'
    input_folders = [
        baseFolder + '047/',
        ]
    output_folder = baseFolder + 'SceneNetRGBD_3_level_train'


# debug=True


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
    

def get_dict(sdf,gt,gt_df, level):
    key_input = _RESOLUTIONS[level] + '_' + _INPUT_FEATURE
    key_target = _RESOLUTIONS[level] + '_' + _TARGET_FEATURE
    key_target_sem = _RESOLUTIONS[level] + '_' + _TARGET_SEM_FEATURE
    
    serialization = {
        key_input: _float_feature(sdf.ravel()),#tf.convert_to_tensor(sdf),
        key_target: _float_feature(gt_df.ravel()),# tf.convert_to_tensor(gt),
        key_target_sem: _bytes_feature(gt.tobytes()), #tf.convert_to_tensor(gt),
    }
    if for_eval:
        serialization[key_input + "/dim"] = util.int64_feature(sdf.shape)
        
        if debug:
            print('\nshape: ', sdf.shape, '\n')
    
    return serialization
def Feature(sdf, gt, gt_df, hierarchy_level=1):
    feature = tf.train.Example(features=tf.train.Features(feature=get_dict(sdf,gt,gt_df,hierarchy_level-1)))
    return feature.SerializeToString()
    
    
def LoadSequencePairToTFRecord(input_file_name, record_file):
    datasets = [Dataset(os.path.join(base_folder, 'train', input_file_name),
                        os.path.join(base_folder, 'gt', input_file_name),
                        os.path.join(base_folder, 'gt_df', input_file_name)) for base_folder in input_folders]
    # dataset = Dataset(sdf_path,gt_path,mask_path)
    with tf.io.TFRecordWriter(record_file) as writer:
        for i in tqdm(range(len(datasets[0]))): 
            serialization = dict()
            for level in range(len(datasets)):
                sdf, gt, gt_df = datasets[level].__getitem__(i)
            
                sdf = sdf.astype(float)
                gt = np.uint8(gt)
                gt_df = gt_df.astype(float)
                sdf = sdf * TRUNCATION # from [-1,1] to voxel distance
                serialized = get_dict(sdf,gt, gt_df, level)
                for key, value in serialized.items():
                    serialization[key] = value
            
            feature = tf.train.Example(features=tf.train.Features(feature=serialization))
            writer.write(feature.SerializeToString())
            # if debug and i >= 1:
            #     break
            
if __name__ == '__main__':
    input_folder_names = sorted(os.listdir(os.path.join(input_folders[0], 'train')))
    createFolder(output_folder)
    pool = mp.Pool(threads)
    pool.daemon = True
    results=[]
   
    import re
    for input_file_name in input_folder_names:
        number = re.findall('\d+',input_file_name)
        if len(number)  == 0:
            output_file_name = 'eval.tfrecords'
        else:
            output_file_name = 'train_{}.tfrecords'.format(number[0])
        
        output_path = os.path.join(output_folder, output_file_name)
        print(input_file_name)
        print(output_path)
        print('')
      
        if debug:
            LoadSequencePairToTFRecord(input_file_name, output_path)
            break
        else:
            results.append(
                pool.apply_async(LoadSequencePairToTFRecord, 
                                  (input_file_name,output_path))
                )
    pool.close()
    pool.join()
    results = [r.get() for r in results]
    for r in results:
          print(r)
