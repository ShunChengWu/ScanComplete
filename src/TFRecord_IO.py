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
threads = 4
debug=False
TRUNCATION = 3
p_norm = 1
up_limit=0

#baseFolder='/media/sc/BackupDesk/TrainingData_TSDF_0220/'

sub_folder_names = ['train','gt','gt_df']

# for training
if 0:
    for_eval = True
    baseFolder = '/media/sc/BackupDesk/TrainingDataScanNet_0614_TSDF/'
    input_folders = [
        [baseFolder + '050_200/' + 'train/'],
        [baseFolder + '100_200/' + 'train/'],
        [baseFolder + '200_200/' + 'train/'],
        ]
    output_folder = '/media/sc/SSD1TB/ReconstructionFromGT_TSDF_s200/' + 'ScanNet_3_level_train'
# for testing
if 0:
    for_eval = True
    baseFolder = '/media/sc/BackupDesk/TrainingDataScanNet_0614_TSDF/'
    input_folders = [
        [baseFolder + '050_200/' + 'test/'],
        [baseFolder + '100_200/' + 'test/'],
        [baseFolder + '200_200/' + 'test/'],
        ]
    output_folder = '/media/sc/SSD1TB/ReconstructionFromGT_TSDF_s200/' + 'ScanNet_3_level_test'


# for evaluation (whole scene)
if 1:
    for_eval = True
    baseFolder = '/media/sc/SSD1TB/ReconstructionFromGT_TSDF_whole_s200/'
    input_folders = [
        [baseFolder + '050/' + 'test/'],
        [baseFolder + '100/' + 'test/'],
        [baseFolder + '200/' + 'test/'],
        ]
    output_folder = baseFolder + 'ScanNet_3_level_test'



#debug=True


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
        
      #  if debug:
      #      print('\nshape: ', sdf.shape, '\n')
    
    return serialization
def Feature(sdf, gt, gt_df, hierarchy_level=1):
    feature = tf.train.Example(features=tf.train.Features(feature=get_dict(sdf,gt,gt_df,hierarchy_level-1)))
    return feature.SerializeToString()
    
    
def LoadSequencePairToTFRecord(input_file_name, record_file):
    datasets = [Dataset(base_folder, input_file_name) for base_folder in input_folders]
    # dataset = Dataset(sdf_path,gt_path,mask_path)
    with tf.io.TFRecordWriter(record_file) as writer:
        for i in tqdm(range(len(datasets[0]))): 
            #print(i)
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

            #if debug and i >= 1:
            #    break
            
if __name__ == '__main__':
    input_folder_names = sorted(os.listdir(os.path.join(input_folders[0][0], 'train')))
    createFolder(output_folder)
    pool = mp.Pool(threads)
    pool.daemon = True
    results=[]
   
    import re
    counter=0
    for input_file_name in input_folder_names:
        counter+=1
        number = re.findall('\d+',input_file_name)
        
        if len(number)  == 0:
            output_file_name = 'eval.tfrecords'
        else:
            nums=number[0]
            if(len(number))>1:
                for i in range(len(number)):
                    if i == 0: 
                        continue
                    nums=nums+"_"+number[i]
            output_file_name = 'train_{}.tfrecords'.format(nums)
        
        output_path = os.path.join(output_folder, output_file_name)
        print('input name', input_file_name)
        print('output_name',output_path)

        extend_name = [n+'/'+input_file_name+'/' for n in sub_folder_names]
        if debug:
            LoadSequencePairToTFRecord(extend_name, output_path)
            break
        else:
            results.append(
                pool.apply_async(LoadSequencePairToTFRecord, 
                                  (extend_name,output_path))
                )
        if up_limit>0:
            if counter >= up_limit:
                print('Reach up limit',up_limit, '. Break')
                break;
    if not debug:
        pool.close()
        pool.join()
        results = [r.get() for r in results]
        for r in results:
          print(r)
