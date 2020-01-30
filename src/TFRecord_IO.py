import tensorflow as tf
import numpy as np
import reader
import tensorflow.contrib.slim as slim
from dataset_volume import Dataset
import os 
from tqdm import tqdm 

_RESOLUTIONS = ['5cm', '9cm', '19cm']
# _RESOLUTIONS = ['19cm']
_INPUT_FEATURE = 'input_sdf'
_TARGET_FEATURE = 'target_df'
_TARGET_SEM_FEATURE = 'target_sem'
# stored_dim_block=64
# stored_height_block=64


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
    gt = gt.astype(float)
    comp_gt = (gt > 0).astype(float)
    
    serialization = {
        key_input: _float_feature(sdf.ravel()),#tf.convert_to_tensor(sdf),
        key_target: _float_feature(comp_gt.ravel()),# tf.convert_to_tensor(gt),
        key_target_sem: _float_feature(gt.ravel()), #tf.convert_to_tensor(gt),
    }
    feature = tf.train.Example(features=tf.train.Features(feature=serialization))
    return feature.SerializeToString()
    
    
def LoadSequencePairToTFRecord(sdf_path,gt_path,mask_path,record_file, scale=1.0, level = 1):
    # sdf_path = '/media/sc/BackupDesk/TrainingData_TSDF/SceneNet_train_188/train/scenenet_rgbd_train_0'
    # gt_path = '/media/sc/BackupDesk/TrainingData_TSDF/SceneNet_train_188/gt/scenenet_rgbd_train_0'
    # mask_path = '/media/sc/BackupDesk/TrainingData_TSDF/SceneNet_train_188/mask/scenenet_rgbd_train_0'
    dataset = Dataset(sdf_path,gt_path,mask_path)
    
    # print('data length:',len(dataset))
    
    # record_file = 'images.tfrecords'
    # print('processing...')
    with tf.io.TFRecordWriter(record_file) as writer:
        for i in tqdm(range(len(dataset))): 
            sdf, gt, mask = dataset.__getitem__(i)
            
            sdf = sdf * scale
            # print(i)
            
            feature = Feature(sdf, gt, mask, hierarchy_level=level)
            writer.write(feature)
            
        # i+=1
    # print(i)
    # print('done')
       
        
if __name__ is '__main__':
    scale=18.8
    level = 3
    input_base_folder = '/media/sc/BackupDesk/TrainingData_TSDF/SceneNet_train_188'
    output_folder = '/home/sc/research/ScanComplete/train_SceneNetRGBD'
    input_folder_names = sorted(os.listdir(os.path.join(input_base_folder, 'train')))
    createFolder(output_folder)
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
        
        LoadSequencePairToTFRecord(input_path,gt_path,mask_path,output_path, scale=scale, level = level)
    