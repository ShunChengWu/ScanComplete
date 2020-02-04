import os
import numpy as np
import tensorflow as tf

import constants
import model
import util
import reader
import re

_RESOLUTIONS = ['5cm', '9cm', '19cm']
_INPUT_FEATURE = 'input_sdf'
_TARGET_FEATURE = 'target_df'
_TARGET_SEM_FEATURE = 'target_sem'
_DIMS = [64,32,16]
# num_quant_levels = 256
threads = 17
debug=False
TRUNCATION = 3
# p_norm = 1

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('base_dir', '',
                    'Root directory. Expects a directory containing the model.')
flags.DEFINE_string('input_dir', '/home/sc/research/ScanComplete/train_SceneNetRGBD_094',
                    'Directory to input TFRecords.')
flags.DEFINE_string('input_dir_pre', '/home/sc/research/ScanComplete/train_SceneNetRGBD_188',
                    'Directory to previous TFRecords.')
flags.DEFINE_string('output_dir', '/home/sc/research/ScanComplete/train_SceneNetRGBD_094_pred', '')

flags.DEFINE_integer('hierarchy_level', 2, 'Hierachy level (1: finest level).')
flags.DEFINE_integer('num_quant_levels', 256, 'Number of quantization bins.')
flags.DEFINE_integer('pad_test', 0, 'Scene padding.')
flags.DEFINE_integer('p_norm', 1, 'P-norm loss (0 to disable).')
flags.DEFINE_bool('predict_semantics', False,
                  'Also predict semantic labels per-voxel.')
flags.DEFINE_float('temperature', 100.0, 'Softmax temperature for sampling.')

# 1. Load two levels of tfrecords
# 2. load pre-trained model
# 3. compute prediction of pre-level input
# 4. save current level of data entry + predicted input 

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

def read_input_float_feature(feature_map, key, shape):
    """Reads a float array from Example proto to np array."""
    if shape is None:
      (dim_z, dim_y, dim_x) = feature_map.feature[key + '/dim'].int64_list.value
    else:
      (dim_z, dim_y, dim_x) = shape
    tensor = np.array(feature_map.feature[key].float_list.value[:]).reshape(
        dim_z, dim_y, dim_x)
    return tensor


def read_input_bytes_feature(feature_map, key, shape):
    """Reads a byte array from Example proto to np array."""
    if shape is None:
      (dim_z, dim_y, dim_x) = feature_map.feature[key + '/dim'].int64_list.value
    else:
      (dim_z, dim_y, dim_x) = shape
    tensor = np.fromstring(
        feature_map.feature[key].bytes_list.value[0], dtype=np.uint8).reshape(
            dim_z, dim_y, dim_x)
    return tensor


def read_inputs(filename, feature_map, height, padding, num_quant_levels, p_norm,
                predict_semantics, processing, shape = [16,16,16]):
    """Reads inputs for scan completion.
    
    Reads input_sdf, target_df/sem (if any), previous predicted df/sem (if any).
    Args:
      filename: TFRecord containing input_sdf.
      height: height in voxels to be processed by model.
      padding: amount of padding (in voxels) around test scene (height is cropped
               by padding for processing).
      num_quant_levels: amount of quantization (if applicable).
      p_norm: which p-norm is used (0, 1, 2; 0 for none).
      predict_semantics: whether semantics is predicted.
    Returns:
      input scan: input_scan as np array.
      ground truth targets: target_scan/target_semantics as np arrays (if any).
      previous resolution predictions: prediction_scan_low_resolution /
                                       prediction_semantics_low_resolution as
                                       np arrays (if any).
    """
    hierarchy_level = FLAGS.hierarchy_level
    
    key_input = _RESOLUTIONS[hierarchy_level - 1] + '_' + _INPUT_FEATURE
    key_target = _RESOLUTIONS[hierarchy_level - 1] + '_' + _TARGET_FEATURE
    key_target_sem = _RESOLUTIONS[hierarchy_level - 1] + '_' + _TARGET_SEM_FEATURE
    
    # Input scan as sdf.
    input_scan = read_input_float_feature(feature_map, key_input, shape=shape)
    (scene_dim_z, scene_dim_y, scene_dim_x) = input_scan.shape
    
    # Target scan as df.
    if key_target in feature_map.feature:
      target_scan = read_input_float_feature(
          feature_map, key_target, [scene_dim_z, scene_dim_y, scene_dim_x])
    if key_target_sem in feature_map.feature:
      target_semantics = read_input_bytes_feature(
          feature_map, key_target_sem, [scene_dim_z, scene_dim_y, scene_dim_x])
      
    # Default values for previous resolution inputs.
    prediction_scan_low_resolution = np.zeros(
      [scene_dim_z // 2, scene_dim_y // 2, scene_dim_x // 2, 2])
    prediction_semantics_low_resolution = np.zeros(
      [scene_dim_z // 2, scene_dim_y // 2, scene_dim_x // 2], dtype=np.uint8)
    if processing:
        # Adjust dimensions for model (clamp height, make even for voxel groups).
        height_y = min(height, scene_dim_y - padding)
        scene_dim_x = (scene_dim_x // 2) * 2
        scene_dim_y = (height_y // 2) * 2
        scene_dim_z = (scene_dim_z // 2) * 2
        input_scan = input_scan[:scene_dim_z, padding:padding + scene_dim_y, :
                                scene_dim_x]
        input_scan = util.preprocess_sdf(input_scan, constants.TRUNCATION)
        if target_scan is not None:
          target_scan = target_scan[:scene_dim_z, padding:padding + scene_dim_y, :
                                    scene_dim_x]
          target_scan = util.preprocess_df(target_scan, constants.TRUNCATION)
        if target_semantics is not None:
          target_semantics = target_semantics[:scene_dim_z, padding:
                                              padding + scene_dim_y, :scene_dim_x]
          target_semantics = util.preprocess_target_sem(target_semantics)
      
        
        if target_semantics is None:
          target_semantics = np.zeros([scene_dim_z, scene_dim_y, scene_dim_x])
      
        # Load previous level prediction.
        # if not FLAGS.is_base_level:
        #   previous_file = os.path.join(
        #       FLAGS.output_dir_prev, 'level' + str(FLAGS.hierarchy_level - 1) + '_' +
        #       os.path.splitext(os.path.basename(filename))[0] + 'pred.tfrecord')
        #   tf.logging.info('Reading previous predictions frome file: %s',
        #                   previous_file)
        #   assert os.path.isfile(previous_file)
        #   for record in tf.python_io.tf_record_iterator(previous_file):
        #     prev_example = tf.train.Example()
        #     prev_example.ParseFromString(record)
        #     prev_feature_map = prev_example.features
        #   prediction_scan_low_resolution = read_input_float_feature(
        #       prev_feature_map, 'prediction_df', None)
        #   (prev_scene_dim_z, prev_scene_dim_y,
        #    prev_scene_dim_x) = prediction_scan_low_resolution.shape
        #   offset_z = (prev_scene_dim_z - scene_dim_z // 2) // 2
        #   offset_x = (prev_scene_dim_x - scene_dim_x // 2) // 2
        #   prediction_scan_low_resolution = prediction_scan_low_resolution[
        #       offset_z:offset_z + scene_dim_z // 2, :scene_dim_y // 2, offset_x:
        #       offset_x + scene_dim_x // 2]
        #   prediction_scan_low_resolution = util.preprocess_target_sdf(
        #       prediction_scan_low_resolution, num_quant_levels, constants.TRUNCATION,
        #       p_norm == 0)
        #   if predict_semantics:
        #     prediction_semantics_low_resolution = read_input_bytes_feature(
        #         prev_feature_map, 'prediction_sem',
        #         [prev_scene_dim_z, prev_scene_dim_y, prev_scene_dim_x])
        #     prediction_semantics_low_resolution = prediction_semantics_low_resolution[
        #         offset_z:offset_z + scene_dim_z // 2, :scene_dim_y // 2, offset_x:
        #         offset_x + scene_dim_x // 2]
                
    return (input_scan, target_scan, target_semantics,
            prediction_scan_low_resolution, prediction_semantics_low_resolution)
 
    
def get_dict(sdf,gt,gt_df, level):
    key_input = _RESOLUTIONS[level] + '_' + _INPUT_FEATURE
    key_target = _RESOLUTIONS[level] + '_' + _TARGET_FEATURE
    key_target_sem = _RESOLUTIONS[level] + '_' + _TARGET_SEM_FEATURE
    
    serialization = {
        key_input: _float_feature(sdf.ravel()),#tf.convert_to_tensor(sdf),
        key_target: _float_feature(gt_df.ravel()),# tf.convert_to_tensor(gt),
        key_target_sem: _bytes_feature(gt.tobytes()), #tf.convert_to_tensor(gt),
    }
    return serialization
def Feature(sdf, gt, gt_df, hierarchy_level=1, sdf_pre=None, gt_pre=None, gt_df_pre=None):
    level = hierarchy_level - 1
    key_input = _RESOLUTIONS[level] + '_' + _INPUT_FEATURE
    key_target = _RESOLUTIONS[level] + '_' + _TARGET_FEATURE
    key_target_sem = _RESOLUTIONS[level] + '_' + _TARGET_SEM_FEATURE
    
    sdf = sdf.astype(float)
    gt = np.uint8(gt)
    gt_df = gt_df.astype(float)
    
    sdf = sdf * TRUNCATION # from [-1,1] to voxel distance
    gt_df = gt_df * TRUNCATION
    
    serialization = {
        key_input: _float_feature(sdf.ravel()),#tf.convert_to_tensor(sdf),
        key_target: _float_feature(gt_df.ravel()),# tf.convert_to_tensor(gt),
        key_target_sem: _bytes_feature(gt.tobytes()), #tf.convert_to_tensor(gt),
    }
    
    if  level > 0 and sdf_pre is not None and gt_pre is not None and gt_df_pre is not None:
        level -= 1
        key_input = _RESOLUTIONS[level] + '_' + _INPUT_FEATURE
        key_target = _RESOLUTIONS[level] + '_' + _TARGET_FEATURE
        key_target_sem = _RESOLUTIONS[level] + '_' + _TARGET_SEM_FEATURE
        serialization[key_input] = _float_feature(sdf_pre.ravel())
        serialization[key_target] = _float_feature(gt_df_pre.ravel())
        serialization[key_target_sem] = _bytes_feature(gt_pre.tobytes())
    
    feature = tf.train.Example(features=tf.train.Features(feature=serialization))
    return feature.SerializeToString()
    
def process(path, path_pre, path_out):
    counter=0
    
    with tf.io.TFRecordWriter(path_out) as writer:
        for record, record_pre in zip(tf.python_io.tf_record_iterator(path), tf.python_io.tf_record_iterator(path_pre)):
            counter += 1
            example = tf.train.Example()
            example.ParseFromString(record)
            feature_map = example.features
            
            example_pre = tf.train.Example()
            example_pre.ParseFromString(record_pre)
            feature_map_pre = example_pre.features
            
            # load data at current level
            (input_scan, target_scan, target_semantics, prediction_scan_low_resolution,
              prediction_semantics_low_resolution) = read_inputs(
              FLAGS.input_dir_pre, feature_map, _DIMS[FLAGS.hierarchy_level-1], 0,
              FLAGS.num_quant_levels, FLAGS.p_norm, FLAGS.predict_semantics, processing=0,
              shape=[ _DIMS[FLAGS.hierarchy_level-1], _DIMS[FLAGS.hierarchy_level-1], _DIMS[FLAGS.hierarchy_level-1]])
            
                  
            # # load data at previous level
            # (input_scan, target_scan, target_semantics, prediction_scan_low_resolution,
            #   prediction_semantics_low_resolution) = read_inputs(
            #   FLAGS.input_dir_pre, feature_map, _DIMS[FLAGS.hierarchy_level-1], 0,
            #   FLAGS.num_quant_levels, FLAGS.p_norm, FLAGS.predict_semantics, processing=1,
            #   shape=[ _DIMS[FLAGS.hierarchy_level-1], _DIMS[FLAGS.hierarchy_level-1], _DIMS[FLAGS.hierarchy_level-1]])
        print('counter:',counter)
    
   
if __name__ == '__main__':
    input_folder_names = sorted(os.listdir(FLAGS.input_dir))
    input_folder_names_pre = sorted(os.listdir(FLAGS.input_dir_pre))
                                 
    for i in range(len(input_folder_names)):
        number = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?",input_folder_names[i])      
        output_file_name = 'train_{}.tfrecords'.format(number[0])
        
        in_path = os.path.join(FLAGS.input_dir,input_folder_names[i])
        in_path_pre = os.path.join(FLAGS.input_dir_pre,input_folder_names_pre[i])
        out_path = os.path.join(FLAGS.output_dir, output_file_name)
        
        
        
        print('input:', in_path)
        print('input_pre:', in_path_pre)
        print('in_path_pre:', in_path_pre)
        process(in_path, in_path_pre,in_path_pre)
        break
        