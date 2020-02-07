import os
import numpy as np
import tensorflow as tf
import constants
import model
import util
import reader
import re
import multiprocessing as mp

_RESOLUTIONS = ['5cm', '9cm', '19cm']
_INPUT_FEATURE = 'input_sdf'
_TARGET_FEATURE = 'target_df'
_TARGET_SEM_FEATURE = 'target_sem'
_DIMS = [64,32,16]
# num_quant_levels = 256
threads = 3
TRUNCATION = 3
# p_norm = 1

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('base_dir', '',
                    'Root directory. Expects a directory containing the model.')
flags.DEFINE_string('input_dir', '/media/sc/BackupDesk/TrainingData_TSDF/train_SceneNetRGBD_3_level',
                    'Directory to input TFRecords.')
flags.DEFINE_string('predict_dir', '/media/sc/BackupDesk/TrainingData_TSDF/train_SceneNetRGBD_3_level_pred',
                    'Directory to input TFRecords.')
flags.DEFINE_string('output_dir', '/media/sc/BackupDesk/TrainingData_TSDF/train_SceneNetRGBD_3_level_pred', '')
flags.DEFINE_string('model_path', '/home/sc/research/ScanComplete/train/train_v003', '')
flags.DEFINE_string('model_checkpoint', '',
                    'Model checkpoint to use (empty for latest).')
flags.DEFINE_integer('hierarchy_level', 3, 'Hierachy level (1: finest level).')
flags.DEFINE_integer('num_quant_levels', 256, 'Number of quantization bins.')
flags.DEFINE_bool('is_base_level', False, 'If base level of hierarchy.')
flags.DEFINE_integer('pad_test', 0, 'Scene padding.')
flags.DEFINE_integer('p_norm', 1, 'P-norm loss (0 to disable).')
flags.DEFINE_bool('predict_semantics', True,
                  'Also predict semantic labels per-voxel.')
flags.DEFINE_bool('debug', False, '')
flags.DEFINE_float('temperature', 100.0, 'Softmax temperature for sampling.')


# 1. Load two levels of tfrecords
# 2. load pre-trained model
# 3. compute prediction of pre-level input
# 4. save current level of data entry + predicted input 


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
    tensor = np.frombuffer(
        feature_map.feature[key].bytes_list.value[0], dtype=np.uint8).reshape(
            dim_z, dim_y, dim_x)
    return tensor

def read_prediction(hierarchy_level, feature_map, num_quant_levels, p_norm,
                predict_semantics, shape = [16,16,16]):
    
    key_samples = 'samples_' + _RESOLUTIONS[hierarchy_level] 
    key_samples_sem = 'sem_samples_' + _RESOLUTIONS[hierarchy_level] 
     
    prediction_scan_low_resolution = read_input_float_feature(
          feature_map, key_samples, shape)
    (scene_dim_z, scene_dim_y,
        scene_dim_x) = prediction_scan_low_resolution.shape
    prediction_scan_low_resolution = util.preprocess_target_sdf(
          prediction_scan_low_resolution, num_quant_levels, constants.TRUNCATION,
          p_norm == 0)
    
    
    if predict_semantics:
        prediction_semantics_low_resolution = read_input_bytes_feature(
            feature_map, key_samples_sem,
            [scene_dim_z, scene_dim_y, scene_dim_x])
    return prediction_scan_low_resolution, prediction_semantics_low_resolution
  

def read_inputs(hierarchy_level, feature_map, height, padding, num_quant_levels, p_norm,
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

    
    key_input = _RESOLUTIONS[hierarchy_level] + '_' + _INPUT_FEATURE
    key_target = _RESOLUTIONS[hierarchy_level] + '_' + _TARGET_FEATURE
    key_target_sem = _RESOLUTIONS[hierarchy_level] + '_' + _TARGET_SEM_FEATURE
    
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
      
    return (input_scan, target_scan, target_semantics,
            prediction_scan_low_resolution, prediction_semantics_low_resolution)



class Prediction():
    def __init__(self,dims,model_path):
        self.dims = dims
        
        (input_placeholder, target_placeholder, target_lo_placeholder,
             target_sem_placeholder, target_sem_lo_placeholder, logits) = self.create_model(dims[0], dims[1], dims[2])
        
        logit_groups_geometry = logits['logits_geometry']
        logit_groups_semantics = logits['logits_semantics']
        feature_groups = logits['features']
          
        predictions_geometry_list, predictions_semantics_list =self.predict_from_model(
        logit_groups_geometry, logit_groups_semantics, FLAGS.temperature)
        
        self.input_placeholder = input_placeholder
        self.target_placeholder = target_placeholder
        self.target_lo_placeholder = target_lo_placeholder
        self.target_sem_placeholder = target_sem_placeholder
        self.target_sem_lo_placeholder = target_sem_lo_placeholder
        self.feature_groups = feature_groups
        self.predictions_geometry_list = predictions_geometry_list
        self.predictions_semantics_list =predictions_semantics_list
        
        init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
        # Run on the cpu - don't need to worry about scene sizes
        config = tf.ConfigProto( device_count = {'GPU': 0} )
        session = tf.Session(config=config)
        session.run(init_op)
        
        if FLAGS.model_checkpoint:
          checkpoint_path = os.path.join(model_path, FLAGS.model_checkpoint)
        else:
          checkpoint_path = tf.train.latest_checkpoint(model_path)
          
        assign_fn = tf.contrib.framework.assign_from_checkpoint_fn(
            checkpoint_path, tf.contrib.framework.get_variables_to_restore())
        assign_fn(session)
        tf.logging.info('Checkpoint loaded.')
        
        self.session = session
    def predict(self, input_scan, target_scan, target_semantics, prediction_scan_low_resolution,
                prediction_semantics_low_resolution):
        feature_groups = self.feature_groups
        input_placeholder = self.input_placeholder
        target_lo_placeholder = self.target_lo_placeholder
        target_placeholder = self.target_placeholder
        target_sem_lo_placeholder = self.target_sem_lo_placeholder
        target_sem_placeholder  = self.target_sem_placeholder
        predictions_geometry_list=self.predictions_geometry_list
        predictions_semantics_list=self.predictions_semantics_list
        session = self.session
        
        tf.logging.info('Predicting...')
        
        # Make batch size 1 to input to model.
        input_scan = input_scan[np.newaxis, :, :, :, :]
        prediction_scan_low_resolution = prediction_scan_low_resolution[
            np.newaxis, :, :, :, :]
        prediction_semantics_low_resolution = prediction_semantics_low_resolution[
            np.newaxis, :, :, :]
        output_prediction_scan = np.ones(shape=input_scan.shape)
        # Fill with truncation, known values.
        output_prediction_scan[:, :, :, :, 0] *= constants.TRUNCATION
        output_prediction_semantics = np.zeros(
            shape=[1, self.dims[0], self.dims[1], self.dims[2]], dtype=np.uint8)
    
        # First get features.
        feed_dict = {
            input_placeholder: input_scan,
            target_lo_placeholder: prediction_scan_low_resolution,
            target_placeholder: output_prediction_scan,
            target_sem_lo_placeholder: prediction_semantics_low_resolution,
            target_sem_placeholder: output_prediction_semantics
        }
        # Cache these features.
        feature_groups_ = session.run(feature_groups, feed_dict)
        for n in range(8):
          tf.logging.info('Predicting group [%d/%d]', n + 1, 8)
          # Predict
          feed_dict[feature_groups[n]] = feature_groups_[n]
          predictions = session.run(
              {
                  'prediction_geometry': predictions_geometry_list[n],
                  'prediction_semantics': predictions_semantics_list[n]
              },
              feed_dict=feed_dict)
          prediction_geometry = predictions['prediction_geometry']
          prediction_semantics = predictions['prediction_semantics']
          # Put into [-1,1] for next group.
          if FLAGS.p_norm == 0:
            prediction_geometry = prediction_geometry.astype(np.float32) / (
                (FLAGS.num_quant_levels - 1) / 2.0) - 1.0
    
          util.assign_voxel_group(output_prediction_scan, prediction_geometry,
                                  n + 1)
          if FLAGS.predict_semantics:
            util.assign_voxel_group(output_prediction_semantics,
                                    prediction_semantics, n + 1)
    
        # Final outputs.
        output_prediction_semantics = output_prediction_semantics[0]
        # Make distances again.
        input_scan, output_prediction_scan = self.create_dfs_from_output(
            input_scan, output_prediction_scan, target_scan)
        
        return output_prediction_scan,output_prediction_semantics   

    def create_dfs_from_output(self, input_sdf, output_df, target_scan):
      """Rescales model output to distance fields (in voxel units)."""
      input_sdf = (input_sdf[0, :, :, :, 0].astype(np.float32) + 1
                  ) * 0.5 * constants.TRUNCATION
      if FLAGS.p_norm > 0:
        factor = 0.5 if target_scan is not None else 1.0
        output_df = factor * constants.TRUNCATION * (
            output_df[0, :, :, :, 0] + 1)
      else:
        output_df = (output_df[0, :, :, :, 0] + 1) * 0.5 * (
            FLAGS.num_quant_levels - 1)
        output_df = util.dequantize(output_df, FLAGS.num_quant_levels,
                                    constants.TRUNCATION)
      return input_sdf, output_df
    def predict_from_model(self, logit_groups_geometry, logit_groups_semantics,
                           temperature):
      """Reconstruct predicted geometry and semantics from model output."""
      predictions_geometry_list = []
      for logit_group in logit_groups_geometry:
        if FLAGS.p_norm > 0:
          predictions_geometry_list.append(logit_group[:, :, :, :, 0])
        else:
          logit_group_shape = logit_group.shape_as_list()
          logit_group = tf.reshape(logit_group, [-1, logit_group_shape[-1]])
          samples = tf.multinomial(temperature * logit_group, 1)
          predictions_geometry_list.append(
              tf.reshape(samples, logit_group_shape[:-1]))
      predictions_semantics_list = []
      if FLAGS.predict_semantics:
        for logit_group in logit_groups_semantics:
          predictions_semantics_list.append(tf.argmax(logit_group, 4))
      else:
        predictions_semantics_list = [
            tf.zeros(shape=predictions_geometry_list[0].shape, dtype=tf.uint8)
        ] * len(predictions_geometry_list)
      return predictions_geometry_list, predictions_semantics_list
  
    def create_model(self, scene_dim_x, scene_dim_y, scene_dim_z):
      """Init model graph for scene."""
      input_placeholder = tf.placeholder(
          tf.float32,
          shape=[1, scene_dim_z, scene_dim_y, scene_dim_x, 2],
          name='pl_scan')
      target_placeholder = tf.placeholder(
          tf.float32,
          shape=[1, scene_dim_z, scene_dim_y, scene_dim_x, 2],
          name='pl_target')
      target_lo_placeholder = tf.placeholder(
          tf.float32,
          shape=[1, scene_dim_z // 2, scene_dim_y // 2, scene_dim_x // 2, 2],
          name='pl_target_lo')
      target_sem_placeholder = tf.placeholder(
          tf.uint8,
          shape=[1, scene_dim_z, scene_dim_y, scene_dim_x],
          name='pl_target_sem')
      target_sem_lo_placeholder = tf.placeholder(
          tf.uint8,
          shape=[1, scene_dim_z // 2, scene_dim_y // 2, scene_dim_x // 2],
          name='pl_target_sem_lo')
      # No previous level input if at base level.
      if FLAGS.is_base_level:
        target_scan_low_resolution = None
        target_semantics_low_resolution = None
      else:
        target_scan_low_resolution = target_lo_placeholder
        target_semantics_low_resolution = target_sem_lo_placeholder
      logits = model.model(
          input_scan=input_placeholder,
          target_scan_low_resolution=target_scan_low_resolution,
          target_scan=target_placeholder,
          target_semantics_low_resolution=target_semantics_low_resolution,
          target_semantics=target_sem_placeholder,
          num_quant_levels=FLAGS.num_quant_levels,
          predict_semantics=FLAGS.predict_semantics,
          use_p_norm=FLAGS.p_norm > 0)
      return (input_placeholder, target_placeholder, target_lo_placeholder,
              target_sem_placeholder, target_sem_lo_placeholder, logits)

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
def Feature(sdf, gt, gt_df, hierarchy_level=1):
    feature = tf.train.Example(features=tf.train.Features(feature=get_dict(sdf,gt,gt_df,hierarchy_level-1)))
    return feature.SerializeToString()
    
def process(pred, path, path_out, path_pred=None):
    counter=0
    
    with tf.io.TFRecordWriter(path_out) as writer:
        if FLAGS.is_base_level:
            for record in tf.python_io.tf_record_iterator(path):
                counter += 1
                
                example = tf.train.Example()
                example.ParseFromString(record)
                feature_map = example.features
                
                # get raw data from input file
                serialization=dict()
                serialization['data'] = _bytes_feature(tf.train.Example(features=feature_map).SerializeToString())
    
                if FLAGS.hierarchy_level > len(_RESOLUTIONS):
                    raise RuntimeError("out of range")
                key_samples = 'samples_' + _RESOLUTIONS[FLAGS.hierarchy_level-1] 
                key_samples_sem = 'sem_samples_' + _RESOLUTIONS[FLAGS.hierarchy_level-1] 
                
                # predict 
                ## load data at previous leve
                (input_scan, target_scan, target_semantics, prediction_scan_low_resolution,
                  prediction_semantics_low_resolution) = read_inputs(
                  FLAGS.hierarchy_level-1, feature_map, _DIMS[FLAGS.hierarchy_level-1], 0,
                  FLAGS.num_quant_levels, FLAGS.p_norm, FLAGS.predict_semantics, processing=1,
                  shape=[ _DIMS[FLAGS.hierarchy_level-1], _DIMS[FLAGS.hierarchy_level-1], _DIMS[FLAGS.hierarchy_level-1]])
                ## predict
                output_prediction_scan,output_prediction_semantics = \
                    pred.predict(input_scan, target_scan, target_semantics, prediction_scan_low_resolution, 
                      prediction_semantics_low_resolution)
                    
                serialization[key_samples] = _float_feature(output_prediction_scan.ravel())
                serialization[key_samples_sem] =_bytes_feature(output_prediction_semantics.tobytes())
                        
                feature = tf.train.Example(features=tf.train.Features(feature=serialization))
                
                
                writer.write(feature.SerializeToString())
               
                if FLAGS.debug is True:
                    print('counter: ', counter)
                    if counter > 10:
                        break
            print('counter:',counter)
        else:
            for record, record_pred in zip(tf.python_io.tf_record_iterator(path),\
                                           tf.python_io.tf_record_iterator(path_pred)):
                counter += 1
                
                example = tf.train.Example()
                example.ParseFromString(record)
                feature_map = example.features
                example.ParseFromString(record_pred)
                feature_map_pred = example.features
                
                # get raw data from input file
                serialization=dict()
                serialization['data'] = _bytes_feature(tf.train.Example(features=feature_map).SerializeToString())
    
                if FLAGS.hierarchy_level > len(_RESOLUTIONS):
                    raise RuntimeError("out of range")
                key_samples = 'samples_' + _RESOLUTIONS[FLAGS.hierarchy_level-1] 
                key_samples_sem = 'sem_samples_' + _RESOLUTIONS[FLAGS.hierarchy_level-1] 
                
                # predict 
                ## load data at previous level
                (input_scan, target_scan, target_semantics, prediction_scan_low_resolution,
                  prediction_semantics_low_resolution) = read_inputs(
                  FLAGS.hierarchy_level-1, feature_map, _DIMS[FLAGS.hierarchy_level-1], 0,
                  FLAGS.num_quant_levels, FLAGS.p_norm, FLAGS.predict_semantics, processing=1,
                  shape=[ _DIMS[FLAGS.hierarchy_level-1], _DIMS[FLAGS.hierarchy_level-1], _DIMS[FLAGS.hierarchy_level-1]])
                
                prediction_scan_low_resolution,prediction_semantics_low_resolution = read_prediction(
                    FLAGS.hierarchy_level, feature_map_pred, FLAGS.num_quant_levels, FLAGS.p_norm, FLAGS.predict_semantics,
                    shape=[ _DIMS[FLAGS.hierarchy_level], _DIMS[FLAGS.hierarchy_level], _DIMS[FLAGS.hierarchy_level]])
                      
                      ## predict
                output_prediction_scan,output_prediction_semantics = \
                    pred.predict(input_scan, target_scan, target_semantics, prediction_scan_low_resolution, 
                      prediction_semantics_low_resolution)
                    
                serialization[key_samples] = _float_feature(output_prediction_scan.ravel())
                serialization[key_samples_sem] =_bytes_feature(output_prediction_semantics.tobytes())
                        
                feature = tf.train.Example(features=tf.train.Features(feature=serialization))
                
                
                writer.write(feature.SerializeToString())
               
                if FLAGS.debug is True:
                    print('counter: ', counter)
                    if counter > 10:
                        break
            print('counter:',counter)
    
    
   
if __name__ == '__main__':
    pred = Prediction([ _DIMS[FLAGS.hierarchy_level-1],  _DIMS[FLAGS.hierarchy_level-1],  _DIMS[FLAGS.hierarchy_level-1]], 
                  FLAGS.model_path)
    
    if os.path.isfile(FLAGS.input_dir):
        print('Single Process')
        in_path = FLAGS.input_dir
        out_path = FLAGS.output_dir
        in_path_pred = FLAGS.predict_dir
        print('input:', in_path)
        print('in_path_pred:', in_path_pred)
        print('out_path:', out_path)
        
        process(pred, in_path,out_path,in_path_pred)
    else:
        print('Batch Process')
        input_folder_names = sorted(os.listdir(FLAGS.input_dir))
        createFolder(FLAGS.output_dir)
        
        for i in range(len(input_folder_names)):
            number = re.findall('\d+',input_folder_names[i]) 
            output_file_name = 'train_{}.tfrecords'.format(number[0])
            
            in_path = os.path.join(FLAGS.input_dir,input_folder_names[i])
            out_path = os.path.join(FLAGS.output_dir, output_file_name)
            in_path_pred = os.path.join(FLAGS.predict_dir, input_folder_names[i])
            
            print('input:', in_path)
            print('in_path_pred:', in_path_pred)
            print('out_path:', out_path)
            
            process(pred, in_path,out_path,in_path_pred)
        
            if FLAGS.debug:
                break
      
        
        