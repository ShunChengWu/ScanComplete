"""Applies model to a single test scene and exports results.

(Note: uses matlab to convert predicted complete dfs to meshes.)
"""

import os
import numpy as np
import tensorflow as tf

import constants
import model
import util
from reader import _RESOLUTIONS, _INPUT_FEATURE, _TARGET_FEATURE, _TARGET_SEM_FEATURE 

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('base_dir', '',
                    'Root directory. Expects a directory containing the model.')
flags.DEFINE_string('input_scene', '',
                    'Path to file containing scenes encoded as TFRecord.')
flags.DEFINE_string('model_checkpoint', '',
                    'Model checkpoint to use (empty for latest).')
flags.DEFINE_string('output_dir_prev', '', 'Output folder for previous level.')
flags.DEFINE_string('output_folder', '/tmp/vis',
                    'Folder in which to put the results.')
# Parameters for applying model.
flags.DEFINE_integer('height_input', 64, 'Input block y dim.')
flags.DEFINE_integer('hierarchy_level', 1, 'Hierachy level (1: finest level).')
flags.DEFINE_bool('is_base_level', True, 'If base level of hierarchy.')
flags.DEFINE_integer('num_quant_levels', 256, 'Number of quantization bins.')
flags.DEFINE_integer('num_total_hierarchy_levels', 1,
                     'Number of total hierarchy levels.')
flags.DEFINE_integer('pad_test', 0, 'Scene padding.')
flags.DEFINE_integer('p_norm', 1, 'P-norm loss (0 to disable).')
flags.DEFINE_bool('predict_semantics', False,
                  'Also predict semantic labels per-voxel.')
flags.DEFINE_float('temperature', 100.0, 'Softmax temperature for sampling.')


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
                predict_semantics):
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
  # counter=0
  # for record in tf.python_io.tf_record_iterator(filename):
  #         counter += 1
  #         example = tf.train.Example()
  #         example.ParseFromString(record)
  #         feature_map = example.features
  # print('counter:',counter)
    
  key_input = _RESOLUTIONS[hierarchy_level - 1] + '_' + _INPUT_FEATURE
  key_target = _RESOLUTIONS[hierarchy_level - 1] + '_' + _TARGET_FEATURE
  key_target_sem = _RESOLUTIONS[hierarchy_level - 1] + '_' + _TARGET_SEM_FEATURE
  # key_input = _INPUT_FEATURE
  # key_target = _TARGET_FEATURE
  # key_target_sem = _TARGET_SEM_FEATURE
  # print('key_input', key_input)
    
  # Input scan as sdf.
  input_scan = read_input_float_feature(feature_map, key_input, shape=[16,16,16])
  (scene_dim_z, scene_dim_y, scene_dim_x) = input_scan.shape
  # Target scan as df.
  if key_target in feature_map.feature:
    target_scan = read_input_float_feature(
        feature_map, key_target, [scene_dim_z, scene_dim_y, scene_dim_x])
  if key_target_sem in feature_map.feature:
    target_semantics = read_input_bytes_feature(
        feature_map, key_target_sem, [scene_dim_z, scene_dim_y, scene_dim_x])
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

  # Default values for previous resolution inputs.
  prediction_scan_low_resolution = np.zeros(
      [scene_dim_z // 2, scene_dim_y // 2, scene_dim_x // 2, 2])
  prediction_semantics_low_resolution = np.zeros(
      [scene_dim_z // 2, scene_dim_y // 2, scene_dim_x // 2], dtype=np.uint8)
  if target_semantics is None:
    target_semantics = np.zeros([scene_dim_z, scene_dim_y, scene_dim_x])

  # Load previous level prediction.
  if not FLAGS.is_base_level:
    previous_file = os.path.join(
        FLAGS.output_dir_prev, 'level' + str(FLAGS.hierarchy_level - 1) + '_' +
        os.path.splitext(os.path.basename(filename))[0] + 'pred.tfrecord')
    tf.logging.info('Reading previous predictions frome file: %s',
                    previous_file)
    assert os.path.isfile(previous_file)
    for record in tf.python_io.tf_record_iterator(previous_file):
      prev_example = tf.train.Example()
      prev_example.ParseFromString(record)
      prev_feature_map = prev_example.features
    prediction_scan_low_resolution = read_input_float_feature(
        prev_feature_map, 'prediction_df', None)
    (prev_scene_dim_z, prev_scene_dim_y,
     prev_scene_dim_x) = prediction_scan_low_resolution.shape
    offset_z = (prev_scene_dim_z - scene_dim_z // 2) // 2
    offset_x = (prev_scene_dim_x - scene_dim_x // 2) // 2
    prediction_scan_low_resolution = prediction_scan_low_resolution[
        offset_z:offset_z + scene_dim_z // 2, :scene_dim_y // 2, offset_x:
        offset_x + scene_dim_x // 2]
    prediction_scan_low_resolution = util.preprocess_target_sdf(
        prediction_scan_low_resolution, num_quant_levels, constants.TRUNCATION,
        p_norm == 0)
    if predict_semantics:
      prediction_semantics_low_resolution = read_input_bytes_feature(
          prev_feature_map, 'prediction_sem',
          [prev_scene_dim_z, prev_scene_dim_y, prev_scene_dim_x])
      prediction_semantics_low_resolution = prediction_semantics_low_resolution[
          offset_z:offset_z + scene_dim_z // 2, :scene_dim_y // 2, offset_x:
          offset_x + scene_dim_x // 2]
  return (input_scan, target_scan, target_semantics,
          prediction_scan_low_resolution, prediction_semantics_low_resolution)


def predict_from_model(logit_groups_geometry, logit_groups_semantics,
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


def create_dfs_from_output(input_sdf, output_df, target_scan):
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


def export_prediction_to_example(filename, pred_geo, pred_sem):
  """Saves predicted df/sem to file."""
  with tf.python_io.TFRecordWriter(filename) as writer:
    out_feature = {
        'prediction_df/dim': util.int64_feature(pred_geo.shape),
        'prediction_df': util.float_feature(pred_geo.flatten().tolist())
    }
    if FLAGS.predict_semantics:
      out_feature['prediction_sem'] = util.bytes_feature(
          pred_sem.flatten().tobytes())
    example = tf.train.Example(features=tf.train.Features(feature=out_feature))
    writer.write(example.SerializeToString())


def create_model(scene_dim_x, scene_dim_y, scene_dim_z):
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

class Prediction():
    def __init__(self,dims,model_path):
        self.dims = dims
        
        (input_placeholder, target_placeholder, target_lo_placeholder,
             target_sem_placeholder, target_sem_lo_placeholder, logits) = create_model(dims[0], dims[1], dims[2])
        
        logit_groups_geometry = logits['logits_geometry']
        logit_groups_semantics = logits['logits_semantics']
        feature_groups = logits['features']
          
        predictions_geometry_list, predictions_semantics_list = predict_from_model(
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
        input_scan, output_prediction_scan = create_dfs_from_output(
            input_scan, output_prediction_scan, target_scan)
        
        return output_prediction_scan,output_prediction_semantics
        
def main(_):
    model_path = FLAGS.base_dir
    output_folder = FLAGS.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    dims = 16
    pred = Prediction([dims, dims, dims], model_path)
        
    # Read TFRecords
    hierarchy_level = FLAGS.hierarchy_level
    counter=0
    for record in tf.python_io.tf_record_iterator(FLAGS.input_scene):
        counter += 1
        example = tf.train.Example()
        example.ParseFromString(record)
        feature_map = example.features
        
        # First load the data to figure out sizes of things.
        (input_scan, target_scan, target_semantics, prediction_scan_low_resolution,
         prediction_semantics_low_resolution) = read_inputs(
             FLAGS.input_scene, feature_map, FLAGS.height_input, FLAGS.pad_test,
             FLAGS.num_quant_levels, FLAGS.p_norm, FLAGS.predict_semantics)
        (scene_dim_z, scene_dim_y, scene_dim_x) = input_scan.shape[:3]
    
        print('scene_dim_z, scene_dim_y, scene_dim_x',scene_dim_z, scene_dim_y, scene_dim_x)
  
        output_prediction_scan,output_prediction_semantics = pred.predict(input_scan, target_scan, target_semantics, prediction_scan_low_resolution, 
                  prediction_semantics_low_resolution)
        print(counter)

if __name__ == '__main__':
  tf.app.run(main)