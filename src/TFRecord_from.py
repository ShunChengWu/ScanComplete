import numpy as np
import tensorflow as tf
import constants
import util
import model_predict
import metrics_ssc
import os 

_RESOLUTIONS = ['5cm', '9cm', '19cm']
_INPUT_FEATURE = 'input_sdf'
_TARGET_FEATURE = 'target_df'
_TARGET_SEM_FEATURE = 'target_sem'
_PREDICT_DF = 'prediction_df'
_PREDICT_SEM = 'prediction_sem'
_KEY_DIMS = _PREDICT_DF + "/dim"
threads = 3
TRUNCATION = 3
# p_norm = 1

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir', '/home/sc/research/scslam/cmake-build-debug/App/TrainingDataGenerator/tmp/SceneNetRGBD_3_level_train/train_0.tfrecords',
                    'Directory to input TFRecords.')
flags.DEFINE_string('output_eva', '/home/sc/research/scslam/cmake-build-debug/App/TrainingDataGenerator/tmp/SceneNetRGBD_3_level_train/','')
flags.DEFINE_integer('height_input', 64, 'Input block y dim.')
flags.DEFINE_integer('class_num', 14, '')
flags.DEFINE_integer('hierarchy_level', 1, 'Hierachy level (1: finest level).')
flags.DEFINE_integer('num_quant_levels', 256, 'Number of quantization bins.')
flags.DEFINE_bool('is_base_level', True, 'If base level of hierarchy.')
flags.DEFINE_integer('pad_test', 0, 'Scene padding.')
flags.DEFINE_integer('p_norm', 1, 'P-norm loss (0 to disable).')
flags.DEFINE_bool('predict_semantics', True,
                  'Also predict semantic labels per-voxel.')
flags.DEFINE_float('temperature', 100.0, 'Softmax temperature for sampling.')
flags.DEFINE_integer('debug', 0, '')
flags.DEFINE_integer('mesh',0,'output mesh.')
flags.DEFINE_integer('save_npy',1,'save numpy array.')
flags.DEFINE_integer('write_output',0,'write output to TFRecords.')
flags.DEFINE_integer('target_num', -1,'')
flags.DEFINE_integer('start_from', -1,'')
flags.DEFINE_integer('max_num', -1,'')

def numpy_to_string(x):
        string = ''
        for n in x:
            string += '%5.3f\t' % n
        return string
def formatString(inpuy):
    return '{}{}'.format(numpy_to_string(inpuy), '%5.3f' % np.mean(inpuy))

def export_prediction_to_mesh(outprefix, input_sdf, output_df, output_sem,
                              target_df, target_sem, saveMesh = True):
  print('save_iso_meshes called!')
  """Saves predicted df/sem + input (+ target, if any) to mesh visualization."""
  # Add back (below floor) padding for vis (creates the surface on the bottom).
  (scene_dim_z, scene_dim_y, scene_dim_x) = input_sdf.shape
  save_input_sdf = constants.TRUNCATION * np.ones(
      [scene_dim_z, 2 * FLAGS.pad_test + scene_dim_y, scene_dim_x])
  save_prediction = np.copy(save_input_sdf)
  save_target = None if target_df is None else np.copy(save_input_sdf)
  save_input_sdf[:, FLAGS.pad_test:FLAGS.pad_test + scene_dim_y, :] = input_sdf
  save_prediction[:, FLAGS.pad_test:FLAGS.pad_test + scene_dim_y, :] = output_df
  if target_df is not None:
    save_target[:, FLAGS.pad_test:FLAGS.pad_test + scene_dim_y, :] = target_df
    # For error visualization as colors on mesh.
    save_errors = np.zeros(shape=save_prediction.shape)
    save_errors[:, FLAGS.pad_test:FLAGS.pad_test + scene_dim_y, :] = np.abs(
        output_df - target_df)
  if FLAGS.predict_semantics:
    save_pred_sem = np.zeros(shape=save_prediction.shape, dtype=np.uint8)
    save_pred_sem[:, FLAGS.pad_test:
                  FLAGS.pad_test + scene_dim_y, :] = output_sem
    save_pred_sem[np.greater(save_prediction, 1)] = 0
    if target_sem is not None:
      save_target_sem = np.zeros(shape=save_prediction.shape, dtype=np.uint8)
      save_target_sem[:, FLAGS.pad_test:
                      FLAGS.pad_test + scene_dim_y, :] = target_sem

  # Save as mesh.
  util.save_iso_meshes(
        [save_input_sdf, save_prediction, save_target],
        [None, save_errors, save_errors], [None, save_pred_sem, save_target_sem],
        [
            outprefix + 'input.obj', outprefix + 'pred.obj',
            outprefix + 'target.obj'
        ],
        isoval=1, semantic_only=True)
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

def read_input_float_feature(feature_map, key, shape):
    """Reads a float array from Example proto to np array."""
    if shape is None:
      (dim_z, dim_y, dim_x) = feature_map.feature[key + '/dim'].int64_list.value
      # (dim_x, dim_y, dim_z) = feature_map.feature[key + '/dim'].int64_list.value
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

def export_prediction_to_example(filename, pred_geo, pred_sem):
  """Saves predicted df/sem to file."""
  with tf.python_io.TFRecordWriter(filename) as writer:
    out_feature = {
        _PREDICT_DF+'/dim': util.int64_feature(pred_geo.shape),
        _PREDICT_DF: util.float_feature(pred_geo.flatten().tolist())
    }
    if FLAGS.predict_semantics:
      out_feature[_PREDICT_SEM] = util.bytes_feature(
          pred_sem.flatten().tobytes())
    example = tf.train.Example(features=tf.train.Features(feature=out_feature))
    writer.write(example.SerializeToString())
    


def read_inputs(hierarchy_level, feature_map, prev_feature_map, height, padding, num_quant_levels, p_norm,
                predict_semantics, processing):
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
    input_scan = read_input_float_feature(feature_map, key_input, shape=None)
    (scene_dim_z, scene_dim_y, scene_dim_x) = input_scan.shape
    
    # print('input_scan.shape',input_scan.shape)
    
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
          
        # Default values for previous resolution inputs.
        prediction_scan_low_resolution = np.zeros(
          [scene_dim_z // 2, scene_dim_y // 2, scene_dim_x // 2, 2])
        prediction_semantics_low_resolution = np.zeros(
          [scene_dim_z // 2, scene_dim_y // 2, scene_dim_x // 2], dtype=np.uint8)
        if target_semantics is None:
          target_semantics = np.zeros([scene_dim_z, scene_dim_y, scene_dim_x])

    # print('input_scan.shape',input_scan.shape)

    # Load previous level prediction.
    if not FLAGS.is_base_level:
        assert prev_feature_map is not None
      
        prediction_scan_low_resolution = read_input_float_feature(
            prev_feature_map, _PREDICT_DF, None)
        
        (prev_scene_dim_z, prev_scene_dim_y,
         prev_scene_dim_x) = prediction_scan_low_resolution.shape
        offset_z = (prev_scene_dim_z - scene_dim_z // 2) // 2
        offset_x = (prev_scene_dim_x - scene_dim_x // 2) // 2        
        
        if offset_z < 0:        
            if abs(offset_z)>2:
                pad_first_z=-offset_z/2
                pad_last_z=-offset_z/2
            else:
                pad_first_z=0
                pad_last_z=-offset_z
            prediction_scan_low_resolution = np.pad(prediction_scan_low_resolution, 
                (((pad_first_z,pad_last_z),(0,0),(0,0))),'constant')
        else:
            prediction_scan_low_resolution = prediction_scan_low_resolution[
            offset_z:offset_z + scene_dim_z // 2, :, :]
        if offset_x < 0:
            if abs(offset_x)>2:
                pad_first_x=-offset_x/2
                pad_last_x=-offset_x/2
            else:
                pad_first_x=0
                pad_last_x=-offset_x
            prediction_scan_low_resolution = np.pad(prediction_scan_low_resolution, 
                (((0,0),(0,0),(pad_first_x,pad_last_x))),'constant')
        else:
            prediction_scan_low_resolution = prediction_scan_low_resolution[
            :, :, offset_x:offset_x + scene_dim_x // 2]
        prediction_scan_low_resolution = prediction_scan_low_resolution[
            :, :scene_dim_y // 2, :]
            
        prediction_scan_low_resolution = util.preprocess_target_sdf(
            prediction_scan_low_resolution, num_quant_levels, constants.TRUNCATION,
            p_norm == 0)
        if predict_semantics:
          prediction_semantics_low_resolution = read_input_bytes_feature(
              prev_feature_map, _PREDICT_SEM,
              [prev_scene_dim_z, prev_scene_dim_y, prev_scene_dim_x])
          
        if offset_z < 0:        
            if abs(offset_z)>2:
                pad_first_z=-offset_z/2
                pad_last_z=-offset_z/2
            else:
                pad_first_z=0
                pad_last_z=-offset_z
            prediction_semantics_low_resolution = np.pad(prediction_semantics_low_resolution, 
                (((pad_first_z,pad_last_z),(0,0),(0,0))),'constant')
        else:
            prediction_semantics_low_resolution = prediction_semantics_low_resolution[
            offset_z:offset_z + scene_dim_z // 2, :, :]
        if offset_x < 0:
            if abs(offset_x)>2:
                pad_first_x=-offset_x/2
                pad_last_x=-offset_x/2
            else:
                pad_first_x=0
                pad_last_x=-offset_x
            prediction_semantics_low_resolution = np.pad(prediction_semantics_low_resolution, 
                (((0,0),(0,0),(pad_first_x,pad_last_x))),'constant')
        else:
            prediction_semantics_low_resolution = prediction_semantics_low_resolution[
            :, :, offset_x:offset_x + scene_dim_x // 2]
        prediction_semantics_low_resolution = prediction_semantics_low_resolution[
            :, :scene_dim_y // 2, :]
          # prediction_semantics_low_resolution = prediction_semantics_low_resolution[
          #     offset_z:offset_z + scene_dim_z // 2, :scene_dim_y // 2, offset_x:
          #     offset_x + scene_dim_x // 2]
      
    return (input_scan, target_scan, target_semantics,
            prediction_scan_low_resolution, prediction_semantics_low_resolution)

def get_dict(sdf,gt,gt_df, level):
    key_input = _RESOLUTIONS[level] + '_' + _INPUT_FEATURE
    key_target = _RESOLUTIONS[level] + '_' + _TARGET_FEATURE
    key_target_sem = _RESOLUTIONS[level] + '_' + _TARGET_SEM_FEATURE
    
    serialization = {
        key_input:  util.float_feature(sdf.flatten().tolist()),
        key_target: util.float_feature(gt_df.flatten().tolist()),
        key_target_sem:  util.bytes_feature(gt.flatten().tobytes()),
    }
    return serialization
def Feature(sdf, gt, gt_df, hierarchy_level=1):
    feature = tf.train.Example(features=tf.train.Features(feature=get_dict(sdf,gt,gt_df,hierarchy_level-1)))
    return feature.SerializeToString()
    

def kernel(counter, feature_map):
    if FLAGS.target_num >=0:
        if counter-1 != FLAGS.target_num:
            return
    if FLAGS.start_from >=0:
        if counter-1 < FLAGS.start_from:
            return
    print('counter: ', counter)

    if FLAGS.hierarchy_level > len(_RESOLUTIONS):
        raise RuntimeError("out of range")
    
    # predict 
    ## load data at previous level
    (input_scan, target_scan, target_semantics, prediction_scan_low_resolution,
      prediction_semantics_low_resolution) = read_inputs(
      FLAGS.hierarchy_level-1, feature_map, None, FLAGS.height_input, FLAGS.pad_test,
      FLAGS.num_quant_levels, FLAGS.p_norm, FLAGS.predict_semantics, processing=0)
    if FLAGS.save_npy > 0:        
        outprefix = os.path.join(FLAGS.output_eva, str(counter-1) + '_in.npy') 
        np.save(outprefix, input_scan)
        outprefix = os.path.join(FLAGS.output_eva, str(counter-1) + '_gt.npy') 
        np.save(outprefix, target_semantics)
         
def process(path):
    counter=0
    
    for record in tf.python_io.tf_record_iterator(path):
        counter += 1
        example = tf.train.Example()
        example.ParseFromString(record)
        feature_map = example.features
        # kernel(counter, feature_map, None, dims_pre,ious,accs,recalls,None)
        try:
            kernel(counter, feature_map)
        except:
            print('error during processing sequence', counter)
        if FLAGS.debug > 0:
            print('counter: ', counter)
            if counter >= 0:
                break
        if FLAGS.max_num > 0:
            if counter >= FLAGS.max_num:
                break   
   
if __name__ == '__main__':
    in_path = FLAGS.input_dir
    print('input:', in_path)
    util.createFolder(FLAGS.output_eva)
    
    process(in_path)