import numpy as np
import tensorflow as tf
import constants
import util
import model_predict
import metrics_ssc

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

flags.DEFINE_string('input_dir', '/media/sc/SSD1TB/test_SceneNetRGBD_3_level/eval.tfrecords',
                    'Directory to input TFRecords.')
flags.DEFINE_string('predict_dir', '/media/sc/SSD1TB/test_SceneNetRGBD_3_level/pred_level_3.tfrecords',
                    'Directory to input TFRecords.')
flags.DEFINE_string('output_dir', '/media/sc/SSD1TB/test_SceneNetRGBD_3_level/pred_level_3.tfrecords', '')
flags.DEFINE_string('output_eva', '/media/sc/SSD1TB/test_SceneNetRGBD_3_level//eva.txt','')
flags.DEFINE_string('model_path', '/home/sc/research/ScanComplete/train/train_v003', '')
flags.DEFINE_string('model_checkpoint', '',
                    'Model checkpoint to use (empty for latest).')
flags.DEFINE_integer('height_input', 16, 'Input block y dim.')
flags.DEFINE_integer('class_num', 14, '')
flags.DEFINE_integer('hierarchy_level', 3, 'Hierachy level (1: finest level).')
flags.DEFINE_integer('num_quant_levels', 256, 'Number of quantization bins.')
flags.DEFINE_bool('is_base_level', False, 'If base level of hierarchy.')
flags.DEFINE_integer('pad_test', 0, 'Scene padding.')
flags.DEFINE_integer('p_norm', 1, 'P-norm loss (0 to disable).')
flags.DEFINE_bool('predict_semantics', True,
                  'Also predict semantic labels per-voxel.')
flags.DEFINE_integer('debug', 0, '')
flags.DEFINE_float('temperature', 100.0, 'Softmax temperature for sampling.')


# def _bytes_feature(value):
#   """Returns a bytes_list from a string / byte."""
#   if isinstance(value, type(tf.constant(0))):
#     value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
#   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# def _float_feature(value):
#   """Returns a float_list from a float / double."""
#   return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# def _floats_feature(value):
#   return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

# def _int64_feature(value):
#   """Returns an int64_list from a bool / enum / int / uint."""
#   return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

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
    if not FLAGS.is_base_level:
        assert prev_feature_map is not None
      
        prediction_scan_low_resolution = read_input_float_feature(
            prev_feature_map, _PREDICT_DF, None)
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
              prev_feature_map, _PREDICT_SEM,
              [prev_scene_dim_z, prev_scene_dim_y, prev_scene_dim_x])
          prediction_semantics_low_resolution = prediction_semantics_low_resolution[
              offset_z:offset_z + scene_dim_z // 2, :scene_dim_y // 2, offset_x:
              offset_x + scene_dim_x // 2]
      
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
    
def process(path, path_out, path_pred=None):
    counter=0
    pred = None
    
    ious = np.zeros(FLAGS.class_num)
    accs = np.zeros(FLAGS.class_num)
    recalls = np.zeros(FLAGS.class_num)
    IoU = metrics_ssc.IoU(FLAGS.class_num)
    RoC = metrics_ssc.PerClassAccRecall(FLAGS.class_num)
    
    with tf.io.TFRecordWriter(path_out) as writer:
        if FLAGS.is_base_level:
            for record in tf.python_io.tf_record_iterator(path):
                counter += 1
                example = tf.train.Example()
                example.ParseFromString(record)
                feature_map = example.features
                
                # get raw data from input file
                serialization=dict()
                
                if FLAGS.hierarchy_level > len(_RESOLUTIONS):
                    raise RuntimeError("out of range")
                
                # predict 
                ## load data at previous leve
                (input_scan, target_scan, target_semantics, prediction_scan_low_resolution,
                  prediction_semantics_low_resolution) = read_inputs(
                  FLAGS.hierarchy_level-1, feature_map, None, FLAGS.height_input, 0,
                  FLAGS.num_quant_levels, FLAGS.p_norm, FLAGS.predict_semantics, processing=1)
                      
                #print(input_scan.shape)
                if pred is None:
                    pred = model_predict.Prediction(input_scan.shape,
                         FLAGS.model_path, FLAGS.temperature, FLAGS.model_checkpoint, constants.TRUNCATION,
                         FLAGS.p_norm, FLAGS.num_quant_levels, FLAGS.predict_semantics, FLAGS.is_base_level)
                ## predict
                output_prediction_scan,output_prediction_semantics = \
                    pred.predict(input_scan, target_scan, target_semantics, prediction_scan_low_resolution, 
                      prediction_semantics_low_resolution)
                    
                serialization[_PREDICT_DF+'/dim'] = util.int64_feature(output_prediction_scan.shape)
                serialization[_PREDICT_DF] = util.float_feature(output_prediction_scan.flatten().tolist())
                if FLAGS.predict_semantics:
                    serialization[_PREDICT_SEM] = util.bytes_feature(output_prediction_semantics.flatten().tobytes())
                    iou, inter, union = IoU(output_prediction_semantics,target_semantics)
                    acc, recall, inter2,gt_sum,pred_sum = RoC(output_prediction_semantics, target_semantics)
                    ious += iou
                    accs += acc
                    recalls += recall
                    # print(iou)
                    # print(acc)
                    # print(recall)
                    
                feature = tf.train.Example(features=tf.train.Features(feature=serialization))
                writer.write(feature.SerializeToString())
               
                
                
                if FLAGS.debug > 0:
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
                
                if FLAGS.hierarchy_level > len(_RESOLUTIONS):
                    raise RuntimeError("out of range")
               
                # predict 
                ## load data at previous level
                (input_scan, target_scan, target_semantics, prediction_scan_low_resolution,
                  prediction_semantics_low_resolution) = read_inputs(
                  FLAGS.hierarchy_level-1, feature_map, feature_map_pred, FLAGS.height_input, 0,
                  FLAGS.num_quant_levels, FLAGS.p_norm, FLAGS.predict_semantics, processing=1)
                      
                if pred is None:
                    pred = model_predict.Prediction(input_scan.shape,
                         FLAGS.model_path, FLAGS.temperature, FLAGS.model_checkpoint, constants.TRUNCATION,
                         FLAGS.p_norm, FLAGS.num_quant_levels, FLAGS.predict_semantics, FLAGS.is_base_level)
                      
                ## predict
                output_prediction_scan,output_prediction_semantics = \
                    pred.predict(input_scan, target_scan, target_semantics, prediction_scan_low_resolution, 
                      prediction_semantics_low_resolution)
                    
                serialization[_PREDICT_DF+'/dim'] = util.int64_feature(output_prediction_scan.shape)
                serialization[_PREDICT_DF] = util.float_feature(output_prediction_scan.flatten().tolist())
                if FLAGS.predict_semantics:
                    serialization[_PREDICT_SEM] = util.bytes_feature(output_prediction_semantics.flatten().tobytes())
                    iou, _, _ = IoU(output_prediction_semantics,target_semantics)
                    acc, recall, _,_,_ = RoC(output_prediction_semantics, target_semantics)
                    ious += iou
                    accs += acc
                    recalls += recall
                    
                feature = tf.train.Example(features=tf.train.Features(feature=serialization))
                writer.write(feature.SerializeToString())
                
                
                
                if FLAGS.debug > 0:
                    print('counter: ', counter)
                    if counter > 10:
                        break
            print('counter:',counter)
    return ious/counter, accs/counter,recalls/counter
    
   
if __name__ == '__main__':
    # if os.path.isfile(FLAGS.input_dir):
    print('Single Process')
    in_path = FLAGS.input_dir
    out_path = FLAGS.output_dir
    in_path_pred = FLAGS.predict_dir
    print('input:', in_path)
    print('in_path_pred:', in_path_pred)
    print('out_path:', out_path)
    
    ious, accs, recalls = process(in_path,out_path,in_path_pred)
   
    
    def numpy_to_string(x):
            string = ''
            for n in x:
                string += '%5.3f\t' % n
            return string
    def formatString(inpuy):
        return '{}{}'.format(numpy_to_string(inpuy), '%5.3f' % np.mean(inpuy))

    classes = "{:>5.5}\t".format('Metrics')
    for name in metrics_ssc.NYU14_name_list:
        classes += '{:>5.5}\t'.format(name)
    classes += '{:>5.5}'.format('Mean')
    print('')
    print(classes)
    print('IoU:\t{}'.format(formatString( ious)))
    print('Acc:\t{}'.format(formatString(accs)))
    print('Recall:\t{}'.format(formatString(recalls)))
    with open(FLAGS.output_eva, 'w+') as f:
        f.write('{}\n'.format(classes))
        f.write('IoU:\t{}\n'.format(formatString(ious)))
        f.write('Acc:\t{}\n'.format(formatString(accs)))
        f.write('Rec:\t{}\n'.format(formatString(recalls)))