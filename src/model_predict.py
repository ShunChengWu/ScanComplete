import tensorflow as tf
import os
import numpy as np
import util
import model

class Prediction():
    def __init__(self,dims,model_path, temperature, model_checkpoint, TRUNCATION,
                 p_norm, num_quant_levels,predict_semantics, is_base_level):
        self.dims = dims
        self.TRUNCATION = TRUNCATION
        self.p_norm = p_norm
        self.num_quant_levels = num_quant_levels
        self.predict_semantics = predict_semantics
        self.is_base_level = is_base_level
        
        (input_placeholder, target_placeholder, target_lo_placeholder,
             target_sem_placeholder, target_sem_lo_placeholder, logits) = self.create_model(dims[0], dims[1], dims[2])
        
        logit_groups_geometry = logits['logits_geometry']
        logit_groups_semantics = logits['logits_semantics']
        feature_groups = logits['features']
          
        predictions_geometry_list, predictions_semantics_list =self.predict_from_model(
        logit_groups_geometry, logit_groups_semantics, temperature)
        
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
        
        if model_checkpoint:
          checkpoint_path = os.path.join(model_path, model_checkpoint)
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
        output_prediction_scan[:, :, :, :, 0] *= self.TRUNCATION
        output_prediction_semantics = np.zeros(
            shape=[1, self.dims[2], self.dims[1], self.dims[0]], dtype=np.uint8)
    
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
          if self.p_norm == 0:
            prediction_geometry = prediction_geometry.astype(np.float32) / (
                (self.num_quant_levels - 1) / 2.0) - 1.0
    
          util.assign_voxel_group(output_prediction_scan, prediction_geometry,
                                  n + 1)
          if self.predict_semantics:
            util.assign_voxel_group(output_prediction_semantics,
                                    prediction_semantics, n + 1)
    
        # Final outputs.
        output_prediction_semantics = output_prediction_semantics[0]
        # Make distances again.
        input_scan, output_prediction_scan = self.create_dfs_from_output(
            input_scan, output_prediction_scan, target_scan)
        
        return output_prediction_scan,output_prediction_semantics, input_scan

    def create_dfs_from_output(self, input_sdf, output_df, target_scan):
      """Rescales model output to distance fields (in voxel units)."""
      input_sdf = (input_sdf[0, :, :, :, 0].astype(np.float32) + 1
                  ) * 0.5 * self.TRUNCATION
      if self.p_norm > 0:
        factor = 0.5 if target_scan is not None else 1.0
        output_df = factor * self.TRUNCATION * (
            output_df[0, :, :, :, 0] + 1)
      else:
        output_df = (output_df[0, :, :, :, 0] + 1) * 0.5 * (
            self.num_quant_levels - 1)
        output_df = util.dequantize(output_df, self.num_quant_levels,
                                    self.TRUNCATION)
      return input_sdf, output_df
    def predict_from_model(self, logit_groups_geometry, logit_groups_semantics,
                           temperature):
      """Reconstruct predicted geometry and semantics from model output."""
      predictions_geometry_list = []
      for logit_group in logit_groups_geometry:
        if self.p_norm > 0:
          predictions_geometry_list.append(logit_group[:, :, :, :, 0])
        else:
          logit_group_shape = logit_group.shape_as_list()
          logit_group = tf.reshape(logit_group, [-1, logit_group_shape[-1]])
          samples = tf.multinomial(temperature * logit_group, 1)
          predictions_geometry_list.append(
              tf.reshape(samples, logit_group_shape[:-1]))
      predictions_semantics_list = []
      if self.predict_semantics:
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
      if self.is_base_level:
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
          num_quant_levels=self.num_quant_levels,
          predict_semantics=self.predict_semantics,
          use_p_norm=self.p_norm > 0)
      return (input_placeholder, target_placeholder, target_lo_placeholder,
              target_sem_placeholder, target_sem_lo_placeholder, logits)
