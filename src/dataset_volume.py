if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
# import torch
import numpy as np
# from torch.utils.data import DataLoader

def scene_model_id_pair(path, dataset_portion=[0]):
    '''
    Load sceneId, model names
    '''

    scene_name_pair = []  # full path of the objs files

    model_path = path
#    models = os.listdir(model_path)

    foo = [1]
    for root, dirs, files in os.walk(os.path.abspath(model_path)):
        diff = root[len(model_path)::]
        folder_name = ''
        for file in files:
            if(os.path.isdir(file)):
                folder_name = file
            else:
                scene_name_pair.extend([(model_path, os.path.join(diff,folder_name, file)) for file__ in foo])

#    scene_name_pair.extend([(model_path, model_id) for model_id in models])

    num_models = len(scene_name_pair)
    portioned_scene_name_pair = scene_name_pair[int(num_models *
                                                    dataset_portion[0]):]

    return portioned_scene_name_pair

def checkEnd(x):
    return x + '/' if x[-1] != '/' else x

def checkStart(x):
    return x[1:] if x[0] == '/' else x

class Dataset():
    def __init__(self, volume_path, gt_path, mask_path=None):
        
        self.input_base_folder = checkEnd(os.path.abspath(volume_path))
        self.gt_base_folder = checkEnd(os.path.abspath(gt_path))
        self.mask_base_folder = checkEnd(os.path.abspath(mask_path)) if mask_path is not None else None
        
        self.data = scene_model_id_pair(self.input_base_folder)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index][0], self.data[index][1])
            item = self.load_item(0)

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name[1])

    def load_item(self, index):
        volume = self.get_volume(index)
        gt = self.get_gt(index)
        
        if self.mask_base_folder is not None:
            mask = self.get_mask(index)
            return volume, gt, mask
        else:
            return volume,gt,None

    def get_volume(self, index):
        _, model_id = self.data[index]
        # print(self.input_base_folder+checkStart(model_id))
        return np.load(self.input_base_folder+checkStart(model_id))

    def get_gt(self, index):
        _, model_id = self.data[index]
        return np.load(self.gt_base_folder+checkStart(model_id))
    
    def get_mask(self, index):
        _, model_id = self.data[index]
        return np.load(self.mask_base_folder+checkStart(model_id))
    
    

if __name__ == '__main__':
    sdf_path = '/media/sc/BackupDesk/TrainingData_TSDF/SceneNet_train_188/train/scenenet_rgbd_train_0'
    gt_path = '/media/sc/BackupDesk/TrainingData_TSDF/SceneNet_train_188/gt/scenenet_rgbd_train_0'
    mask_path = '/media/sc/BackupDesk/TrainingData_TSDF/SceneNet_train_188/mask/scenenet_rgbd_train_0'
    dataset = Dataset(sdf_path,gt_path,mask_path)
    
    volume, gt, mask = dataset.__getitem__(0)
    print('volume', volume.shape, 'gt',gt.shape, 'mask',mask.shape)
    
  
    # print('Go through all data...')
    # for items in train_loader:
    #     volume, gt = items
    # print('done!')