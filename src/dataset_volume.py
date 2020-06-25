if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os
import numpy as np

def scene_model_id_pair(path, dataset_portion=1.0, shuffle=False):
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
     
    if shuffle is True:
        random.shuffle(scene_name_pair)

    num_models = len(scene_name_pair)
    portioned_scene_name_pair = scene_name_pair[int(num_models *
                                                    (1-dataset_portion)):]
    if not shuffle is True:
        portioned_scene_name_pair = sorted(portioned_scene_name_pair)
    return portioned_scene_name_pair

def checkEnd(x):
    return x + '/' if x[-1] != '/' else x

def checkStart(x):
    return x[1:] if x[0] == '/' else x

class Dataset():
    def __init__(self, input_base_folders, folder_names, data_portion=1.0, shuffle=False):
        self.folder_names = folder_names
        self.shuffle = shuffle
   
        if len(input_base_folders) is 0:
            raise RuntimeError('input base folder has size 0')
        
        paths = dict()
        for name in folder_names:
            paths[name] = list()
            
        
        for base_folder in input_base_folders:
            data = scene_model_id_pair(os.path.join(base_folder, folder_names[0]), data_portion)
            for d in data:
                for name in folder_names:    
                    paths[name].append( checkEnd(base_folder) + name + d[1] )
        self.len = len(paths[folder_names[0]])
        self.paths = paths
        #print('paths',paths)
        #for n in folder_names:
        #   print('paths[',n,']',paths[name][0])
        #print('input_base_folders',input_base_folders)
        #print('folder_names',folder_names)
        #print('self.len',self.len)
    def __len__(self):
        return self.len 

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            for name in self.folder_names:
              print('loading error: ' + self.paths[name][index])

        return item

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name[1])

    def load_item(self, index):
        return [np.load(self.paths[name][index]) for name in self.folder_names]

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
