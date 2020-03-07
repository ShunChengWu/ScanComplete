from pytorch_meshing import occupancy_meshing, write_ply
import os 
import torch
import numpy as np


if __name__ == '__main__':
    folder='/home/sc/research/scslam/cmake-build-debug/App/TrainingDataGenerator/tmp/188/gt/scenenet_rgbd_train_0'
    files = sorted(os.listdir(folder))
    files = [os.path.join(folder,n) for n in files if n.find('.npy') >0]
    print(files)
    for file in files:
        
        # file = '/media/sc/SSD1TB/Evaluation_ScanComplete/SceneNetRGBD_3_level_pred/eval_pred_1/475_pd.npy'
        volume = torch.from_numpy(np.load(file))
        def Meshing(path, volume, label, threshold):
                    # print('volume.shape',volume.shape)
                    # print('label.shape',label.shape)
                    vertices, faces, colors = occupancy_meshing(volume,
                                                                label,threshold=threshold)
                    
                    print('saving...')
                    # print('vertices.shape:',vertices.shape)
                    # print('faces.shape:',faces.shape)
                    # print('colors.shape:',colors.shape)
                    write_ply(path,vertices,faces,colors)
                    print('\nsaving sample ' + path)
                    return vertices, faces, colors
        name = os.path.join(folder, os.path.splitext(os.path.basename(file))[0] + ".ply")
        Meshing(name, volume, volume,0.5)
        # break
        