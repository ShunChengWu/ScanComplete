from pytorch_meshing import occupancy_meshing, write_ply
import os
import torch
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--pth_in', metavar='path to the input folder', type=str,
                    default='/media/sc/BackupDesk/Reconstruction_From_GT/SLAM_load',required=False)
parser.add_argument('--tsdf', metavar='path to the input folder', type=bool,
                    default='tsdf',required=False)
args = parser.parse_args()

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

if __name__ == '__main__':
    folder=args.pth_in#'/media/sc/BackupDesk/TrainingData_TSDF_0311/SceneNetRGBD_3_level_1_eval/train_0_eval_1'
    
    #folder='/home/sc/research/scslam/cmake-build-debug/App/TrainingDataGenerator/tmp/SceneNetRGBD_3_level_train'
    output_folder=os.path.join(folder,'mesh')
    createFolder(output_folder)
    files = sorted(os.listdir(folder))
    files = [os.path.join(folder,n) for n in files if n.find('.npy') >0]
    # print(files)
    for file in files:
        
        # file = '/media/sc/SSD1TB/Evaluation_ScanComplete/SceneNetRGBD_3_level_pred/eval_pred_1/475_pd.npy'
        volume = torch.from_numpy(np.load(file))
        def Meshing(path, volume, label, threshold):
            # print('volume.shape',volume.shape)
            # print('label.shape',label.shape)
            if args.tsdf:
              if path.find('_in.ply') > 0:
                  vertices, faces, colors = occupancy_meshing(torch.abs(volume) < 0.5,
                                                         label,threshold=threshold)
              else:
                  # return
                  vertices, faces, colors = occupancy_meshing(volume>0,
                                                          label,threshold=threshold)
            else:
                # return
                vertices, faces, colors = occupancy_meshing(volume,
                                                        label,threshold=threshold)
            
            print('saving...')
            # print('vertices.shape:',vertices.shape)
            # print('faces.shape:',faces.shape)
            # print('colors.shape:',colors.shape)
            write_ply(path,vertices,faces,colors)
            print('\nsaving sample ' + path)
            return vertices, faces, colors
        name = os.path.join(output_folder, os.path.splitext(os.path.basename(file))[0] + ".ply")
        Meshing(name, volume, volume,0.5)
        # break
        
