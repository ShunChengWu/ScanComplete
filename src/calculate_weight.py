#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 13:35:14 2020

@author: sc
"""
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Calculate weight for each classes.')
parser.add_argument('--pth_in', metavar='path to the input folder', type=str,
                    default='/media/sc/BackupDesk/TrainingDataScanNet_0614_TSDF/050_200/test/gt',required=False)
parser.add_argument('--label_num', type=int,default=12, required=False)
parser.add_argument('--num', type=int,default=-1,required=False)
parser.add_argument('--pth_out', metavar='path to the input folder', type=str,
                    default='./weight.txt',required=False)
args = parser.parse_args()

def calculateWeight(x, class_num):
    return np.abs(1.0 / (np.log(x+1)+1))        

if __name__ == '__main__':
    paths=list()
    print('search all npy files')
    for root, dirs, files in os.walk(os.path.abspath(args.pth_in)):
        # diff = root[len(model_path)::]
        folder_name = ''
        for file in files:
            if(os.path.isdir(file)):
                folder_name = file
            else:
                # print(root, dirs, file)
                paths.append(os.path.join(root,file))
                # break
            # else:
            #     scene_name_pair.extend([(model_path, os.path.join(diff,folder_name, file)) for file__ in foo])
    print('sanity check')
    for file in paths:
        if not os.path.isfile(file):
            raise RuntimeError('args')
    
    print('sum all labels')
    sum_all = np.zeros(args.label_num)
    for file in paths:
        buffer = np.load(file)
        # print(buffer)
        # count labels
        counter = [buffer[buffer==i].size for i in range(args.label_num)]
        sum_ = np.array(counter)
        sum_all += sum_
        # print(sum_all)
        # break
    
    # calculate weight
    print('calculate weight')
    sum_all = calculateWeight(sum_all, args.label_num)
    print('weight:\n',sum_all)
    
    # write to file
    print('write to file')
    with open(args.pth_out, 'w') as f:
        for i in range(args.label_num):
            f.write('{}, '.format(sum_all[i]))
            
        
            