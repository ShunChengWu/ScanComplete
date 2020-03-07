#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 22:07:23 2020

@author: sc
"""
import os

def process(pth_in,pth_out,patten):
    files = os.listdir(dirname)
    with open(pth_out, 'w+') as f:
        f.write('Metri\tUnkno\tBed\tBooks\tCeili\tChair\tFloor\tFurni\tObjec\tPictu\tSofa\tTable\tTV\tWall\tWindo\tMean\n')
    filtered = [n for n in files if n.find(patten) >0]
    results=dict()
    for file in filtered:
        number=file.replace('_', ' ').split()[0]
        # print(number, file)
        # break
        with open(os.path.join(dirname,file), 'r') as f:
            lines = f.readlines()
            if len(lines) == 2:
                line = lines[1]
                tokens = line.split()[1:]
                results[str(number)] = tokens
        # break
    with open(output_name, 'a+') as fout:
        for name, tokens in results.items():
            print(name,tokens)
            fout.write('{}\t'.format(str(name)))
            for token in tokens:
                fout.write('{}\t'.format(str(token)))
            fout.write('\n')
                          
if __name__ == '__main__':
    dirname = '/media/sc/SSD1TB/Evaluation_ScanComplete/SceneNetRGBD_3_level_pred/eval_pred_1'
    
    patten = '_IoU.txt'
    output_name = '/media/sc/SSD1TB/Evaluation_ScanComplete/SceneNetRGBD_3_level_pred/eval_pred_1/All' + patten
    process(dirname, output_name,patten)
    
    patten = '_Acc.txt'
    output_name = '/media/sc/SSD1TB/Evaluation_ScanComplete/SceneNetRGBD_3_level_pred/eval_pred_1/All' + patten
    process(dirname, output_name,patten)
    
    patten = '_Recall.txt'
    output_name = '/media/sc/SSD1TB/Evaluation_ScanComplete/SceneNetRGBD_3_level_pred/eval_pred_1/All' + patten
    process(dirname, output_name,patten)