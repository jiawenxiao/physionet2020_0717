#!/usr/bin/env python

import numpy as np, os, sys
import joblib
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_12ECG_features(data,header_data):
    set_length=5000
    
    data_num=np.zeros((1,12,set_length))
    data_external=np.zeros((1,2))
    length=data.shape[1]
    if length>=set_length:
        data_num[:,:,:]=data[:,:set_length]/30000
    else:
        data_num[:,:,:length]=data/30000
      
    for lines in header_data:
        if lines.startswith('#Age'):
            age=lines.split(': ')[1].strip()
            if age=='NaN':
                age='60'     
        if lines.startswith('#Sex'):
            sex=lines.split(': ')[1].strip()
           
    data_external[:,0]=float(age)/100
    data_external[:,1]=np.array(sex=='Male').astype(int) 
    
    return data_num,data_external



def load_12ECG_model(input_directory):
    # load the model from disk 
    f_out='resnet_0628.pkl'
    filename = os.path.join(input_directory,f_out)
    loaded_model = torch.load(filename,map_location=device)
    return loaded_model

def run_12ECG_classifier(data,header_data,classes,model):   
    num_classes = len(classes)
    
    # Use your classifier here to obtain a label and score for each class. 
    feats_reshape,feats_external = get_12ECG_features(data,header_data)
    
    feats_reshape = torch.tensor(feats_reshape,dtype=torch.float,device=device)
    feats_external = torch.tensor(feats_external,dtype=torch.float,device=device)
    
    
    pred = model.forward(feats_reshape,feats_external)
    pred = torch.sigmoid(pred)
    
 
    current_score = pred.squeeze().cpu().detach().numpy()    
    current_label= np.where(current_score>0.2,1,0)
 
    num_positive_classes = np.sum(current_label)

    
    #窦性心律标签处于有评分的标签排序后的第14位
    normal_index=classes.index('426783006')

    ##最多有三个1
    if num_positive_classes>3:
        sort_index=np.argsort(current_score)
        min_index=sort_index[:24]
        current_label[min_index]=0 
        
       
    ##至少为一个标签，如果所有标签都没有，就设置为窦性心律,现在窦性心律可以和其他标签共存        
    if num_positive_classes==0:
        current_label[normal_index] = 1
    
    return current_label, current_score

