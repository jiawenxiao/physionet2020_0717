#!/usr/bin/env python

import numpy as np, os, sys, joblib
from scipy.io import loadmat
from get_12ECG_features import get_12ECG_features

import pandas as pd
import os,time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from config import config
import utils
# from resnet import  ECGNet
import warnings

warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(666)
torch.cuda.manual_seed(666)

class BasicBlock1d(nn.Module):

    def __init__(self, inplanes, planes, stride, size,downsample):
        super(BasicBlock1d, self).__init__()

        
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=size, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        
        self.conv2 = nn.Conv1d( planes, planes, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        
        self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes ,kernel_size=size, stride=stride, bias=False),
                nn.BatchNorm1d(planes))
        self.dropout = nn.Dropout(.2)     
        self.sigmoid = nn.Sigmoid()


        self.globalAvgPool =nn.AdaptiveAvgPool1d(1)         
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)    
        

    def forward(self, x):  
        
        x=x.squeeze(2)        
        residual = self.downsample(x)
        
        
        out = self.conv1(x)        
        out = self.bn1(out)
        out = self.relu(out)
        


        out = self.dropout(out)
        
        out = self.bn2(out)        
        out = self.conv2(out)

 
        #Squeeze-and-Excitation (SE)      
        original_out = out
        out = self.globalAvgPool(out) 
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1),1)
        out = out * original_out
        
        
        #resnet         
        out += residual
        out = self.relu(out)

        return out



class BasicBlock2d(nn.Module):

    def __init__(self, inplanes, planes, stride, size,downsample):
        super(BasicBlock2d, self).__init__()  
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(1,size), stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(1,1), stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes ,kernel_size=(1,size), stride=stride, bias=False),
                nn.BatchNorm2d(planes))
        

        self.dropout = nn.Dropout(.2)
        self.sigmoid = nn.Sigmoid()
        
        self.globalAvgPool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes) 


    def forward(self, x):

        residual = self.downsample(x)
        
        
        out = self.conv1(x)        
        out = self.bn1(out)
        out = self.relu(out)


        out = self.dropout(out)
        
        out = self.bn2(out)        
        out = self.conv2(out)   
    
    #Squeeze-and-Excitation (SE)   
        original_out=out
        out = self.globalAvgPool(out) 
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
      
        out = out.view(out.size(0), out.size(1),1,1)
        out = out * original_out

    #resnet           
        out += residual
        out = self.relu(out)

        return out
    
    
class ECGNet(nn.Module):
    def __init__(self, BasicBlock1d, BasicBlock2d , num_classes):
        super(ECGNet, self).__init__()
        self.sizes=[5,7,9]
        self.external = 2        
        
        self.relu = nn.ReLU(inplace=True)  
        
        self.conv1 =  nn.Conv2d(12,32, kernel_size=(1,50), stride=(1,2),padding=(0,0),bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.AvgPool = nn.AdaptiveAvgPool1d(1)
     
      
        self.layers=nn.Sequential()
        self.layers.add_module('layer_1',self._make_layer( BasicBlock2d,inplanes=32,planes=32,blocks=1,stride=(1,2),size=15))
        self.layers.add_module('layer_2',self._make_layer( BasicBlock2d,inplanes=32,planes=32,blocks=1,stride=(1,2),size=15))
        self.layers.add_module('layer_3',self._make_layer( BasicBlock2d,inplanes=32,planes=32,blocks=1,stride=(1,2),size=15))
        
        self.layers1_list=nn.ModuleList()
        self.layers2_list=nn.ModuleList()   
        
        
        for size in self.sizes:
            
   
            self.layers1=nn.Sequential()
            
            self.layers1.add_module('layer{}_1_1'.format(size),self._make_layer( BasicBlock2d,inplanes=32, planes=32,blocks=32,
                                                                                  stride=(1,1),size=size))

            
            
            self.layers2=nn.Sequential()   
            self.layers2.add_module('layer{}_2_1'.format(size),self._make_layer(BasicBlock1d,inplanes=32, planes=256,blocks=1,
                                                                                  stride=2,size=size))        

            self.layers2.add_module('layer{}_2_2'.format(size),self._make_layer(BasicBlock1d,inplanes=256, planes=256,blocks=1,
                                                                                  stride=2,size=size)) 
            
            self.layers2.add_module('layer{}_2_3'.format(size),self._make_layer(BasicBlock1d,inplanes=256, planes=256,blocks=1,
                                                                                  stride=2,size=size)) 
            
            self.layers2.add_module('layer{}_2_4'.format(size),self._make_layer(BasicBlock1d,inplanes=256, planes=256,blocks=1,
                                                                                  stride=2,size=size)) 
            

            self.layers1_list.append(self.layers1)
            self.layers2_list.append(self.layers2)
                     
        
        self.fc = nn.Linear(256*len(self.sizes)+self.external, num_classes)


    def _make_layer(self, block,inplanes, planes, blocks, stride ,size,downsample = None):
        layers = []
        for i in range(blocks):
            layers.append(block(inplanes, planes, stride, size,downsample))
        return nn.Sequential(*layers) 


    def forward(self, x0, fr):
        
        x0=x0.unsqueeze(2)

        x0 = self.conv1(x0)        
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.layers(x0)
        
        xs=[]
        for i in range(len(self.sizes)):
            
            x=self.layers1_list[i](x0)
            x=torch.flatten(x,start_dim=2,end_dim=3)
            x=self.layers2_list[i](x0)
            x= self.AvgPool(x)
            xs.append(x)
            

        out = torch.cat(xs,dim=2)
        out = out.view(out.size(0), -1)
        out = torch.cat([out,fr], dim=1)
        out = self.fc(out)

        return out    
    
    
# Load challenge data.
def load_challenge_data(filename):
    x = loadmat(filename)
    data = np.asarray(x['val'], dtype=np.float64)

    new_file = filename.replace('.mat','.hea')
    input_header_file = os.path.join(new_file)

    with open(input_header_file,'r') as f:
        header_data=f.readlines()

    return data, header_data

# Find unique classes.
def get_classes(input_directory, filenames):
    classes = set()
    for filename in filenames:
        input_file=os.path.join(input_directory,filename)
        with open( input_file, 'r') as f:
            for l in f:
                if l.startswith('#Dx'):
                    tmp = l.split(': ')[1].split(',')
                    for c in tmp:
                        classes.add(c.strip())
    return sorted(classes)


def train(x_train,x_val,x_train_external,x_val_external,y_train,y_val, num_class):
    # model
    model = ECGNet( BasicBlock1d, BasicBlock2d ,num_classes= num_class)
    model = model.to(device)
    
    # optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
#   optimizer = optim. RMSProp(model.parameters(), lr=config.lr)
   
    
    wc = y_train.sum(axis=0)
    wc = 1. / (np.log(wc)+1)
    
    
    w = torch.tensor(wc, dtype=torch.float).to(device)
    criterion1 = utils.WeightedMultilabel(w)
    criterion2 = nn.BCEWithLogitsLoss()

    
    lr = config.lr
    start_epoch = 1
    stage = 1
    best_auc = -1
    
    # =========>开始训练<=========
    print("*" * 10, "step into stage %02d lr %.5f" % (stage, lr))
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss,train_auc = train_epoch(model, optimizer, criterion1,x_train,x_train_external,y_train,num_class)
        val_loss,val_auc = val_epoch(model, criterion2, x_val,x_val_external,y_val,num_class)
        print('#epoch:%02d stage:%d train_loss:%.4f train_auc:%.4f  val_loss:%.4f val_auc:%.4f  time:%s'
              % (epoch, stage, train_loss, train_auc,val_loss,val_auc, utils.print_time_cost(since)))

        if epoch in config.stage_epoch:
            stage += 1
            lr /= config.lr_decay
            print("*" * 10, "step into stage %02d lr %.5f" % (stage, lr))
            utils.adjust_learning_rate(optimizer, lr)
    return model

def train_epoch(model, optimizer, criterion,x_train,x_train_external,y_train,num_class):
    model.train()
    auc_meter,loss_meter, it_count = 0, 0,0
    batch_size=config.batch_size

    for i in range(0,len(x_train)-batch_size,batch_size):      
        inputs1 = torch.tensor(x_train[i:i+batch_size],dtype=torch.float,device=device)
        inputs2 = torch.tensor(x_train_external[i:i+batch_size],dtype=torch.float,device=device)
        target =  torch.tensor(y_train[i:i+batch_size],dtype=torch.float,device=device)         
        output = model.forward(inputs1,inputs2) 
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        auc_meter = auc_meter+ utils.calc_auc(target, torch.sigmoid(output)) 
        
    return loss_meter / it_count, auc_meter/it_count

def val_epoch(model, criterion, x_val,x_val_external,y_val,num_class):
    model.eval()
    auc_meter,loss_meter, it_count = 0, 0,0
    batch_size=config.batch_size
    
    with torch.no_grad():
        for i in range(0,len(x_val)-batch_size,batch_size):      
            inputs1 = torch.tensor(x_val[i:i+batch_size],dtype=torch.float,device=device)
            inputs2 = torch.tensor(x_val_external[i:i+batch_size],dtype=torch.float,device=device)
            target =  torch.tensor(y_val[i:i+batch_size],dtype=torch.float,device=device)
            output = model(inputs1,inputs2)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1 
            auc_meter =auc_meter + utils.calc_auc(target, torch.sigmoid(output))          
    return loss_meter / it_count, auc_meter/ it_count


def train_12ECG_classifier(input_directory, output_directory):
    
    input_files=[]
    header_files=[]
    
    train_directory=input_directory
    for f in os.listdir(train_directory):
        if os.path.isfile(os.path.join(train_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            g = f.replace('.mat','.hea')
            input_files.append(f)
            header_files.append(g)

    # the 27 scored classes
    classes_weight=['270492004','164889003','164890007','426627000','713427006','713426002','445118002','39732003',
                  '164909002','251146004','698252002','10370003','284470004','427172004','164947007','111975006',
                  '164917005','47665007','59118001','427393009','426177001','426783006','427084000','63593006',
                  '164934002','59931005','17338001']
    
    classes_name=sorted(classes_weight)
    
    num_files=len(input_files)
    num_class=len(classes_name)

    # initilize the array
    data_num = np.zeros((num_files,12,10*500))
    data_external=np.zeros((num_files,2))
    classes_num=np.zeros((num_files,num_class))


    for cnt,f in enumerate(input_files):
        classes=set()
        tmp_input_file = os.path.join(train_directory,f)
        data,header_data = load_challenge_data(tmp_input_file)

        for lines in header_data:
            if lines.startswith('#Dx'):
                tmp = lines.split(': ')[1].split(',')
                for c in tmp:
                    classes.add(c.strip())

            if lines.startswith('#Age'):
                age=lines.split(': ')[1].strip()    
                if age=='NaN':
                    age='60'  
            if lines.startswith('#Sex'):
                sex=lines.split(': ')[1].strip()

        for j in classes:                         
            if j in classes_name:
                class_index=classes_name.index(j)
                classes_num[cnt,class_index]=1
                         

        data_external[cnt,0]=float(age)/100
        data_external[cnt,1]=np.array(sex=='Male').astype(int)                                 

        if data.shape[1]>=5000:
            data_num[cnt,:,:] = data[:,:5000]/30000 
        else:
            length=data.shape[1]
            data_num[cnt,:,:length] = data/30000              
                
    #split the training set and testing set
    x_train,x_val,x_train_external,x_val_external,y_train,y_val = train_test_split(data_num,data_external,
                                               classes_num,test_size=0.2, random_state=2020)
    #build the pre_train model
    model= train(x_train,x_val,x_train_external,x_val_external,y_train,y_val, num_class)
    
    #save the model
    output_directory=os.path.join(output_directory, 'resnet_0628.pkl')
    torch.save(model, output_directory)    
    