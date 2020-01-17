import json
import torch
import collections
import os 
import numpy as np

class PosDataset(torch.utils.data.Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self,datapath,t): #初始化一些需要传入的参数
        super(PosDataset,self).__init__()
        data = []
        self.type = t
        self.posDic = collections.defaultdict(int)
        self.root = os.getcwd()
        with open(datapath, 'r') as f:
            data = json.load(f)
        # print(data)
        self.data = data
        self.labelList = [
  'YB'  , 'YO'  ,  'YG' ,  'YR'  ,
  'BW'  , 'BO'  ,  'BY' ,  'BR'  ,
  'WG'  , 'WO'  ,  'WB' ,  'WR'  ,
  'GY'  , 'GO'  ,  'GW' ,  'GR'  ,
  'RY'  , 'RG'  ,  'RW' ,  'RB', 
  'OG'  , 'OY'  ,  'OB' ,  'OW'
  ]
        
    def __getitem__(self, index): 
        data = self.data[index]
        # print(data)
        labelStr = data['label']
        label = self.labelList.index(labelStr)
        attitudeData = data['attitudeData']
        # print(attitudeData)
     

        if attitudeData==[]:
           
            return self.__getitem__(index+1)
        attData= [attitudeData['x'],
                   attitudeData['y'],
                   attitudeData['z'],
                   attitudeData['w']]
        # print(attData)
        attData = torch.from_numpy(np.array(attData))
        return attData, label
       

    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.data)

class PreDataset(torch.utils.data.Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self,datapath,t): #初始化一些需要传入的参数
        super(PreDataset,self).__init__()
        data = []
        self.type = t
        self.posDic = collections.defaultdict(int)
        self.root = os.getcwd()
        with open(datapath, 'r') as f:
            data = json.load(f)
        self.data = data
        
    def __getitem__(self, index): 
        data = self.data[index]
        label = data['label']
        attitudeData = data['data']

        if attitudeData==[]:
            return self.__getitem__(index+1)

        attData= [attitudeData['x'],
                   attitudeData['y'],
                   attitudeData['z'],
                   attitudeData['w']]
        # print(attData)
        attData = torch.from_numpy(np.array(attData))
        return attData, label
       

    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.data)
