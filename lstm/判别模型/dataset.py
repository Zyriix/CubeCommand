import json
import torch
import collections
import os 
import numpy as np
import loadTagSeq
class CubeDataset(torch.utils.data.Dataset): #创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self,datatxt,t): #初始化一些需要传入的参数
        super(CubeDataset,self).__init__()
        data = []
        self.type = t
        self.actDic = collections.defaultdict(int)
        self.attDic = collections.defaultdict(int)
        self.root = os.getcwd()
        self.steps = []
        fh = open(datatxt, 'r') #按照传入的路径和txt文本参数，打开这个文本，并读取内容
        attLabel = ['x','x\'', 'y' ,'y\'', 'z', 'z\'']
        label =   [
    'u', 'u\'', 'l', 'l\'',
    'f', 'f\'', 'r', 'r\'',
    'b', 'b\'', 'd', 'd\'',
    'M', 'M\'', 'S', 'S\'',
    'E', 'E\'', 'x', 'x\'',
    'y', 'y\'', 'z', 'z\'', 'noise']
        
        
        for index,act in enumerate(label):
                     self.actDic[act] = index
        for index,att in enumerate(attLabel):
                     self.attDic[att] = index
        self.actDic['noise'] = 6
        for i,line in enumerate(fh):
                line = line.rstrip() 
                words = line.split()   
                data.append((words[0])) 
           
        # print(data)

        self.data = data

        # self.transform = transform
        # self.target_transform = target_transform
    
 
    def __getitem__(self, index): 
       
        filePath = self.data[index]
        # print(filePath)
        path = os.getcwd()
        if self.type =='train':
            dic = "/data/train/"
        elif self.type == 'test':
            dic = "/data/test/"
        with open(self.root+dic+filePath,"r") as f:
            data = json.load(f)
        
        attitudeData = data['attitudeData']
        # print(data['attitudeLabel'])
        
        
        # print(filePath)
        # print(data['steps'])
        steps= data['steps']
        # if steps not in self.steps:
        #     self.steps.append(steps)
        # steps = self.steps.index(steps)
        # if len(steps)==2:
        #     steps = self.stepDict[sum(steps)]
        attLabel = self.attDic[data['attitudeLabel']]
        label = self.actDic[data['label']]
        # print(attLabel)
        attData=[]
        for d in attitudeData:
            attData.append([d['x']*100,d['y']*100,d['z']*100,d['w']*100])

        if attData==[]:
            # print("无数据",filePath)
            return self.__getitem__(index+1)
   
        attData = torch.from_numpy(np.array(attData))
        # l = attData.new_zeros(18)
        # l[label]=1
        # l=l.view(18)
        # print("l",l.size())
        # print("label",l)
        # k = attData.new_zeros()
        # attData=attData.view(len(attitudeData),6)
        # rotData=rotData.view(len(rotateData),4)
        # print(attData)
        
        # print(attData.size(),rotData.size())
        # label = data['label']
        # print(label)
        if self.type == 'train':
          return attData, attLabel
        elif self.type == 'test':
          return attData, attLabel, steps, label

    def __len__(self): #这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.data)
