import json
import os 
import numpy as np

def getData(datapath,t):
        f = open(datapath, 'r')
        datas = json.load(f)
        rs_x = []
        rs_y = []
        labelList = [
  'YB'  , 'YO'  ,  'YG' ,  'YR'  ,
  'BW'  , 'BO'  ,  'BY' ,  'BR'  ,
  'WG'  , 'WO'  ,  'WB' ,  'WR'  ,
  'GY'  , 'GO'  ,  'GW' ,  'GR'  ,
  'RY'  , 'RG'  ,  'RW' ,  'RB', 
  'OG'  , 'OY'  ,  'OB' ,  'OW'
  ]
        
        
        for  data in datas:
          
          labelStr = data['label']
          if t == "pre":
              label = labelStr
              attitudeData = data['data']
          else:
            label = labelList.index(labelStr)
            attitudeData = data['attitudeData']
          
        
          if attitudeData==[]:
             continue

          attData = [attitudeData['x'],
                   attitudeData['y'],
                   attitudeData['z'],
                   attitudeData['w']]
          attData = np.array(attData)
          rs_x.append(attData )
          rs_y.append( label )
        return rs_x, rs_y
       
