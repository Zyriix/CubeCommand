import joblib
import json
import random

import numpy
pre_model = '../model/pre_svm.pkl'
model = '../model/svm.pkl'
table = '../model/t.json'
test = '../data/test.json'

def output(q1,q2):
  labelList = [
  'YB'  , 'YO'  ,  'YG' ,  'YR'  ,
  'BW'  , 'BO'  ,  'BY' ,  'BR'  ,
  'WG'  , 'WO'  ,  'WB' ,  'WR'  ,
  'GY'  , 'GO'  ,  'GW' ,  'GR'  ,
  'RY'  , 'RG'  ,  'RW' ,  'RB', 
  'OG'  , 'OY'  ,  'OB' ,  'OW'
  ]
  global table
  f = open(table,'r')
  
  t = json.load(f)
  pre_clf = joblib.load(pre_model)
  clf = joblib.load(model)
  pre_data = numpy.array([
    q2['x']-q1['x'],
    q2['y']-q1['y'],
    q2['z']-q1['z'],
    q2['w']-q1['w']]).reshape(1, -1)
  
  
  pre = pre_clf.predict(pre_data)
  # print(pre)
  if pre == 0:
    return "无"


  pos1_data =  numpy.array([
      q1['x'],
      q1['y'],
      q1['z'],
      q1['w']]).reshape(1, -1)

  pos2_data =  numpy.array([
      q2['x'],
      q2['y'],
      q2['z'],
      q2['w']]).reshape(1, -1)

  pos1 = clf.predict(pos1_data)
  pos2 = clf.predict(pos2_data)
  
  pos1Str = labelList[int(pos1)]
  pos2Str = labelList[int(pos2)]
  key = pos1Str+pos2Str
  # print("预测姿态：",key)

  output = t[key]
#   print(output)
  return output



if __name__ == "__main__":
   f = open(table,'r')
   t = json.load(f)
   testData = json.load(open(test,'r'))
   len = len(testData)
   wc =0
   cc=0
   total=0
   for i in range(2000):
       data1 = testData[random.randint(0,len-1)]
       data2 = testData[random.randint(0,len-1)]
       label = data1['label']+data2['label']
       if data1['label'] == data2['label']:
      #  print(label,table)
         rs = "无"
        
        
       else:
          rs = t[label]
      #  print("姿态{}\n实际动作：{}".format(label,rs))

       pre = output(data1['attitudeData'],data2['attitudeData'])
      #  print("输出动作：{}\n".format(pre))

       if(pre == rs):
         cc+=1
         total+=1
       else:
         wc+=1
         total+=1
   print(cc/total)
   
