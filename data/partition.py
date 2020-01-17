import os
import collections
import json

root  = "d:/mcube/python/LR/data"
dataset = "position.json"
train = "train.json"
test = "test.json"




trainData = []
testData = []
with open(os.path.join(root,dataset),'r') as f, open(os.path.join(root,train),'w') as trainFile, open(os.path.join(root,test),'w') as testFile:
  js = json.load(f)
  for data in js:
    label = data['label']
    attDatas = data['attitudeData']
    for i,att in enumerate(attDatas):
      data = {'label':label,'attitudeData':att}
      
      if i%5 == 0:
          testData.append(data)
      else:
          trainData.append(data)
  json.dump(trainData,trainFile)
  json.dump(testData,testFile)