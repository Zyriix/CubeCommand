import os
import collections
import json

root  = "d:/mcube/python/LR/data"
dataset = "0-1.json"
train = "train_01.json"
test = "test_01.json"




trainData = []
testData = []
with open(os.path.join(root,dataset),'r') as f, open(os.path.join(root,train),'w') as trainFile, open(os.path.join(root,test),'w') as testFile:
    js = json.load(f)
    for i, data in enumerate(js):
        label = data['label']
        if data['attitudeData'] == []:continue
        att0 = data['attitudeData'][0]
        att1 = data['attitudeData'][-1]
        newAtt = {
            'x':att1['x'] - att0['x'],
            'y':att1['y'] - att0['y'],
            'z':att1['z'] - att0['z'],
            'w':att1['w'] - att0['w']
        }
        data = {
            'label':label,
            'data':newAtt
        }
        if i%5==0:
            testData.append(data)
        else:
            trainData.append(data)
    json.dump(trainData,trainFile)
    json.dump(testData,testFile)
