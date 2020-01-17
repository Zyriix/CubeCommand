import os
import json
root = "d:/mcube/python/lstm/data"
ori = os.path.join(root,"ori")
train = os.path.join(root,"train")

jsonoriPaths = os.listdir(ori)

for p in jsonoriPaths:
    oriFp = os.path.join(ori,p)
    f = open(oriFp,"r")
    # print(oriFp)
    # print(os.path.join(train,p))
    newf = open(os.path.join(train,p),'w')
    js = json.load(f)
    newAtt =  [{'x':data['x'],'y':data['y'], 'z':data['z'], 'w':data['w']}  for data in js ['attitudeData'] ] 
    newRot =  [{'face':data['face'],'circle':data['circle']}  for data in js['rotateData']] 
    newLabel = js['label']
    newJson = {'attitudeData':newAtt,'rotateData':newRot, 'label':newLabel}

    f.close()
    newf.write(json.dumps(newJson))
