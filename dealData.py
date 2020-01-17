import os
import json
import math
import time
import collections
import random
def getCorrects(rot):
    circle = 0
    finalFace = 0
    fs = [0]*6
    rs = ""
    for data in rot:
        fs[data['face']] += data['circle']
    for i, f in enumerate(fs):
        if abs(f)>=360-72:
            rs+=str([i, -i][f<0])
    return rs
def  rnd( seed ):
    seed = ( seed * 9301 + 49297 ) % 233280
    return seed / ( 233280.0 )

def   rand(number):
    seed = random.random()
    return seed

oriPath = "d:/mcube/python/lstm/data/ori"
try:
  lastIndex = max(int(open("d:/mcube/python/lstm/data/train.txt","r").readlines()[-1].split('.')[0]),
  int(open("d:/mcube/python/lstm/data/test.txt","r").readlines()[-1].split('.')[0]))
except:
   lastIndex = 1

paths = os.listdir(oriPath)
l = len(paths)
test = []


trainPath = "d:/mcube/python/lstm/data/train"
testPath = "d:/mcube/python/lstm/data/test"
attLabelList = [
    'y', 'y\'','x\'', 'x', 
    'z', 'z\'','x',   'x\'',
    'z\'','z', 'y\'','y',
    'x\'','x', 'z', 'z\'',
    'y', 'y\'','x', 'x\'',
    'y', 'y\'','z', 'z\'', 'noise'
  ]
attLabelIndex = ['x', 'x\'',
    'y', 'y\'','z', 'z\'', 'noise']
labelList= str("u u' l l' f f' r r' b b' d d' M M' S S' E E' x x' y y' z z' noise").split(' ')
# print(labelList)
# print(labelList.index('u\''))
labelMaps = collections.defaultdict(list)
for i, p in enumerate(paths[lastIndex:]):
    with open(os.path.join(oriPath,p),'r') as f:
       js = json.load(f)
       label = js['label']
       index = labelList.index(label)
       attLabel = attLabelList[index]
       att  = [{'x':data['x'],'y':data['y'], 'z':data['z'], 'w':data['w']}  for data in js ['attitudeData'] ] 
       rot  =   [{'face':data['face'],'circle':data['circle']}  for data in js['rotateData']] 
       rotLabel = getCorrects(rot)
       jsonObj = {'attitudeData':att,'steps':rotLabel,'label':label,'attitudeLabel':attLabel}
       if (rotLabel,  str(attLabelIndex.index(attLabel))) not in labelMaps[labelList.index(label)]:
           if len(rotLabel)==0 and label not in attLabelIndex:
               continue
        #    if len(rotLabel)==2:
            #    if rotLabel not in stepDict:
            #        stepDict[rotLabel] = index
            #        index +=1
            #        print(rotLabel,index)
            #    rotLabel = stepDict[rotLabel]
           labelMaps[labelList.index(label)].append((rotLabel, str(attLabelIndex.index(attLabel))))
    #    print(labelMaps)
       if i%10 ==0:
           jsonpath = os.path.join(testPath,p)
           txtFile = open("d:/mcube/python/lstm/data/test.txt","a")
       else:
           jsonpath = os.path.join(trainPath,p)
           txtFile = open("d:/mcube/python/lstm/data/train.txt","a")
       with open(jsonpath, 'w') as out_file :
         json.dump(jsonObj, out_file)
       with open("d:/mcube/python/lstm/data/maps.txt",'w') as maps:
           json.dump(labelMaps,maps)
       txtFile.write("{}".format("\n"+p))
       