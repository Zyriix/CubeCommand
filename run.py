import os
import json
root = "d:/mcube/python/lstm/data"
train = os.path.join(root,"ori")
path = os.listdir("d:/mcube/python/lstm/data")
jsonPath = []
for p in path:
    if p.split(".")[-1]!='json':
        continue
    jsonPath.append(p)

trainJsonPath = []
trainPath = os.listdir(train)
for p in trainPath:
    if p.split(".")[-1]!='json':
      continue
    trainJsonPath.append(p)

if len(trainJsonPath)==0:
    index = 0
else:
    index = int(trainJsonPath[-1].split('.')[0])
print(index)



   
print(jsonPath)
for path in jsonPath:
      

  with open(os.path.join(root,path),"r") as j:
    dataList = json.load(j)

      
    for data in dataList:
        index += 1
        fileName = str(index).zfill(5)+".json"
        with open(train+"/"+fileName,"w") as rs:
            rs.write(json.dumps(data))
              
