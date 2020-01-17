import joblib
import json
import random

pre_model = './model/pre_svm.pkl'
model = './model/svm.pkl'
table = './model/t.json'
test = '../data/test.json'

def output(q1,q2):
  f = open(table,'r')
  table = json.load(f)
  pre_clf = joblib.load(pre_model)
  clf = joblib.load(model)
  pre_data = [
    q2['x']-q1['x'],
    q2['y']-q1['y'],
    q2['z']-q1['z'],
    q2['w']-q1['w'],]

  pre = pre_clf.predict(pre_data)
  if pre == 0:
    return
  pos1_data = [
      q1['x'],
      q1['y'],
      q1['z'],
      q1['w']]

  pos2_data = [
      q2['x'],
      q2['y'],
      q2['z'],
      q2['w']]
  
  pos1 = clf.predict(pos1_data)
  pos2 = clf.predict(pos2_data)

  key = pos1+pos2
  print("预测姿态：",key)

  output = table[key]
#   print(output)
  return output



if __name__ == "__main__":
   testData = json.load(open(test,'r'))
   len = len(testData)
   for i in range(500):
       data1 = testData[random.randint(0,len-1)]
       data2 = testData[random.randint(0,len-1)]
       label = data1['label']+data2['label']
       
       rs = table[label]
       print("姿态{}\n实际动作：{}".format(label,rs))

       pre = output(data1['data'],data2['data'])
       print("输出动作：{}".format(pre))
   
