
from sklearn import svm, datasets 
from sklearn.multiclass import OneVsRestClassifier 
from torch.utils.data.dataloader import DataLoader
from dataset import getData
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
# %matplotlib inline


##################################
# Load Data
##################################
trainPath = "../data/train.json"
testPath = "../data/test.json"

X ,y=getData(trainPath,t="classifier")
X_t ,y_t=getData(testPath,t="classifier")

train_01Path = "../data/train_01.json"
test_01Path = "../data/test_01.json"
X_01 ,y_01=getData(train_01Path,t="pre")
X_t_01 ,y_t_01=getData(test_01Path,t="pre")

#################################
# Get best args
#################################
# from sklearn.model_selection import GridSearchCV


# #   print(x,y)

# grid = GridSearchCV(svm.SVC(), param_grid={"C":[0.7], "gamma": [0.5,0.75]}, cv=4)
# grid.fit(X_01, y_01)
# print(grid.score(X_t_01 ,y_t_01))
# print("The best parameters are %s with a score of %0.2f"
#       % (grid.best_params_, grid.best_score_))

#################################
# Save Model
#################################
import joblib
model = OneVsRestClassifier(svm.SVC(C=2.5, kernel='rbf', gamma=9.5))
model.fit(X ,y)
print(model.score(X_t, y_t))
joblib.dump(model, '../model/svm.pkl') 

premodel = OneVsRestClassifier(svm.SVC(C=0.75, kernel='rbf', gamma=0.75))
premodel.fit(X_01 ,y_01)
print(premodel.score(X_t_01, y_t_01))
joblib.dump(premodel, '../model/pre_svm.pkl') 