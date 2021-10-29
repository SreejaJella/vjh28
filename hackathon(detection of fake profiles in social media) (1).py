#importing libraries
import sys
import csv
import os
import datetime
import math
import numpy as np
import pandas as pd
import matplotlib pyplot as p
from datetime import datetime
import sexmachine.detector as gender
from sklearn.preprocessing import Imputer
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestclassifier
from sklearn import preprocessing
from sklearn.cross_validation import StartifiedKFold, train_test_split
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.metrics import accuracy_score
from sklearn.learning import learning_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import Classification_report
from pybrain.structure import SigmoidLayer
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percent Error
from pybrain.structure.modules import softmaxLayer
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
get_ipython().magic(u'matplotlib inline')



#calling the function to read the dataset

def read_datasets():
    #reads user profile from CSV files
    actual_users= pd.read_csv("C:/Users/thanusri/Downloads/actual_users.csv")
    fake_users= pd.read_csv("C:/Users/thanusri/Downloads/fake_users.csv")
    #printing actual users
    x= pd.concate([actual_users,fake_users])
    y= len(fake_users)*[0]+len(actual_users)*[1]
    return x,y

#fuction for gender using name of the user
def gender_classification(name):
    gender_predictor = gender.Detector(unknown_value=u"unknown", case_sensitive=False)
    first_name= name.str.split(' ').str.get(0)
    gender = first_name.apply(sex_predictor.get_gender)
    gender_dict= {'female': -1, 'unknown': 0 , 'male': 1}
    gender_code = gender.map(gender_dict).astype(int)
    return gender_code
def extract_features(x):
    language_list = list(enumerate(np.unique(x['language'])))
    language_dict = (name: i for i , name in language_list)
    x.loc[:,'language']
    x.loc[:,'gender_code']= gender_classification(x['name'])
    feature_columns_to_use = ['statuses_count', 'followers_count' , 'friends_count' , 'fav_count','gender code' ]
    x=x.loc[:, feature_columns_to_use]
    return x
def plot_confusion_matrix(cm,title='confusion matrix",camp=pil.cm.Blues)_:
    target_names=['Fake','genuine']
    pli.imshow(cm,interpolation='nearest,cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks=np.arange(len(target_names))
    plt.xticks(tick_marks,target_names,rotation=45)
    plt.yticks(tick_marks,target_names)
    plt.tight_layout()
    plt.ylable('true label')
    plt.xlabel('predicted label')


# function for plotting ROC curve

def plot_roc_curve(y_test,y_pred):
    false_positive_rate,true_positive_rate,thresholds=roc_curve(y_test,y_pred)
    print ("False positive rate: ",false_positive _rate)
    print ("True positive rate: "true_positive_rate)
    roc_auc = auc(false_positive_rate,true_positive_rate)
    plt.title('Receiver operating characteristics')
    plt.plot(false_positive_rate,true_positive_rate,'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True positive Rate')
    plt.xlabel('False positive rate')
    plt.show()
#function for training data using Neural Network

def train(x,y):
    """Trains and predicts dataset with a Neural Network classifier"""
    ds=ClassificationDataSet(len(X.columns),1,nb_classes=2)
    for k in range(len(X)):
        ds.addSample(X.iloc[k],np.array(y[k]))

        tstdata,trndata=ds.splitWithPropotion(0.20)
        trndata._convertToOneOf Many()
        tstdata._convertToOneOfMany()
        input_size=len(X.columns)
        target_size=1
        hidden_size=5
        fnn=None
        if  os.path.isfile('fnn.xml'):
            fnn=NetworkReader.readFrom('fnn,xml')
        else:
            fnn=buildNetwork(trndata.indim,hidden_size,trndata.outdim,outclass=Softmaxlayer)
        trainer=BackpropTrainer(fnn,dataset=trndata,momentum=0.05,learningrate=0.1,verbose=False,weightdecay=0.01)


        trainer.trainUntilConvergence(verbose=False,validationProportion=0.15,maxEpochs=100,continueEpochs=10)
        NetworkWriter.writeToFile(fn,'oliv.xml')
        predictions=trainer.testOnClassData (dataset=tstdata)
        return tstdata['class'],predictions
print "reading datasets.....\n"
x=extract_features(x)
print x.columns
print x.describe()

print "training datasets.....\n"
y_test_pred=train(x,y)
print 'classification accuracy on test dataset:' ,accuracy_score)y_test,y_pred)


print('Classification Accuracy on test dataset:',accuracy_score(y_test, y_pred)
print("Percent error on test dataset:" , percentError(y_pred, y_test))
cm = confusion _matrix(y_test, y_pred)
print("Confusion Matrix , without normalization")
print(cm)
plot_confusion_matrix(cm)


print(classification_report(y_test, y_pred, target_names=['Fake', 'Genuine'])
s= roc_auc_score(y_test, y_pred)
print("roc_auc_score:" , s)




      
      
        
        
    
    
               

                        

    










