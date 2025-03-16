from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import VotingClassifier
import os
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import normalize


main = tkinter.Tk()
main.title("AI-Driven Clinical Decision Support Framework for Early Diagnosis of Genetic Diseases in Children")
main.geometry("1300x1200")

global filename

global classifier
global left_X_train, left_X_test, left_y_train, left_y_test
global right_X_train, right_X_test, right_y_train, right_y_test

global left_X, left_Y

global left_pupil
global right_pupils
global count
global left
global right
global ids
global left_svm_acc
global right_svm_acc
global left_classifier
global right_classifier,ensemble_acc
global elm_acc
global lstm_acc,bilstm_acc

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir = ".")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'Pupillometric  dataset loaded\n')

def filtering():
    global left_pupil
    global right_pupil
    global count
    global left
    global right
    global ids
    left_pupil = []
    right_pupil = []
    count = 0
    left = 'Patient_ID,MAX,MIN,DELTA,CH,LATENCY,MCV,label\n'
    right = 'Patient_ID,MAX,MIN,DELTA,CH,LATENCY,MCV,label\n'
    ids = 1
    for root, dirs, directory in os.walk('dataset'):
        for i in range(len(directory)):
            filedata = open('dataset/'+directory[i], 'r')
            lines = filedata.readlines()
            left_pupil.clear()
            right_pupil.clear()
            count = 0
            for line in lines:
                line = line.strip()
                arr = line.split("\t")
                if len(arr) == 8:
                    if arr[7] == '.....':
                        left_pupil.append(float(arr[3].strip()))
                        right_pupil.append(float(arr[6].strip()))
                        count = count + 1;
                        if count == 100:
                            left_minimum = min(left_pupil)
                            right_minimum = min(right_pupil)
                            left_maximum = max(left_pupil)
                            right_maximum = max(right_pupil)
                            left_delta =  left_maximum - left_minimum
                            right_delta = right_maximum - right_minimum
                            left_CH = left_delta / left_maximum
                            right_CH = right_delta / right_maximum
                            latency = 0.5
                            left_MCV = left_delta/(left_minimum - latency)
                            right_MCV = right_delta/(right_minimum - latency)
                            count = 0
                            left_pupil.clear()
                            right_pupil.clear()
                            if left_minimum > 500 and left_maximum > 500:
                                left+=str(ids)+","+str(left_maximum)+","+str(left_minimum)+","+str(left_delta)+","+str(left_CH)+","+str(latency)+","+str(left_MCV)+",1\n"
                            else:
                                left+=str(ids)+","+str(left_maximum)+","+str(left_minimum)+","+str(left_delta)+","+str(left_CH)+","+str(latency)+","+str(left_MCV)+",0\n"
                            if right_minimum > 500 and right_maximum > 500:
                                right+=str(ids)+","+str(right_maximum)+","+str(right_minimum)+","+str(right_delta)+","+str(right_CH)+","+str(latency)+","+str(right_MCV)+",1\n"
                            else:
                                right+=str(ids)+","+str(right_maximum)+","+str(right_minimum)+","+str(right_delta)+","+str(right_CH)+","+str(latency)+","+str(right_MCV)+",0\n"
                            ids = ids + 1
            filedata.close()
    
    text.delete('1.0', END)
    text.insert(END,'Features filteration process completed\n')
    text.insert(END,'Total patients found in dataset : '+str(ids)+"\n")
    
def featuresExtraction():
    f = open("left.txt", "w")
    f.write(left)
    f.close()
    f = open("right.txt", "w")
    f.write(right)
    f.close()
    text.delete('1.0', END)
    text.insert(END,'Both eye pupils extracted features saved inside left.txt and right.txt files \n')
    text.insert(END,"Extracted features are \nPatient ID, MAX, MIN, Delta, CH, Latency, MDV, CV and MCV\n")

def featuresReduction():
    text.delete('1.0', END)
    global left_X, left_Y
    global left_X_train, left_X_test, left_y_train, left_y_test
    global right_X_train, right_X_test, right_y_train, right_y_test
    left_pupil =  pd.read_csv('left.txt')
    right_pupil =  pd.read_csv('right.txt')
    cols = left_pupil.shape[1]

    left_X = left_pupil.values[:, 1:(cols-1)] 
    left_Y = left_pupil.values[:, (cols-1)]

    right_X = right_pupil.values[:, 1:(cols-1)] 
    right_Y = right_pupil.values[:, (cols-1)]

    indices = np.arange(left_X.shape[0])
    np.random.shuffle(indices)
    left_X = left_X[indices]
    left_Y = left_Y[indices]

    indices = np.arange(right_X.shape[0])
    np.random.shuffle(indices)
    right_X = right_X[indices]
    right_Y = right_Y[indices]

    left_X = normalize(left_X)
    right_X = normalize(right_X)
    

    left_X_train, left_X_test, left_y_train, left_y_test = train_test_split(left_X, left_Y, test_size = 0.2,random_state=42)
    right_X_train, right_X_test, right_y_train, right_y_test = train_test_split(right_X, right_Y, test_size = 0.2,random_state=42)

    text.insert(END,"Left pupil features training size : "+str(len(left_X_train))+" & testing size : "+str(len(left_X_test))+"\n")
    text.insert(END,"Right pupil features training size : "+str(len(right_X_train))+" & testing size : "+str(len(right_X_test))+"\n")

    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Diameter')
    plt.plot(left_pupil['MAX'], 'ro-', color = 'indigo')
    plt.plot(right_pupil['MAX'], 'ro-', color = 'green')
    plt.legend(['Left Pupil', 'Right Pupil'], loc='upper left')
    plt.title('Pupil Diameter Graph')
    plt.show()   
    
    

def prediction(X_test, cls): 
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
      print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred     
    
def rightSVM():
    global right_classifier
    text.delete('1.0', END)
    global right_svm_acc
    temp = []
    for i in range(len(right_y_test)):
        temp.append(right_y_test[i])
    temp = np.asarray(temp)    
    right_classifier = svm.SVC(C=80,kernel='rbf', class_weight='balanced', probability=True)
    right_classifier.fit(right_X_train, right_y_train)
    text.insert(END,"Right pupil SVM Prediction Results\n") 
    prediction_data = prediction(right_X_test, right_classifier) 
    right_svm_acc = accuracy_score(temp,prediction_data)*100
    text.insert(END,"Right pupil SVM Accuracy : "+str(right_svm_acc)+"\n")

    cm = confusion_matrix(temp, prediction_data)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    text.insert(END,'Right pupil SVM Algorithm Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    text.insert(END,'Right pupil SVM Algorithm Specificity : '+str(specificity)+"\n\n")

def leftSVM():
    global left_classifier
    global left_svm_acc
    temp = []
    for i in range(len(left_y_test)):
        temp.append(left_y_test[i])
    temp = np.asarray(temp) 
    left_classifier = svm.SVC(C=80,kernel='rbf', class_weight='balanced', probability=True)
    left_classifier.fit(left_X_train, left_y_train)
    text.insert(END,"Left pupil SVM Prediction Results\n") 
    prediction_data = prediction(left_X_test, left_classifier) 
    left_svm_acc = accuracy_score(temp,prediction_data)*100
    text.insert(END,"Left pupil SVM Accuracy : "+str(left_svm_acc)+"\n")

    cm = confusion_matrix(temp, prediction_data)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    text.insert(END,'Left pupil SVM Algorithm Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    text.insert(END,'Left pupil SVM Algorithm Specificity : '+str(specificity)+"\n\n")

def ensemble():
    global classifier
    global ensemble_acc
    trainX = np.concatenate((right_X_train, left_X_train))
    trainY = np.concatenate((right_y_train, left_y_train))

    testX = np.concatenate((right_X_test, left_X_test))
    testY = np.concatenate((right_y_test, left_y_test))

    indices = np.arange(trainX.shape[0])
    np.random.shuffle(indices)
    trainX = trainX[indices]
    trainY = trainY[indices]

    left_classifier = svm.SVC(C=200,kernel='rbf', class_weight='balanced', probability=True)
    right_classifier = svm.SVC(C=200,kernel='rbf', class_weight='balanced', probability=True)

    temp = []
    for i in range(len(testY)):
        temp.append(testY[i])
    temp = np.asarray(temp) 

    classifier = VotingClassifier(estimators=[
         ('SVMLeft', left_classifier), ('SVMRight', right_classifier)], voting='hard')
    classifier.fit(trainX, trainY)
    text.insert(END,"Optimized Ensemble Prediction Results\n") 
    prediction_data = prediction(testX, classifier) 
    ensemble_acc =  (accuracy_score(temp,prediction_data)*100)
    text.insert(END,"Ensemble OR Accuracy : "+str(ensemble_acc)+"\n")

    cm = confusion_matrix(temp, prediction_data)
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    text.insert(END,'Right pupil Ensemble OR SVM Algorithm Sensitivity : '+str(sensitivity)+"\n")
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    text.insert(END,'Right pupil Ensemble OR SVM Algorithm Specificity : '+str(specificity)+"\n\n")


def predict():
    global classifier
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "testData")
    test = pd.read_csv(filename)
    test = test.values[:, 0:7]
    total = len(test)
    text.insert(END,filename+" test file loaded\n");
    test = normalize(test)
    y_pred = classifier.predict(test)
    print(y_pred)
    for i in range(len(test)):
        print(str(y_pred[i]))
        if str(y_pred[i]) == '0.0':
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'No disease detected')+"\n\n")
        else:
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Disease detected')+"\n\n")


def graph():
    height = [right_svm_acc,left_svm_acc,ensemble_acc]
    bars = ('Right Pupil SVM Acc','Left Pupil SVM Acc','Ensemble OR (L & R Pupil) Acc')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("All Algorithms Accuracy Comparison Graph")
    plt.show()
    
font = ('times', 16, 'bold')
title = Label(main, text='AI-Driven Clinical Decision Support Framework for Early Diagnosis of Genetic Diseases in Children')
title.config(bg='#E8F5E9', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Dataset", command=upload)
upload.place(x=10,y=100)
upload.config(font=font1)  
upload.config(bg='#3F51B5', fg='white')

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=10,y=150)

filterButton = Button(main, text="Filtering", command=filtering)
filterButton.place(x=10,y=200)
filterButton.config(font=font1) 
filterButton.config(bg='#3F51B5', fg='white')

extractButton = Button(main, text="Features Extraction", command=featuresExtraction)
extractButton.place(x=10,y=250)
extractButton.config(font=font1) 
extractButton.config(bg='#3F51B5', fg='white')

featuresButton = Button(main, text="Features Reduction", command=featuresReduction)
featuresButton.place(x=10,y=300)
featuresButton.config(font=font1)
featuresButton.config(bg='#3F51B5', fg='white')

rightsvmButton = Button(main, text="SVM on Right Eye Features", command=rightSVM)
rightsvmButton.place(x=10,y=350)
rightsvmButton.config(font=font1)
rightsvmButton.config(bg='#3F51B5', fg='white')

leftsvmButton = Button(main, text="SVM on Left Eye Features", command=leftSVM)
leftsvmButton.place(x=10,y=400)
leftsvmButton.config(font=font1)
leftsvmButton.config(bg='#3F51B5', fg='white')


ensembleButton = Button(main, text="Ensemble Model (Left & Right SVM)", command=ensemble)
ensembleButton.place(x=10,y=450)
ensembleButton.config(font=font1)
ensembleButton.config(bg='#3F51B5', fg='white')


graphButton = Button(main, text="Performance Evaluation", command=graph)
graphButton.place(x=10,y=500)
graphButton.config(font=font1)
graphButton.config(bg='#3F51B5', fg='white')


predictButton = Button(main, text="Predict Disease", command=predict)
predictButton.place(x=10,y=550)
predictButton.config(font=font1)
predictButton.config(bg='#3F51B5', fg='white') 

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=120)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=400,y=100)
text.config(font=font1)


main.config(bg='Light Gray')
main.mainloop()
