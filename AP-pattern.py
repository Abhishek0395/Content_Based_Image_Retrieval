
# coding: utf-8

# In[1]:


import numpy as np 
import cv2
from skimage.feature import local_binary_pattern


# In[2]:


import os
directory_list = list()
dir_path = "C:/Users/hp/Desktop/Thesis project/full_dataset"
for root, dirs, files in os.walk(dir_path, topdown=False):
    for name in dirs:
        directory_list.append((name))
for directory in directory_list:
    print(directory)


# In[3]:


import glob

labels = 'labels.csv'
labelfile = open(labels,'w')
for i in range(len(directory_list)):
    readpath = (dir_path+'/' + directory_list[i]+'/*jpg')
    print("NN")
    print(readpath)
    images = glob.glob(readpath)
    for image in images:
        labelfile.write(image+','+str(i)+'\n')
labelfile.close()


# In[4]:


import random
labels = 'labels.csv'
shuffled_labels = 'Shuffled_labels.csv'
labelfile = open(labels, "r")
lines = labelfile.readlines()
labelfile.close()
random.shuffle(lines)
shufflefile = open(shuffled_labels, "w")
shufflefile.writelines(lines)
shufflefile.close()


# In[5]:


import cv2
import numpy as np
items = []
true_items = []
def add_dict(dic,val1,val2,val3):
    val=0
    for i in range(1,4):
        if(i==1):
            val=val1
        elif(i==2):
            val=val2
        elif(i==3):
            val=val3
        if (1000*i+val) not in dic.keys():
            dic[25000*i] = dic[25000*i] +1
        else:
            dic[i*1000+val] = dic[i*1000+val] + 1
    return 1
def add_patrn(dic,val1,val2,val3,val4):
    val=0
    for i in range(4,8):
        if(i==4):
            val=val1
        elif(i==5):
            val=val2
        elif(i==6):
            val=val3
        elif(i==7):
            val=val4
        if(1000*i+val) not in dic.keys():
            dic[25000*i]=dic[25000*i]+1
        else:
            dic[i*1000+val]=dic[i*1000+val]+1
    return 1
def lap(x,y,img):
    t=7
    sums=0
    var=0
    out=[]
    fx=[0,-1,-1,-1,0,1,1,1]
    fy=[1,1,0,-1,-1,-1,0,1]
    for i in range(0,8):
        sums+=int(img[x+fx[i]][y+fy[i]])
    mean=sums/8.0
    for i in range(0,8):
        var+=int((mean-int(img[x+fx[i]][y+fy[i]]))**2)
    var=var/8.0
    var=var**(0.5)                                               
    for i in range(0, 8):
        if(int(img[x][y])+t>=int(img[x+fx[i]][y+fy[i]])):
            if(int(img[x][y])-t<=int(img[x+fx[i]][y+fy[i]])):
                out.append(1)
            else:
                out.append(3)
        else:
             out.append(2)
    return out,var,mean
def pattern(var,mean,x,y,img):
    out=[]
    if(var>31):
        p=54
        q=72
        cnt=0
        flag=0
        flag1=1
        cur=0
        prev=0
        first=0
        fx=[0,-1,-1,-1,0,1,1,1,0]
        fy=[1,1,0,-1,-1,-1,0,1,1]
        for i in range(0, 8):
            if(flag==0):
                cur=int(img[x+fx[i]][y+fy[i]])
            if(cur+p<=int(img[x+fx[i+1]][y+fy[i+1]])):
                if(cur+q<=int(img[x+fx[i+1]][y+fy[i+1]])):
                    flag=1
                    out.append(1)
                else:
                    flag=1
                    out.append(3)
            elif(cur-p>=int(img[x+fx[i+1]][y+fy[i+1]])):
                if(cur-q>=int(img[x+fx[i+1]][y+fy[i+1]])):
                    flag=1
                    out.append(2)
                else:
                    flag=1
                    out.append(4)
            else:
                if(flag==0 and flag1):
                    out.append(0)
                elif(flag==1):
                    flag1=0
                    flag=0
                    i=i-1
        return out,1
    else:
        return out,0
def local_ap_pattern(img,dic,q):
    try:
        for x in range(2, len(img)-2):
            for y in range(2, len(img[0])-2):
                value,ab,cd = lap(x,y,img)
                patrn,flag= pattern(ab,cd,x,y,img)
                val1=0
                val2=0
                val3=0
                for i in range(0,len(value)):
                    if(value[i]==1):
                        val1+=2**(7-i)
                    elif(value[i]==2):
                        val2+=2**(7-i)
                    elif(value[i]==3):
                        val3+=2**(7-i)
                add_dict(dic,val1,val2,val3)
                val1=0
                val2=0
                val3=0
                val4=0
                if(flag==1):
                    for i in range(0,len(patrn)):
                        if(value[i]==1):
                            val1+=2**(7-i)
                        elif(value[i]==2):
                            val2+=2**(7-i)
                        elif(value[i]==3):
                            val3+=2**(7-i)
                        elif(value[i]==4):
                            val4+=2**(7-i)
                add_patrn(dic,val1,val2,val3,val4)
    except TypeError as e:
        print ("type error "+str(j))
        return dic,0
    return dic,1
             


# In[6]:


def uniform(pat):
    a=0
    b=0
    cnt=0
    for i in range(0,8):
        if( i==0 ):
            a= int(pat/2**(7-i))
        else:
            b= int(pat/2**(7-i))
            if(b!=a):
                cnt=cnt+1
                a=b
        pat=pat%2**(7-i)
    if(cnt<=2):
        return 1 #uniform hbe
    else:
        return 0
#items.append(0)
for i in range(0, 256):
    if uniform(i):
        items.append(1*1000+i)
        true_items.append(i)
        items.append(2*1000+i)
        items.append(3*1000+i)
        items.append(4*1000+i)
        items.append(5*1000+i)
        items.append(6*1000+i)
        items.append(7*1000+i)
items.append(25000)
items.append(50000)
items.append(25000*3)
items.append(25000*4)
items.append(25000*5)
items.append(25000*6)
items.append(25000*7)
true_items.append(25000)


# In[7]:


import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#from keras.utils import np_utils
from scipy.stats import itemfreq
import time
from functools import wraps

df = pd.read_csv(shuffled_labels,header=None)
rows = df.iterrows()

X_addrs = []
X_hist = []
Y_hist = []
#row = rows[0]
j = 0
a=-1
# uncomment below this for local tetra pattern
dic = {}
def create_dic():
    #for i in range(len(items)):
    for i in items:
        dic[i] = 0
    return 1
start_time = time.time()      
for row in rows:
    #print(row[1][1])
    create_dic()
    #print(row[1][0])
    img = cv2.imread(row[1][0])
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dic,a=local_ap_pattern(img,dic,j)
        new_x = []
        if(a==1):
            for i in dic.keys():
                new_x.append(dic[i])
            new_x = np.array(new_x)
            hist = new_x/np.sum(new_x)
            #print(hist)
            X_addrs.append(row[1][0])
            X_hist.append(hist)
            Y_hist.append(row[1][1])
            if(j%500==0):
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print(str(j)+" done in " + str(elapsed_time))
    except Exception as e:
        print ("assertion error "+str(j))
    j = j + 1
    


# In[8]:


print(X_addrs[0])
print(X_hist[0])
print(Y_hist[0])
#print(X_hist)
X_hist = np.array(X_hist)
Y_hist = np.array(Y_hist)
print(len(X_addrs))
print(len(X_hist[50]))
print(X_hist.shape)
#print((Y_hist))

import pandas as pd 
df = pd.DataFrame(X_hist)
df.to_csv("tet_pattern_normalized_data_X_hist_AP.csv",header=None)
df = pd.DataFrame(Y_hist)
df.to_csv("tet_pattern_normalized_label_Y_hist_AP.csv",header=None)

df = pd.DataFrame(X_addrs)
df.to_csv("tet_pattern_normalized_address_AP.csv",header=None)


# In[9]:


import pandas as pd
import numpy as np
X_hist=pd.read_csv('tet_pattern_normalized_data_X_hist_AP.csv', sep=',',header=None)
#x = df
Y_hist= pd.read_csv('tet_pattern_normalized_label_Y_hist_AP.csv', sep=',',header=None)
X_addrs = pd.read_csv('tet_pattern_normalized_address_AP.csv', sep=',',header=None)
Y_hist = np.array(Y_hist)
X_hist = np.array(X_hist)
Y_hist = Y_hist[:,1]
X_hist = X_hist[:,1:]
#X_hist = []
#x = np.array(x)
#x = x[:,1:472]
#print(len(x))
print(Y_hist.shape)


# In[10]:


def feature_distance(feature1, feature2):
    dist = 0.0
    for i in range(len(feature1)):
        dist += abs((feature1[i]*1.0-feature2[i]*1.0)/(1.0+feature1[i]*1.0+feature2[i]*1.0))
        #print(dist)
    return dist
x = X_hist
print(x.shape)
#print(unique_name[X_addrs[0]])
print(Y_hist[0])
print(Y_hist[1])
#print(feature_distance(x[0], x[10]))
distance_list = []
query = x[0]
query_index = 0
for i in range(len(Y_hist)):
    distance_list.append(feature_distance(query, x[i]))
unsorted = zip(distance_list, Y_hist)
sorted_touple = sorted(unsorted, key = lambda element : element[0])
print(len(sorted_touple))


query_length =25
true_val = 0
false_val = 0
for i in range(query_length):
    if(sorted_touple[i][1] == Y_hist[query_index]):
        true_val = true_val + 1
    else:
        print("this is wrong "+str(i)+" no image is  confusing with "+str(sorted_touple[i][1]))
        false_val = false_val+1
print(true_val*1.0/query_length*1.0)


# In[11]:




#50 0.32
#45 0.37
#40 0.395
#35 0.411
#30 0.43
#25 0.49
#20 0.581
#15 0.61
#10 0.71
#5 0.73

query_length =50
true_val = 0
false_val = 0
for i in range(query_length):
    if(sorted_touple[i][1] == Y_hist[query_index]):
        true_val = true_val + 1
    else:
        print("this is wrong "+str(i)+" no image is  confusing with "+str(sorted_touple[i][1]))
        false_val = false_val+1
print(true_val*1.0/query_length*1.0)


# In[12]:


import numpy as np
def feature_distance(feature1, feature2):
    dist = 0.0
    f1 = np.array(feature1)
    f2 = np.array(feature2)
    dist += np.sum(f1-f2)/1+np.sum(f1+f2)
    return dist

#50 0.2
#45 0.28
#40 0.325
#35 0.371
#30 0.43
#25 0.48
#20 0.61
#15 0.61
#10 0.7
#5 0.7
f= open("AP_Pattern_full_result.txt","w+")
result = []
f.write("Label ")
for i in range(10):
    f.write(str((i+1)*10)+"% ")
f.write("\n")
for i in range(len(Y_hist)):
    if(i%100==0):
        print(i)
    part = 18
    val = 18
    cnt = 1
    true_val = 0
    false_val = 0
    distance_list = []
    query = x[i]
    query_index = i
    for j in range(len(Y_hist)):
        distance_list.append(feature_distance(query, x[j]))
    unsorted = zip(distance_list, Y_hist)
    sorted_touple = sorted(unsorted, key = lambda element : element[0])
    recall = []
    while(cnt!=11):
        for j in range(len(distance_list)):
            if(sorted_touple[j][1] == Y_hist[query_index]):
                true_val = true_val + 1
            if(true_val == val):
                recall.append(j)
                cnt+=1
                val = cnt*part
                if(cnt == 11):
                    break
    result.append(recall)
    f.write(str(Y_hist[query_index])+" ")
    for j in recall:
        f.write(str(j)+" ")
    f.write("\n")
    #print(recall)
            
    #print(len(sorted_touple))
f.close() 

'''
query_length =50
true_val = 0
false_val = 0
for i in range(query_length):
    if(sorted_touple[i][1] == Y_hist[query_index]):
        true_val = true_val + 1
    else:
        print("this is wrong "+str(i)+" no image is  confusing with "+str(sorted_touple[i][1]))
        false_val = false_val+1
print(true_val*1.0/query_length*1.0)
'''


# In[ ]:


for i in range (x.shape[0]):
    x1 = np.array(x[i][0:177])
    x1 = x1/np.sum(x1)
    x2 = np.array(x[i][177:413])
    x2 = x2/np.sum(x2)
    new_x = x1.tolist()+x2.tolist()
    #new_x = np.array(new_x)
    X_hist.append(new_x)

X_hist = np.array(X_hist)
Y_hist = np.array(Y_hist)
print(X_hist.shape, Y_hist.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(np.array(X_hist), np.array(Y_hist), test_size=0.3)
X_train = np.array(X_train)
print(len(X_train))


# In[ ]:


X_hist = np.array(X_hist)
Y_hist = np.array(Y_hist)

print(X_hist.shape, Y_hist.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(np.array(X_hist), np.array(Y_hist), test_size=0.3)
X_train = np.array(X_train)
print(len(X_train))

'''
for i in range(len(X_train)):
    if(len(X_train[i]) != 26):
       print(len(X_train[i]))
'''

print(len(Y_train))


# In[ ]:


from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
#clf = svm.SVC()
#clf = GaussianNB()
#clf = LogisticRegression()
#clf = MLPClassifier(alpha=1)
#clf = AdaBoostClassifier()
clf = RandomForestClassifier()
#clf = DecisionTreeClassifier()

#X_train = np.array(X_train)
print(type(X_train))

clf.fit(X_train,Y_train)

clf.score(X_test, Y_test)


# In[ ]:


import pickle
from sklearn.model_selection import cross_val_score
#clf = pickle.load(open('RandomForest_model.sav', 'rb'))
clf = RandomForestClassifier()
clf.fit(X_train,Y_train)
scores = cross_val_score(clf,X_test,Y_test,cv=5)
print((scores))
clf.score(X_test, Y_test)


# In[ ]:




