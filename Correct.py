
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
def del_dir(dic, direct):
    dic.pop(direct*25000, None)
    for i in true_items:
        dic.pop(direct*1000+i, None)

def get(image, idx, idy):
    if idx < (len(image)) and idy < len(image[0]) and idx>=0 and idy >=0:
        return 1
    else:
        return 0
def dir(x,y,img):
	ax=int(img[x+1,y])-int(img[x,y])
	ay=int(img[x,y-1])-int(img[x,y])
    
	if ax>=0 and ay>=0:
		return 1
	elif ax<0 and ay>=0:
		return 2
	elif ax<0 and ay<0:
		return 3
	elif ax>=0 and ay<0:
		return 4
		
def tetra(x,y,mat):
    out=[]
    fx=[0,-1,-1,-1,0,1,1,1]
    fy=[1,1,0,-1,-1,-1,0,1]
    val=mat[x][y]
    for i in range(0, 8):
        a=mat[x+fx[i]][y+fy[i]]
        if a==val:
            out.append(0)
        else:
            out.append(a)
    return out,val

def add_dict(dic,val1,val2,val3,direc):
    #print("val + direc = "+str(val)+"  "+ str(direc))
    d=1
    val=0
    c=1
    for i in range(1,5):
        if(d!=direc):
            if(c==1):
                val=val1
            elif(c==2):
                val=val2
            elif(c==3):
                val=val3
            if (1000*d+val) not in dic.keys():
                dic[25000*d] = dic[25000*d] +1
            else:
                dic[d*1000+val] = dic[d*1000+val] + 1
            c=c+1
        d=d+1
            
def dir_mat(img):
    mat=[]
    n=len(img)
    m=len(img[0])
    for i in range(n):
        mat.append([0] * m)
    
    for x in range(2, len(img)-2):
        for y in range(2, len(img[0])-2):
            mat[x][y]=dir(x,y,img)
    return mat

def local_tetra_pattern(img,dic1, dic2, dic3, dic4,j):
    try:
        mat=dir_mat(img)
        ab=1
        for x in range(2, len(img)-2):
            for y in range(2, len(img[0])-2):
                value, direc = tetra(x,y,mat)
                val1=0
                val2=0
                val3=0
                val4=0
                for i in range(0,len(value)):
                    if(value[i]==1):
                        val1+=2**(7-i)
                    elif(value[i]==2):
                        val2+=2**(7-i)
                    elif(value[i]==3):
                        val3+=2**(7-i)
                    elif(value[i]==4):
                        val4+=2**(7-i)

                if direc == 1:
                    add_dict(dic1, val2,val3,val4,direc)
                elif direc == 2:
                    add_dict(dic2, val1,val3,val4, direc)
                elif direc == 3:
                    add_dict(dic3,val1,val2,val4, direc)
                elif direc == 4:
                    add_dict(dic4, val1,val2,val3, direc)
    except TypeError as e:
        print ("type error "+str(j))
        ab=0
        for i in items:
            dic1[i]=1
            dic2[i]=1
            dic3[i]=1
            dic4[i]=1
        del_dir(dic1,1)
        del_dir(dic2,2)
        del_dir(dic3,3)
        del_dir(dic4,4)
    return  dic1, dic2, dic3, dic4,ab


# In[6]:


#to check uniform or not 
def uniform(pattern):
    pat= int(pattern)
    a=0
    b=0
    cnt=0
    for i in range(0,8):
        if( i==0 ):
            a= int(pattern/2**(7-i))
        else:
            b= int(pattern/2**(7-i))
            if(b!=a):
                cnt=cnt+1
                a=b
        pattern=pattern%2**(7-i)
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
        #print(i)
items.append(25000)
true_items.append(25000)
items.append(25000*2)
items.append(75000)
items.append(100000)
print(len(items))
for i in items:
    print(i)


# In[7]:


import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
#from keras.utils import np_utils
from scipy.stats import itemfreq
import time
from functools import wraps

df = pd.read_csv(shuffled_labels)
rows = df.iterrows()

X_addrs = []
X_hist = []
Y_hist = []
#row = rows[0]
j = 0

# uncomment below this for local tetra pattern
dic1 = {}
dic2 = {}
dic3 = {}
dic4 = {}
dic_m = {}
        
def create_dic():
    for i in range(len(items)):
        dic1[items[i]] = 0
        dic2[items[i]] = 0
        dic3[items[i]] = 0
        dic4[items[i]] = 0
    del_dir(dic1, 1)
    del_dir(dic2, 2)
    del_dir(dic3, 3)
    del_dir(dic4, 4)
    #print((dic1.keys()))
        
        
def magnitude_pattern(img, dic_m,j):
    for i in true_items:
        dic_m[i] = 0
    dic_m[250]=0
    out=[]
    ab=1
    fx=[0,-1,-1,-1,0,1,1,1]
    fy=[1,1,0,-1,-1,-1,0,1]
    try:
        for row in range(2, len(img)-2):
            for col in range(2, len(img[0])-2):
                val = 0
                centre = (int(img[row+1][col])-int(img[row][col]))**2 + (int(img[row][col-1])-int(img[row][col]))**2
                for i in range(0, 8):
                    new_row = row+fx[i]
                    new_col = row+fy[i]
                    a = (int(img[new_row+1][new_col])-int(img[new_row][new_col]))**2+(int(img[new_row][new_col-1])-int(img[new_row][new_col]))**2
                    if centre<=a:
                        val+=2**(7-i)
                    if val not in dic_m.keys():
                        dic_m[250] = dic_m[250] +1
                    else:
                        dic_m[val] = dic_m[val] + 1
    except TypeError as e:
        ab=0
        print ("type error magnitude "+str(j))
        for i in true_items:
            dic_m[i] = 1
    return dic_m,ab

start_time = time.time()      
for row in rows:
    #print(row[1][1])
    create_dic()
    img = cv2.imread(row[1][0],0)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = cv2.resize(img,(32,32))
    dic1, dic2, dic3, dic4,cnt = local_tetra_pattern(img,dic1, dic2, dic3, dic4,j)
    dic_m,cnt2 = magnitude_pattern(img, dic_m,j)
    
    new_x = []
    
    #change here 
    if cnt==1 and cnt2==1:
        for i in dic1.keys():
            new_x.append(dic1[i])
        for i in dic2.keys():
            new_x.append(dic2[i])
        for i in dic3.keys():
            new_x.append(dic3[i])
        for i in dic4.keys():
            new_x.append(dic4[i])
        for i in dic_m.keys():
            new_x.append(dic_m[i])
        new_x = np.array(new_x)
        hist = new_x/np.sum(new_x)
        #print(hist)
        X_addrs.append(row[1][0])
        X_hist.append(hist)
        Y_hist.append(row[1][1])
        if(j%100 == 0):
            elapsed_time = time.time() - start_time
            start_time = time.time()
            print(str(j)+" done in " + str(elapsed_time))
        j = j + 1
    
    

# uncomment up of this for local tetra pattern

# uncomment below this for lbp 


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
df.to_csv("tet_pattern_normalized_data_X_hist.csv",header=None)
df = pd.DataFrame(Y_hist)
df.to_csv("tet_pattern_normalized_label_Y_hist.csv",header=None)


# In[9]:


import pandas as pd
import numpy as np
df=pd.read_csv('tet_pattern_normalized_data_X_hist.csv', sep=',',header=None)
x = df
Y_hist= pd.read_csv('tet_pattern_normalized_label_Y_hist.csv', sep=',',header=None)
Y_hist = np.array(Y_hist)
Y_hist = Y_hist[:,1]
X_hist = []
x = np.array(x)
x = x[:,1:768]
print(len(x))
print(Y_hist.shape)


# In[10]:


for i in range (x.shape[0]):
    x1 = np.array(x[i][0:177])
    x1 = x1/np.sum(x1)
    x2 = np.array(x[i][177:354])
    x2 = x2/np.sum(x2)
    x3 = np.array(x[i][354:531])
    x3 = x3/np.sum(x3)
    x4 = np.array(x[i][531:708])
    x4 = x4/np.sum(x4)
    x5 = np.array(x[i][708:767])
    x5 = x5/np.sum(x5)
    new_x = x1.tolist()+x2.tolist()+x3.tolist()+x4.tolist()+x5.tolist()
    #new_x = np.array(new_x)
    X_hist.append(new_x)

X_hist = np.array(X_hist)
Y_hist = np.array(Y_hist)
print(X_hist.shape, Y_hist.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(np.array(X_hist), np.array(Y_hist), test_size=0.3)
X_train = np.array(X_train)
print(len(X_train))


# In[11]:


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
#clf = RandomForestClassifier()
clf = DecisionTreeClassifier()

#X_train = np.array(X_train)
print(type(X_train))

clf.fit(X_train,Y_train)

clf.score(X_test, Y_test)


# In[12]:


import pickle
from sklearn.model_selection import cross_val_score
#clf = pickle.load(open('RandomForest_model.sav', 'rb'))
clf = RandomForestClassifier()
clf.fit(X_train,Y_train)
scores = cross_val_score(clf,X_test,Y_test,cv=50)
print((scores))
clf.score(X_test, Y_test)


# In[13]:


#lbp-> Local derivative ternary lbp last e  tetra 

