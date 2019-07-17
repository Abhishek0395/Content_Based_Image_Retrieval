import os
from shutil import copyfile
lst = os.listdir("allRotatedImages")

#for i in range(112):
 #   dir_name = "Catagorized_images/catagory_"+str(i)
 #   os.makedirs(dir_name)
#print(len(lst))
#print(lst[0][5])
"""
for i in range(len(lst)):
    if(lst[i][0] == 'D'):
        j = 1
        dir_name = lst[i][j]
        j = j+1
        while(lst[i][j] != '_'):
            dir_name = dir_name+lst[i][j]
            j = j+1
        #print("allRotatedImages/"+lst[i])
        copyfile("allRotatedImages/"+lst[i],"Catagorized_images/catagory_"+dir_name+"/"+lst[i])

lst2  = os.listdir("Catagorized_images")
for folder in lst2:
    nlst = os.listdir("Catagorized_images/"+folder)
    print(len(nlst))
"""
