import glob   
path = '/home/user/resistor/test/*.jpg'   
files=glob.glob(path)   
for file in files: 
    print(file)
