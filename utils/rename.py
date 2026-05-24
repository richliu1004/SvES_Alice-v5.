import os
from funcMod import get_filepath
from funcMod import data_crawler


path=get_filepath()
datalist=os.listdir(path)
# print(data_crawler(path))
for i in range(len(datalist)):
    if datalist[i]=='desktop.ini':
        pass
    else:
        oldname=os.path.join(path,datalist[i])
        # for image
        # filename=datalist[i].split('_')[0]+datalist[i].split('_')[1]+str('_')+datalist[i].split('_')[2]

        # for mask
        filename=datalist[i].split('_')[0]+datalist[i].split('_')[1]+str('_')+datalist[i].split('_')[2]+str('_')+datalist[i].split('_')[3]
        
        # print(str(filename))
        newname=os.path.join(path,filename)
        os.rename(oldname,newname)