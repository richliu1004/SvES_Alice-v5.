import os
import shutil
import random
import numpy as np
import pandas as pd
import funcMod

def getDatabase():
    fp=funcMod.get_filepath()
    l=funcMod.data_crawler(fp)
    for i in range(len(l)):
        tag=l[i].split('\\')[-1]
        if tag=='images':
            img_database=l[i]
        elif tag=='masks':
            mask_database=l[i]
        elif tag=='txt':
            txt_database=l[i]
    
    imgFpList,maskFpList,txtFpList=[],[],[]
    for i in range(len(os.listdir(img_database))):
        imgFpList.append(os.path.join(img_database,os.listdir(img_database)[i]))
        maskFpList.append(os.path.join(mask_database,os.listdir(mask_database)[i]))
        txtFpList.append(os.path.join(txt_database,os.listdir(txt_database)[i]))

    d=pd.DataFrame(np.array([imgFpList,maskFpList,txtFpList]).T,columns=['images','masks','txt'])
    return d

def oneBase(d,outputfp,tnr=0.8,vlr=0.1,tsr=0.1):
    imgOut=os.path.join(outputfp,'images')
    imgOTr=os.path.join(imgOut,'train')
    imgOVl=os.path.join(imgOut,'val')
    imgOTs=os.path.join(imgOut,'test')
    maskOut=os.path.join(outputfp,'masks')
    maskOTr=os.path.join(maskOut,'train')
    maskOVl=os.path.join(maskOut,'val')
    maskOTs=os.path.join(maskOut,'test')
    txtOut=os.path.join(outputfp,'labels')
    txtOTr=os.path.join(txtOut,'train')
    txtOVl=os.path.join(txtOut,'val')
    txtOTs=os.path.join(txtOut,'test')
    folders=[imgOut,imgOTr,imgOVl,imgOTs,maskOut,maskOTr,maskOVl,maskOTs,txtOut,txtOTr,txtOVl,txtOTs]
    if os.listdir(outputfp)==[]:
        for i in folders:
            os.mkdir(i)
    elif os.listdir(outputfp)!=[]:
        shutil.rmtree(outputfp)
        os.mkdir(outputfp)
        for i in folders:
            os.mkdir(i)
    
    l=[*range(len(d))]
    random.shuffle(l)
    for i in range(len(d)):
        fileID=d['images'][l[i]].split('\\')[-1].split('.')[0]
        if i<=round(len(d)*tnr):
            shutil.copyfile(d['images'][l[i]],os.path.join(imgOTr,str(fileID+'.jpg')))
            shutil.copyfile(d['masks'][l[i]],os.path.join(maskOTr,str(fileID+'.jpg')))
            shutil.copyfile(d['txt'][l[i]],os.path.join(txtOTr,str(fileID+'.txt')))
        elif i>round(len(d)*tnr) and i<=round(len(d)*(tnr+vlr)):
            shutil.copyfile(d['images'][l[i]],os.path.join(imgOVl,str(fileID+'.jpg')))
            shutil.copyfile(d['masks'][l[i]],os.path.join(maskOVl,str(fileID+'.jpg')))
            shutil.copyfile(d['txt'][l[i]],os.path.join(txtOVl,str(fileID+'.txt')))
        elif i>round(len(d)*(tnr+vlr)):
            shutil.copyfile(d['images'][l[i]],os.path.join(imgOTs,str(fileID+'.jpg')))
            shutil.copyfile(d['masks'][l[i]],os.path.join(maskOTs,str(fileID+'.jpg')))
            shutil.copyfile(d['txt'][l[i]],os.path.join(txtOTs,str(fileID+'.txt')))
    print('oneBase shuffle completed. Path:',outputfp)

        
if __name__=='__main__':
    print('hint from: shuffle.py\nThis program require user to input a folder which contains images/masks/txt folders and choose an ouput path.')
    print('folders/files: images/img.jpg, masks/mask.jpg, txt/yolo.txt')
    oneBase(getDatabase(),funcMod.get_filepath())
    # print('Shuffle finished.')