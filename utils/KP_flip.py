import numpy as np
import os
import cv2
import shutil
import test_module as T
from label_process import txt2list
from label_process import list2txt
from funcMod import get_filepath
from funcMod import data_crawler

def imgread(fp_img):
    img=cv2.imread(fp_img,2)
    # print(img.dtype)
    return img

def txtread(fp_txt):
    txt=[]
    content=[]
    f=open(fp_txt,'r')
    for i in f.readlines():
        content.append(i)
    f.close()
    content[0]=content[0].split('\n')[0]
    for i in range(len(content)):
        content[i]=content[i].split(' ')
        for j in ['','0','2','\n']:
            while True:
                try:
                    content[i].remove(j)
                except:
                    break
    txt=content
    return txt

def flip(txt):
    # mask_f=cv2.flip(mask,0)
    for i in [1,5,7,9,11]:
        txt[0][i]=1-float(txt[0][i])
    return txt

def generateFliptxtData(fp_txt,outputfolder):
    txt=txt2list(fp_txt)
    txt_f=flip(txt)
    fileID=fp_txt.split('\\')[-1].split('.')[0]+'_H.txt'
    output=os.path.join(outputfolder,fileID)
    list2txt(txt_f,output)
    # print(output)

def generateFlipimgData(fp_img,outputfolder):
    img=imgread(fp_img)
    img_f=cv2.flip(img,0)
    fileID=fp_img.split('\\')[-1].split('.')[0]+'_H.jpg'
    output=os.path.join(outputfolder,fileID)
    cv2.imwrite(output,img_f)

def main():
    input_fp=get_filepath()
    imgfp=os.path.join(input_fp,'images')
    maskfp=os.path.join(os.path.join(input_fp,'masks'),'combine')
    txtfp=os.path.join(os.path.join(input_fp,'labels'),'txt')
    output_fp=os.path.join(input_fp,'flip')
    imgout=os.path.join(output_fp,'images')
    maskout=os.path.join(os.path.join(output_fp,'masks'),'combine')
    txtout=os.path.join(os.path.join(output_fp,'labels'),'txt')

    for i in [output_fp,imgout,maskout,txtout]:
        if os.path.isdir(i):
            shutil.rmtree(i)
            os.makedirs(i)
        else:
            os.makedirs(i)

    for i in range(len(os.listdir(imgfp))):
        # print(imgfp)
        generateFlipimgData(os.path.join(imgfp,os.listdir(imgfp)[i]),imgout)
        generateFlipimgData(os.path.join(maskfp,os.listdir(maskfp)[i]),maskout)
        generateFliptxtData(os.path.join(txtfp,os.listdir(txtfp)[i]),txtout)
    print('hint from: %s KPFlip complete, filepath: %s'%(__name__,output_fp))
    



if __name__=='__main__':
    fp_img=r'C:\Users\LIU\Desktop\test_funcMOD\KP\images\N40KPH02_0084.jpg'
    fp_mask=r'C:\Users\LIU\Desktop\test_funcMOD\KP\masks\conbine\N40KPH02_0084.jpg'
    fp_txt=r'C:\Users\LIU\Desktop\test_funcMOD\KP\label\txt\N40KPH02_0084.txt'

    outputfolder=r'C:\Users\LIU\Desktop\flip'

    # T.imgstack(imgread(fp_mask),T.makepoint(imgread(fp_mask),txtread(fp_txt))[0])
    # T.imgstack(imgread(fp_img),T.makepoint(imgread(fp_mask),txtread(fp_txt))[1])

    # # flip
    # img_f,txt_f=flip(imgread(fp_img),txtread(fp_txt))
    # mask_f=cv2.flip(imgread(fp_mask),0)

    # T.imgstack(mask_f,T.makepoint(mask_f,txt_f)[0])
    # T.imgstack(img_f,T.makepoint(mask_f,txt_f)[1])
    # cv2.destroyWindow('img')
    generateFliptxtData(fp_txt,outputfolder)
