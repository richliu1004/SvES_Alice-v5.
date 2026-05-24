import os
import shutil
import cv2
import json
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

def get_filepath(): # return: folder filepath (default: folder which contains images/masks/label)
    r=tk.Tk()
    r.withdraw()
    fp=filedialog.askdirectory()
    r.destroy()
    return fp

# old version
# def data_crawler(fp):
    filename=os.listdir(fp)
    filepath=[]
    for i in range(len(filename)):
        filepath.append(str(Path(os.path.join(fp,filename[i]))))
    return filepath

def data_crawler(fp):
    filepath=[str(Path((os.path.join(fp,x)))) for x in os.listdir(fp) if x!='desktop.ini']
    return filepath ##all files' filepath under the input filepath(fp)

def trans_uint8(img):
    img=img/(img.max()-img.min())*255
    return img

def read_image_G(fp):
    img=cv2.imread(fp,2)
    if img.dtype=='uint8':
        pass
    else:
        img=trans_uint8(img)
    return img

def read_mask_G(fp,threshold=40):
    mask=cv2.imread(fp,2)
    if mask.dtype=='uint8':
        pass
    else:
        mask=trans_uint8(mask)
    mask=mask>=threshold
    return mask

def mask_divider(mask): # bio mask edge detect and divide, return: mp*2 / kp*1
    B=[]
    list_img_90=list(sum(np.rot90(mask)))
    for i in range(len(list_img_90)): ## turn flatten mask into one dimensional binary boolean list
        if list_img_90[i]==0:
            pass
        elif list_img_90[i]!=0:
            list_img_90[i]=1
    for i in range(len(list_img_90)): ## check the continuity of the boolean list
        if i==0:
            pass 
        elif list_img_90[i]==list_img_90[i-1]:
            pass
        elif list_img_90[i]!=list_img_90[i-1]:
            B.append(i)
    if len(B)==4:
        center=round((B[1]+B[2])/2) ## len(B) should be "4" cause there will be 2 bboxes
        img_L,img_R=mask.copy(),mask.copy()
        img_L[center:]=0
        img_R[0:center]=0
        mask_output=[img_R,img_L]
    elif len(B)==2:
        mask_output=[mask]
    else:
        raise ValueError('Edge detection fail. There is undesired pixel in mask, please try to fix the mask image and try again.')
    return mask_output

def read_label(fp): # json file reader, return: jdata
    f=open(fp)
    jdata=json.load(f)
    f.close
    return jdata

def create_folder(fp):
    if os.path.isdir(fp):
        shutil.rmtree(fp)
        os.makedirs(fp)
    else:
        os.makedirs(fp)
    return

class Filepath:
    def __init__(self):
        fp=get_filepath()
        if os.path.isdir(fp):
            self.root_fp=fp
            pass
        else:
            raise KeyError('Path unfound.')
        l=data_crawler(fp)
        for i in l:
            if i.split('\\')[-1]=='images':
                self.images_fp=i
            elif i.split('\\')[-1]=='masks':
                self.masks_fp=i
                self.meta_masks_fp=os.path.join(self.masks_fp,'metacarpal')
                self.trap_masks_fp=os.path.join(self.masks_fp,'trapezium')
            elif i.split('\\')[-1]=='labels':
                self.label_fp=i
                self.json_label_fp=os.path.join(self.label_fp,'json')
        return
    
    def combine_fp(self):
        self.conb_masks_fp=os.path.join(self.masks_fp,'combine')
        return self.conb_masks_fp
    
    def yl_fp(self):
        self.yolo_fp=os.path.join(self.label_fp,'txt')
        return self.yolo_fp

# to carry filepath of mask folder 
class Mask_Combination:
    def __init__(self,meta_fp,trap_fp,output_fp): # fp is the folder which contains metacarpal and trapizium segmentation masks
        if os.path.isdir(meta_fp) and os.path.isdir(trap_fp):
            pass
        else:
            raise KeyError('Path unfound.')
        self.meta_filelist=data_crawler(meta_fp)
        self.trap_filelist=data_crawler(trap_fp)
        self.output=output_fp
        return
    
    def union(self,m1,m2): # to make meta and trap masks into one piece
        img=np.array(((1-(1-m1/255)*(1-m2/255)))*255,np.uint8)
        return img
    
    def combine(self): # require input:(1.)meta_filelist (2.)trap_filelist / output: combine mask
        self.combine_image=[]
        for i in range(len(self.meta_filelist)):
            meta=read_image_G(self.meta_filelist[i])
            trap=read_image_G(self.trap_filelist[i])
            self.combine_image.append(self.union(meta,trap))
        return self
    
    def img_save(self):
        # make output folder: combine
        output_fp=self.output
        create_folder(output_fp)
        for i in range(len(self.meta_filelist)):
            if len(self.meta_filelist[i].split('\\')[-1].split('_'))==2:
                filename = self.meta_filelist[i].split('\\')[-1].split('_')[0]+'.jpg'
            elif len(self.meta_filelist[i].split('\\')[-1].split('_'))==3:
                filename = self.meta_filelist[i].split('\\')[-1].split('_')[0]+'_'+self.meta_filelist[i].split('\\')[-1].split('_')[1]+'.jpg'
            cv2.imwrite(os.path.join(output_fp,filename),self.combine_image[i])
            print('img saved',os.path.join(output_fp,filename))
        self.combine_fp=output_fp
        return
    
    def test(self): # this funciton is to check if the mask image in the folder(fp) contains undesired pixels in the binary masks
        return
    
class Yolo_label_Generator:
    def __init__(self,mask_c_fp,json_fp,output_fp):
        self.m_list=data_crawler(mask_c_fp)
        self.j_list=data_crawler(json_fp)
        self.output_fp=output_fp
        return
    
    def generate_sub_bbox(self,mask):
        width,height=mask.shape
        A,B=[],[]
        list_img,list_img_90=list(sum(mask)),list(sum(np.rot90(mask)))
        for i in range(len(list_img)):
            if list_img[i]==0:
                pass
            elif list_img[i]!=0:
                A.append(i)
        for j in range(len(list_img_90)):
            if list_img_90[j]==0:
                pass
            elif list_img_90[j]!=0:
                B.append(j)
        #return min(B),max(B),min(A),max(A)
        y1,y2,x1,x2 = min(B)-2,max(B)+2,min(A)-2,max(A)+2 ## the coordinations of numpy and yolo are different, x axis is verticle in numpy while horizental in yolo
        yolo_mid_x,yolo_mid_y=round((x1+x2)/2)/width,round((y1+y2)/2)/height
        yolo_dx,yolo_dy=round(abs(x1-x2))/width,round(abs(y1-y2))/height
        return yolo_mid_x,yolo_mid_y,yolo_dx,yolo_dy
    
    # func generate_bbox enable to check how many bbox does the input mask contain
    def generate_bbox(self,mask):
        mask=mask_divider(mask)
        if len(mask)==1:
            bbox=[self.generate_sub_bbox(mask[0])]
        elif len(mask)==2:
            bbox=[]
            for i in mask:
                sub_bbox=[self.generate_sub_bbox(i)]
                bbox.extend(sub_bbox)
        else:
            raise KeyError('hint: number of BBOXs is out of range.(MP==2,KP==1)\nThe bbox detected in this mask:%s'%len(mask))
        return bbox
    
    def generate_kypt(self,jdata):
        image_height,image_width=jdata['imageHeight'],jdata['imageWidth']
        kypts=[]
        for i in range(len(jdata['shapes'])):
            k=[]
            for j in range(len(jdata['shapes'][i]['points'])):
                x,y=jdata['shapes'][i]['points'][j][0]/image_width,jdata['shapes'][i]['points'][j][1]/image_height
                k.extend([x,y,2])
            kypts.append(k)
        return kypts # if len(kypts)==2 meas the jdata belongs to MP label, len(kypts)==1 means jdata belongs to KP label

    def label_save(self,bbox,kypts,output):
        lines=[]
        f=open(output,'w')
        for i in range(len(bbox)):
            l=[str(x)+' ' for x in [0]+list(bbox[i])]+[str(x)+' ' for x in list(kypts[i])]+['\n']
            lines.extend(l)
        f.writelines(lines)
        f.close()
        print('label saved:',output)
        return

def combine_masks():
    f=Filepath()
    ###
        # the folder which contains all kinds of data including:
        # (1.)image
        # (2.)mask
        # (3.)label
    ###
    mc=Mask_Combination(f.meta_masks_fp,f.trap_masks_fp,f.combine_fp())
    create_folder(f.conb_masks_fp)
    mc.combine()
    mc.img_save()
    print('hint: Mask Combination Complete. Filepath:',mc.combine_fp)

def Yolo_label():
    f=Filepath()
    f.combine_fp() # generate conbined masks filepath for yolo label generator
    f.yl_fp() # generate yolo label filepath for yolo label_save
    create_folder(f.yolo_fp)
    Y=Yolo_label_Generator(f.conb_masks_fp,f.json_label_fp,f.yolo_fp)
    for i in range(len(Y.m_list)):
        mask=read_mask_G(Y.m_list[i])
        js=read_label(Y.j_list[i])
        output_filepath=os.path.join(Y.output_fp,Y.j_list[i].split('\\')[-1].split('.')[0]+'.txt')
        Y.label_save(Y.generate_bbox(mask),Y.generate_kypt(js),output_filepath)
    print('hint: Yolo label transformation conpleted. Filepath:',Y.output_fp)
    