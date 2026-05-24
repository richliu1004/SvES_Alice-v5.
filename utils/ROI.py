import os
import cv2
import numpy as np
import pandas as pd
import shutil

import label_process as Label
import test_module as Test
from funcMod import get_filepath as GetFilepath
from funcMod import data_crawler as Data

def imgread(fp_img):
    img=cv2.imread(fp_img,2)
    img=cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    # print(img.dtype)
    return img

# transform image or mask in the shortest dimension
def oneAxisDeform(img,mask,txt,expand=10): # the input txt here is not the same one as crop input. Fromat: txt=[0, 0.1, 0.2, 0.5, 0.5]
    img_width,img_height=1024,1024
    TXT=txt.copy()
    bx=[round((txt[1]-txt[3]/2)*img_height-expand),round((txt[1]+txt[3]/2)*img_height+expand)]
    by=[round((txt[0]-txt[2]/2)*img_width-expand),round((txt[0]+txt[2]/2)*img_width+expand)]
    imgc = cv2.resize(img[bx[0]:bx[1],by[0]:by[1]], (1024, 1024), interpolation=cv2.INTER_LINEAR)
    maskc = cv2.resize(mask[bx[0]:bx[1],by[0]:by[1]], (1024, 1024), interpolation=cv2.INTER_NEAREST)
    txt_c=yoloCropTrans(TXT,bx,by,img_width,img_height)
    return imgc,maskc,txt_c

# transform image or mask in 2 dimensions (evenly enlarge)
def twoAxisDeform(img,mask,txt,expand=10):
    img_width,img_height=1024,1024
    TXT=txt.copy()
    if txt[3]>txt[2]:
        txt[2]=txt[3]
    else:
        txt[3]=txt[2]
    bx=[round((txt[1]-txt[3]/2)*img_height-expand),round((txt[1]+txt[3]/2)*img_height+expand)]
    by=[round((txt[0]-txt[2]/2)*img_width-expand),round((txt[0]+txt[2]/2)*img_width+expand)]
    imgc = cv2.resize(img[bx[0]:bx[1],by[0]:by[1]], (1024, 1024), interpolation=cv2.INTER_LINEAR)
    maskc = cv2.resize(mask[bx[0]:bx[1],by[0]:by[1]], (1024, 1024), interpolation=cv2.INTER_NEAREST)
    txt_c=yoloCropTrans(TXT,bx,by,img_width,img_height)
    return imgc,maskc,txt_c

# transform yolo coordinate into cropped yolo coordinate and output txt
def yoloCropTrans(txt,bx,by,img_width,img_height):
    bbox,kypts=txt[0:4],txt[4:]
    yolo_bx,yolo_by=by,bx
    crop_width,crop_height=(yolo_bx[1]-yolo_bx[0]),(yolo_by[1]-yolo_by[0])
    width_scale,height_scale=(crop_width/img_width),(crop_height/img_height)
    crop_ori_x,crop_ori_y=yolo_bx[0]/img_width,yolo_by[0]/img_height
    bbox_c=[(bbox[0]-crop_ori_x)/width_scale,(bbox[1]-crop_ori_y)/height_scale,bbox[2]/width_scale,bbox[3]/height_scale]
    kypts_c=[]
    for i in range(0,len(kypts),2):
        k_x=(kypts[i]-crop_ori_x)/width_scale
        k_y=(kypts[i+1]-crop_ori_y)/height_scale
        kypts_c.extend([k_x,k_y])
    bbox_c.extend(kypts_c)
    return bbox_c

def filesAssign():
    print('hint from %s. Choose filepath include image, mask/combine, labels/txt:'%__name__)
    fp=GetFilepath()
    output=os.path.join(fp,'ROI')
    if os.path.isdir(output):
        shutil.rmtree(output)
        os.makedirs(output)
        for i in ['images','masks','txt']:
            os.makedirs(os.path.join(output,i))
    else:
        os.makedirs(output)
        for i in ['images','masks','txt']:
            os.makedirs(os.path.join(output,i))

    ImgFpList=Data(os.path.join(fp,'images'))
    MaskFpList=Data(os.path.join(os.path.join(fp,'masks'),'combine'))
    TxtFpList=Data(os.path.join(os.path.join(fp,'labels'),'txt'))
    D=pd.DataFrame(np.array([ImgFpList,MaskFpList,TxtFpList]).T,columns=['images','masks','txt'])
    return D,output

def main():
    D,output=filesAssign()
    # print(len(D['images']))
    for i in range(len(D['images'])):
        img=imgread(D['images'][i])
        mask=imgread(D['masks'][i])
        txt=Label.txt2list(D['txt'][i])
        filename=D['images'][i].split('\\')[-1].split('.')[0]
        for j in range(len(txt)):
            imgc,maskc,txtc=oneAxisDeform(img,mask,txt[j])
            imgfn=str(filename)+'_O'+str(j)+'.jpg'
            txtfn=str(filename)+'_O'+str(j)+'.txt'
            cv2.imwrite(os.path.join(os.path.join(output,'images'),imgfn),imgc)
            cv2.imwrite(os.path.join(os.path.join(output,'masks'),imgfn),maskc)
            Label.list2txt([txtc],os.path.join(os.path.join(output,'txt'),txtfn))

            imgc,maskc,txtc=twoAxisDeform(img,mask,txt[j])
            imgfn=str(filename)+'_T'+str(j)+'.jpg'
            txtfn=str(filename)+'_T'+str(j)+'.txt'
            cv2.imwrite(os.path.join(os.path.join(output,'images'),imgfn),imgc)
            cv2.imwrite(os.path.join(os.path.join(output,'masks'),imgfn),maskc)
            Label.list2txt([txtc],os.path.join(os.path.join(output,'txt'),txtfn))
    print('hint from %s: Process Finished. Filepath: %s'%(__name__,output))

if __name__=='__main__':
    D,output=filesAssign()
    print(len(D['images']))
    for i in range(len(D['images'])):
        img=imgread(D['images'][i])
        mask=imgread(D['masks'][i])
        txt=Label.txt2list(D['txt'][i])
        filename=D['images'][i].split('\\')[-1].split('.')[0]
        for j in range(len(txt)):
            imgc,maskc,txtc=oneAxisDeform(img,mask,txt[j])
            imgfn=str(filename)+'_O'+str(j)+'.jpg'
            txtfn=str(filename)+'_O'+str(j)+'.txt'
            cv2.imwrite(os.path.join(os.path.join(output,'images'),imgfn),imgc)
            cv2.imwrite(os.path.join(os.path.join(output,'masks'),imgfn),maskc)
            Label.list2txt([txtc],os.path.join(os.path.join(output,'txt'),txtfn))

            imgc,maskc,txtc=twoAxisDeform(img,mask,txt[j])
            imgfn=str(filename)+'_T'+str(j)+'.jpg'
            txtfn=str(filename)+'_T'+str(j)+'.txt'
            cv2.imwrite(os.path.join(os.path.join(output,'images'),imgfn),imgc)
            cv2.imwrite(os.path.join(os.path.join(output,'masks'),imgfn),maskc)
            Label.list2txt([txtc],os.path.join(os.path.join(output,'txt'),txtfn))
            # print(txt[j])
    print('hint from %s: Process Finished. Filepath: %s'%(__name__,output))
    