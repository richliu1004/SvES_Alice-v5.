import os
import cv2
import json
import math
import shutil
import numpy as np
import pandas as pd
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
import matplotlib.pyplot as plt

class GetFiles:
    def input_folder(self):
        root=tk.Tk()
        root.withdraw()
        fp=filedialog.askdirectory()
        #root.destroy()
        self.fp=fp
        return fp ## return the input folder's path

    def file_miner(self,fp):
        filename = os.listdir(fp)
        filepath=[]
        for i in range(len(filename)):
            filepath.append(str(Path(os.path.join(fp,filename[i]))))
        return filename, filepath ## return the files name and absolute path under the folder input

    def read_image(self):
        image_series=[]
        for i in range(len(self.df_fp)):
            image_series.append(cv2.imread(self.df_fp['images'][i],0))
        self.image_series=image_series
        #return self

    def read_mask(self):
        mask_series=[]
        for i in range(len(self.df_fp)):
            mask_series.append(cv2.imread(self.df_fp['masks'][i],0)>=40) ## the inequality(>=40) is to filter the slightly gradient of the mask's edge
        self.mask_series=mask_series
        #return self

    def read_label(self):
        label_series=[]
        for i in range(len(self.df_fp)):
            f=open(self.df_fp['label'][i])
            jdata=json.load(f)
            f.close
            label_series.append(jdata)
        self.label_series=label_series
        #return self

    def get_dataframe(self):
        root_folder_name,root_folder_path=self.file_miner(self.input_folder())
        self.root_folder_path=root_folder_path
        df_fn = pd.DataFrame() ## dataframe of filename
        df_fp = pd.DataFrame() ## dataframe of filepath
        for i in range(len(root_folder_path)):
            #if root_folder_name[i]!='images' and root_folder_name[i]!='masks' and root_folder_name[i]!='label': 
                #pass
            try:
                df_fn[root_folder_name[i]]=self.file_miner(root_folder_path[i])[0]
                df_fp[root_folder_name[i]]=self.file_miner(root_folder_path[i])[1]
            except:
                pass
        self.df_fn=df_fn
        self.df_fp=df_fp
        #return self

    def read_txt(self):
        txt_series=[]
        for i in range(len(self.df_fp)):
            f=open(self.df_fp['yolo_txt'][i],'r')
            content=[]
            for j in f.readlines():
                content.append(j)
            f.close()
            content[0]=content[0].split('\n')[0]
            for j in range(len(content)):
                content[j]=content[j].split(' ')
                for k in ['','0','2']:
                    while True:
                        try:
                            content[j].remove(k)
                        except:
                            break
            txt_series.append(content)
        self.txt_series=txt_series
        #return txt_series

    def main_BBOX_Kypt(self):
        self.get_dataframe()
        self.read_image()
        self.read_mask()
        self.read_label()
        return self

############################################################################################################
class BBOX_Kypt:
    def __init__(self,type):
        self.dataset=(GetFiles()).main_BBOX_Kypt()
        self.mask=self.dataset.mask_series ## segmentation mask
        self.label=self.dataset.label_series ## labelme json file with linestrip(s)
        self.df_fn=self.dataset.df_fn
        self.root=self.dataset.fp
        if type!='MP' and type!='KP':
            raise ValueError("Type is unidentified.")
        else: 
            self.type=type ## the process type (MP/KP)
        return
    
    def main(self):
        path=os.path.join(self.dataset.fp,'yolo_txt')
        if os.path.isdir(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)
        self.output_folder=path
        
        if self.type=='MP':
            self.mp_main()
        elif self.type=='KP':
            self.kp_main()

    def mp_main(self):
        for i in range(len(self.df_fn)):
            output_path=str(Path(os.path.join(self.output_folder,(str(self.df_fn['images'][i][:-4])+'.txt'))))
            print(output_path)
            self.yolo_txt_MP(self.mask[i],self.label[i],output_path)
        return

    def kp_main(self):
        for i in range(len(self.df_fn)):
            output_path=str(Path(os.path.join(self.output_folder,(str(self.df_fn['images'][i][:-4])+'.txt'))))
            print(output_path)
            self.yolo_txt_KP(self.mask[i],self.label[i],output_path)
        return

    def mp_processor(self,mask):
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
        center=round((B[1]+B[2])/2) ## len(B) should be "4" cause there will be 2 bboxes
        img_L,img_R=mask.copy(),mask.copy()
        img_L[center:]=0
        img_R[0:center]=0
        return img_R,img_L

    def get_yolo_coor(self,mask):
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
    
    def get_kypts_coor(self,jdata):
        image_height,image_width=jdata['imageHeight'],jdata['imageWidth']
        kypts=[]
        for i in range(len(jdata['shapes'])):
            k=[]
            #print(jdata['shapes'][i]['points'])
            for j in range(len(jdata['shapes'][i]['points'])):
                #print(jdata['shapes'][i]['points'][j])
                x,y=jdata['shapes'][i]['points'][j][0]/image_width,jdata['shapes'][i]['points'][j][1]/image_height
                #print(x,y)
                k.extend([x,y,2])
                #k=k+str(x)+str(' ')+str(y)+str(' ')+str('2')+str(' ')
                #print(k)
            #print(K)
            kypts.append(k)
        kypt_r,kypt_l=kypts[0],kypts[1]
        #print(kypts)
        return kypt_r, kypt_l

    def get_kypts_coor_kp(self,jdata):
        image_height,image_width=jdata['imageHeight'],jdata['imageWidth']
        kypts=[]
        for i in range(len(jdata['shapes'])):
            k=[]
            #print(jdata['shapes'][i]['points'])
            for j in range(len(jdata['shapes'][i]['points'])):
                #print(jdata['shapes'][i]['points'][j])
                x,y=jdata['shapes'][i]['points'][j][0]/image_width,jdata['shapes'][i]['points'][j][1]/image_height
                #print(x,y)
                k.extend([x,y,2])
                #k=k+str(x)+str(' ')+str(y)+str(' ')+str('2')+str(' ')
                #print(k)
            #print(K)
            kypts.append(k)
        kypt=kypts[0]
        #print(kypts)
        return kypt

    def yolo_txt_MP(self,mask,jdata,path):
        img_r,img_l = self.mp_processor(mask)
        kypt_r,kypt_l=self.get_kypts_coor(jdata)
        f=open(path,'w')
        lines=[str(x)+' ' for x in [0]+list(self.get_yolo_coor(img_r))]+[str(x)+' ' for x in kypt_r]+['\n']+[str(y)+' ' for y in [0]+list(self.get_yolo_coor(img_l))]+[str(x)+' ' for x in kypt_l]
        f.writelines(lines)
        f.close()
        return
    
    def yolo_txt_KP(self,mask,jdata,path):
        img = mask
        kypt=self.get_kypts_coor_kp(jdata)
        f=open(path,'w')
        lines=[str(x)+' ' for x in [0]+list(self.get_yolo_coor(img))]+[str(x)+' ' for x in kypt]+['\n']
        f.writelines(lines)
        f.close()
        return
    
    def examine(self):
        print('Examination start... the file which is not fit format will be shown below:')
        for i in range(len(self.df_fn)):
            B=[]
            list_img_90=list(sum(np.rot90(self.mask[i])))
            for j in range(len(list_img_90)): ## turn flatten mask into one dimensional binary boolean list
                if list_img_90[j]==0:
                    pass
                elif list_img_90[j]!=0:
                    list_img_90[j]=1
            for j in range(len(list_img_90)): ## check the continuity of the boolean list
                if j==0:
                    pass 
                elif list_img_90[j]==list_img_90[j-1]:
                    pass
                elif list_img_90[j]!=list_img_90[j-1]:
                    B.append(j)
            if self.type=='MP':
                if len(B)==4:
                    pass
                else:
                    print('Detected edges:',len(B),' (should be 4 in MP)')
                    print(self.df_fn['masks'][i])
                    print('--------------------------')
            elif self.type=='KP':
                if len(B)==2:
                    pass
                else:
                    print('Detected edges:',len(B),' (should be 2 in KP)')
                    print(self.df_fn['masks'][i])
                    print('--------------------------')
        print('Examination finished.')
############################################################################################################

class Unet_ROI:
    def __init__(self):
        self.dataset=(GetFiles()).main_BBOX_Kypt()
        self.image_series=self.dataset.image_series
        self.mask_series=self.dataset.mask_series
        self.dataset.read_txt()
        self.txt_series=self.dataset.txt_series
        return
    
    def croping(self,img,mask,txt,filename,side):
        img_width,img_height=np.shape(img)
        mask_width,mask_height=np.shape(mask)
        mask=mask*255 ## transfer boolean array into 8bit
        if img_width!=mask_width or img_height!=mask_height:
            img = cv2.resize(img,(1024,1024),interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask,(1024,1024),interpolation=cv2.INTER_NEAREST)
            img_width,img_height,mask_width,mask_height=1024,1024,1024,1024
        else:
            pass
        self.one_axis_defrom(img,mask,txt,img_width,img_height,filename,side)
        self.two_axis_defrom(img,mask,txt,img_width,img_height,filename,side)
        return

    def one_axis_defrom(self,img,mask,txt,img_width,img_height,filename,side):
        coor=[float(txt[1]),float(txt[0]),float(txt[3]),float(txt[2])]
        bx=[round((coor[0]-coor[2]/2)*img_width-10),round((coor[0]+coor[2]/2)*img_width+10)]
        by=[round((coor[1]-coor[3]/2)*img_height-10),round((coor[1]+coor[3]/2)*img_height+10)]
        img2 = cv2.resize(img[bx[0]:bx[1],by[0]:by[1]], (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask2 = cv2.resize(mask[bx[0]:bx[1],by[0]:by[1]], (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        ########################################################################################################################
        bbox_c,kypts_c = self.yolo_crop(txt,bx,by,img_width,img_height)
        ########################################################################################################################

        output_img_fn=filename.split('.')[0]+'_O'+str(side)+'.jpg'
        output_mask_fn=filename.split('.')[0]+'_O'+str(side)+'.jpg'
        output_label_fn=filename.split('.')[0]+'_O'+str(side)+'.txt'

        output_img_fp=os.path.join(os.path.join(self.output_folder,'images'),output_img_fn)
        output_mask_fp=os.path.join(os.path.join(self.output_folder,'masks'),output_mask_fn)
        output_label_fp=os.path.join(os.path.join(self.output_folder,'txts'),output_label_fn)
        cv2.imwrite(output_img_fp,img2)
        cv2.imwrite(output_mask_fp,mask2)
        f=open(output_label_fp,'w')
        lines=['0 ']+[str(x)+' ' for x in list(bbox_c)]+[str(kypts_c[x])+' '+str(kypts_c[x+1])+' 2 ' for x in range(0,len(kypts_c),2)]
        f.writelines(lines)
        f.close()

    def two_axis_defrom(self,img,mask,txt,img_width,img_height,filename,side):
        coor=[float(txt[1]),float(txt[0]),float(txt[3]),float(txt[2])]
        if coor[2]>coor[3]:
            coor[3]=coor[2]
        else:
            coor[2]=coor[3]
        bx=[round((coor[0]-coor[2]/2)*img_width-10),round((coor[0]+coor[2]/2)*img_width+10)]
        by=[round((coor[1]-coor[3]/2)*img_height-10),round((coor[1]+coor[3]/2)*img_height+10)]
        img2 = cv2.resize(img[bx[0]:bx[1],by[0]:by[1]], (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask2 = cv2.resize(mask[bx[0]:bx[1],by[0]:by[1]], (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        ########################################################################################################################
        bbox_c,kypts_c = self.yolo_crop(txt,bx,by,img_width,img_height)
        ########################################################################################################################

        output_img_fn=filename.split('.')[0]+'_T'+str(side)+'.jpg'
        output_mask_fn=filename.split('.')[0]+'_T'+str(side)+'.jpg'
        output_label_fn=filename.split('.')[0]+'_T'+str(side)+'.txt'

        output_img_fp=os.path.join(os.path.join(self.output_folder,'images'),output_img_fn)
        output_mask_fp=os.path.join(os.path.join(self.output_folder,'masks'),output_mask_fn)
        output_label_fp=os.path.join(os.path.join(self.output_folder,'txts'),output_label_fn)
        cv2.imwrite(output_img_fp,img2)
        cv2.imwrite(output_mask_fp,mask2)
        f=open(output_label_fp,'w')
        lines=['0 ']+[str(x)+' ' for x in list(bbox_c)]+[str(kypts_c[x])+' '+str(kypts_c[x+1])+' 2 ' for x in range(0,len(kypts_c),2)]
        f.writelines(lines)
        f.close()

    def one_axis_defrom_visualize(self,img,mask,txt,img_width,img_height,filename,side):
        coor=[float(txt[1]),float(txt[0]),float(txt[3]),float(txt[2])]
        ########################################################################################################################
        kypts=[float(txt[4]),float(txt[5]),float(txt[6]),float(txt[7]),float(txt[8]),float(txt[9]),float(txt[10]),float(txt[11])]
        ########################################################################################################################
        bx=[round((coor[0]-coor[2]/2)*img_width-10),round((coor[0]+coor[2]/2)*img_width+10)]
        by=[round((coor[1]-coor[3]/2)*img_height-10),round((coor[1]+coor[3]/2)*img_height+10)]
        img2 = cv2.resize(img[bx[0]:bx[1],by[0]:by[1]], (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask2 = cv2.resize(mask[bx[0]:bx[1],by[0]:by[1]], (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        ########################################################################################################################
        img_width_c,img_height_c=(by[1]-by[0]),(bx[1]-bx[0])
        bbox_c,kypts_c = self.yolo_crop(txt,bx,by,img_width,img_height)

        fig=plt.figure(figsize=(18,24))
        plt.subplot(321),plt.imshow(img,cmap='gray')
        plt.subplot(322),plt.imshow(img,cmap='gray'),plt.imshow(mask,cmap='gray',alpha=0.7)
        for i in range(0,len(kypts),2):
            plt.plot(kypts[i]*img_width,kypts[i+1]*img_height,'o')
        plt.subplot(323),plt.imshow(img[bx[0]:bx[1],by[0]:by[1]],cmap='gray')
        plt.subplot(324),plt.imshow(img[bx[0]:bx[1],by[0]:by[1]],cmap='gray'),plt.imshow(mask[bx[0]:bx[1],by[0]:by[1]],cmap='gray',alpha=0.7)

        x1,x2=math.floor((bbox_c[0]-bbox_c[2]/2)*img_width_c),math.ceil((bbox_c[0]+bbox_c[2]/2)*img_width_c)
        y1,y2=math.floor((bbox_c[1]-bbox_c[3]/2)*img_height_c),math.ceil((bbox_c[1]+bbox_c[3]/2)*img_height_c)
        bbox_yolo=np.zeros(np.shape(img[bx[0]:bx[1],by[0]:by[1]]))
        bbox_yolo[y1:y2+1,x1:x2+1]=255

        plt.imshow(bbox_yolo,cmap='Blues',alpha=0.5)
        for i in range(0,len(kypts_c),2):
            plt.plot(round(kypts_c[i]*img_width_c),round(kypts_c[i+1]*img_height_c),'o')
            
        plt.subplot(325),plt.imshow(img2,cmap='gray')
        plt.subplot(326),plt.imshow(img2,cmap='gray'),plt.imshow(mask2,cmap='gray',alpha=0.7)

        x1_ex,x2_ex=math.floor(x1*np.shape(img2)[1]/img_width_c),math.ceil(x2*np.shape(img2)[1]/img_width_c)
        y1_ex,y2_ex=math.floor(y1*np.shape(img2)[0]/img_height_c),math.ceil(y2*np.shape(img2)[0]/img_height_c)
        bbox_yolo_expand=np.zeros(np.shape(img2))
        bbox_yolo_expand[y1_ex:y2_ex+1,x1_ex:x2_ex+1]=255
        plt.imshow(bbox_yolo_expand,cmap='Blues',alpha=0.5)
        for i in range(0,len(kypts_c),2):
            plt.plot(round(kypts_c[i]*np.shape(img2)[1]),round(kypts_c[i+1]*np.shape(img2)[0]),'o')
        ########################################################################################################################

        output_img_fn=filename.split('.')[0]+'_O'+str(side)+'.jpg'
        output_mask_fn=filename.split('.')[0]+'_O'+str(side)+'.jpg'
        ###output_label_fn=filename.split('.')[0]+'_O'+str(side)+'.txt'
        output_img_fp=os.path.join(os.path.join(self.output_folder,'images'),output_img_fn)
        output_mask_fp=os.path.join(os.path.join(self.output_folder,'masks'),output_mask_fn)
        ###output_label_fp=os.path.join(os.path.join(self.output_folder,'txts'),output_mask_fn)
        cv2.imwrite(output_img_fp,img2)
        cv2.imwrite(output_mask_fp,mask2)

    def two_axis_defrom_visualize(self,img,mask,txt,img_width,img_height,filename,side):
        coor=[float(txt[1]),float(txt[0]),float(txt[3]),float(txt[2])]
        ########################################################################################################################
        kypts=[float(txt[4]),float(txt[5]),float(txt[6]),float(txt[7]),float(txt[8]),float(txt[9]),float(txt[10]),float(txt[11])]
        ########################################################################################################################
        if coor[2]>coor[3]:
            coor[3]=coor[2]
        else:
            coor[2]=coor[3]
        bx=[round((coor[0]-coor[2]/2)*img_width-10),round((coor[0]+coor[2]/2)*img_width+10)]
        by=[round((coor[1]-coor[3]/2)*img_height-10),round((coor[1]+coor[3]/2)*img_height+10)]
        img2 = cv2.resize(img[bx[0]:bx[1],by[0]:by[1]], (1024, 1024), interpolation=cv2.INTER_LINEAR)
        mask2 = cv2.resize(mask[bx[0]:bx[1],by[0]:by[1]], (1024, 1024), interpolation=cv2.INTER_NEAREST)
        
        ########################################################################################################################
        img_width_c,img_height_c=(by[1]-by[0]),(bx[1]-bx[0])
        bbox_c,kypts_c = self.yolo_crop(txt,bx,by,img_width,img_height)

        fig=plt.figure(figsize=(18,24))
        plt.subplot(321),plt.imshow(img,cmap='gray')
        plt.subplot(322),plt.imshow(img,cmap='gray'),plt.imshow(mask,cmap='gray',alpha=0.7)
        for i in range(0,len(kypts),2):
            plt.plot(kypts[i]*img_width,kypts[i+1]*img_height,'o')
        plt.subplot(323),plt.imshow(img[bx[0]:bx[1],by[0]:by[1]],cmap='gray')
        plt.subplot(324),plt.imshow(img[bx[0]:bx[1],by[0]:by[1]],cmap='gray'),plt.imshow(mask[bx[0]:bx[1],by[0]:by[1]],cmap='gray',alpha=0.7)

        x1,x2=math.floor((bbox_c[0]-bbox_c[2]/2)*img_width_c),math.ceil((bbox_c[0]+bbox_c[2]/2)*img_width_c)
        y1,y2=math.floor((bbox_c[1]-bbox_c[3]/2)*img_height_c),math.ceil((bbox_c[1]+bbox_c[3]/2)*img_height_c)
        bbox_yolo=np.zeros(np.shape(img[bx[0]:bx[1],by[0]:by[1]]))
        bbox_yolo[y1:y2+1,x1:x2+1]=255

        plt.imshow(bbox_yolo,cmap='Blues',alpha=0.5)
        for i in range(0,len(kypts_c),2):
            plt.plot(round(kypts_c[i]*img_width_c),round(kypts_c[i+1]*img_height_c),'o')
            

        plt.subplot(325),plt.imshow(img2,cmap='gray')
        plt.subplot(326),plt.imshow(img2,cmap='gray'),plt.imshow(mask2,cmap='gray',alpha=0.7)
        x1_ex,x2_ex=math.floor(x1*np.shape(img2)[1]/img_width_c),math.ceil(x2*np.shape(img2)[1]/img_width_c)
        y1_ex,y2_ex=math.floor(y1*np.shape(img2)[0]/img_height_c),math.ceil(y2*np.shape(img2)[0]/img_height_c)
        bbox_yolo_expand=np.zeros(np.shape(img2))
        bbox_yolo_expand[y1_ex:y2_ex+1,x1_ex:x2_ex+1]=255
        plt.imshow(bbox_yolo_expand,cmap='Blues',alpha=0.5)
        for i in range(0,len(kypts_c),2):
            plt.plot(round(kypts_c[i]*np.shape(img2)[1]),round(kypts_c[i+1]*np.shape(img2)[0]),'o')
        ########################################################################################################################

        output_img_fn=filename.split('.')[0]+'_T'+str(side)+'.jpg'
        output_mask_fn=filename.split('.')[0]+'_T'+str(side)+'.jpg'
        output_img_fp=os.path.join(os.path.join(self.output_folder,'images'),output_img_fn)
        output_mask_fp=os.path.join(os.path.join(self.output_folder,'masks'),output_mask_fn)
        cv2.imwrite(output_img_fp,img2)
        cv2.imwrite(output_mask_fp,mask2)

    def yolo_crop(self,txt,bx,by,img_width,img_height):
        bbox=txt[0:4]
        for i in range(len(bbox)):
            bbox[i]=float(bbox[i])
        kypts=txt[4:]
        for i in range(len(kypts)):
            kypts[i]=float(kypts[i])
        yolo_bx=by
        yolo_by=bx

        crop_width,crop_height=(yolo_bx[1]-yolo_bx[0]),(yolo_by[1]-yolo_by[0])
        width_scale,height_scale=(crop_width/img_width),(crop_height/img_height)
        crop_ori_x,crop_ori_y=yolo_bx[0]/img_width,yolo_by[0]/img_height

        bbox_c=[(bbox[0]-crop_ori_x)/width_scale,(bbox[1]-crop_ori_y)/height_scale,bbox[2]/width_scale,bbox[3]/height_scale]
        kypts_c=[]
        for i in range(0,len(kypts),2):
            k_x=(kypts[i]-crop_ori_x)/width_scale
            k_y=(kypts[i+1]-crop_ori_y)/height_scale
            kypts_c.extend([k_x,k_y])
        #print(bbox_c,kypts_c)
        return bbox_c,kypts_c


    def mask_8bit(mask):
        ## this function is used for transform the input image from unidentified data type into 8-bit(0-255)
        mask=np.round(mask/(mask.max()-mask.min())*255)
        return mask
    
    def main(self):
        path=os.path.join(self.dataset.fp,'Unet_ROI')
        if os.path.isdir(path):
            shutil.rmtree(path)
            os.makedirs(path)
            for i in ['images','masks','txts']:
                os.makedirs(os.path.join(path,i))
        else:
            os.makedirs(path)
            for i in ['images','masks','txts']:
                os.makedirs(os.path.join(path,i))
        self.output_folder=path
        for i in range(len(self.dataset.df_fn)):
            for j in range(len(self.txt_series[i])):
                self.croping(self.image_series[i],self.mask_series[i],self.txt_series[i][j],self.dataset.df_fn['images'][i],j)

import matplotlib.pyplot as plt
class visualize:
    def Unet2Yolo_bbox(self,image,mask):
        return
    

class test:
    def __init__(self):
        q = GetFiles.test_g()
        return
