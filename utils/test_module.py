import numpy as np
import cv2
import os

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

def makepoint(mask,txt):
    mask_frame=np.zeros((np.shape(mask)))
    # print(np.shape(mask))
    height=np.shape(mask)[0]
    width=np.shape(mask)[1]
    for i in range(len(txt)):
        for j in range(len(txt[i])):
            txt[i][j]=float(txt[i][j])
    bbox_mask=mask_frame.copy()
    kypt_mask=mask_frame.copy()
    for i in range(len(txt)):
        bbox=np.round(np.array(txt[i][0:4])*np.array([width,height,width,height]))
        kypt01=np.round(np.array(txt[i][4:6])*np.array([width,height]))
        kypt02=np.round(np.array(txt[i][6:8])*np.array([width,height]))
        kypt03=np.round(np.array(txt[i][8:10])*np.array([width,height]))
        kypt04=np.round(np.array(txt[i][10:12])*np.array([width,height]))
        # print(bbox)
        # print(kypt01)
        # print(kypt02)
        # print(kypt03)
        # print(kypt04)
        bbox_mask[round(bbox[1]-bbox[3]/2):round(bbox[1]+bbox[3]/2),round(bbox[0]-bbox[2]/2):round(bbox[0]+bbox[2]/2)]=255
        bbox_mask=np.array(bbox_mask,dtype=np.uint8)
        # print(bbox_mask.dtype)
        r=2
        for i in [kypt01,kypt02,kypt03,kypt04]:
            # print(i)
            kypt_mask[int(i[1])-r:int(i[1])+r,int(i[0])-r:int(i[0])+r]=255
        kypt_mask=np.array(kypt_mask,dtype=np.uint8)
    return bbox_mask,kypt_mask


def imgstack(img,mask):
    cv2.imshow('img',cv2.addWeighted(img,0.7,mask,0.5,20))
    # cv2.imshow('img2',mask)
    cv2.waitKey(0)


if __name__ == '__main__':
    fp_img=r'C:\Users\LIU\Desktop\test_20250218\MP\image\0.jpg'
    fp_mask=r'C:\Users\LIU\Desktop\test_20250218\MP\mask\0.jpg'
    fp_txt=r'C:\Users\LIU\Desktop\test_20250218\MP\txt\0.txt'

    fp_img=r'C:\Users\LIU\Desktop\\01\images\N34MP01_0001.jpg'
    fp_mask=r'C:\Users\LIU\Desktop\01\masks\combine\N34MP01_0001.jpg'
    fp_txt=r'C:\Users\LIU\Desktop\01\labels\txt\N34MP01_0001.txt'

    fp_img=r"E:\0_AI_model_storage\20241217_MP_binary\AL_test\masks\N37MP01_0086_T0.jpg"
    fp_mask=r"E:\0_AI_model_storage\20241217_MP_binary\AL_test\masks\N37MP01_0086_T0.jpg"
    fp_txt=r"E:\0_AI_model_storage\20241217_MP_binary\AL_test\txts\N37MP01_0086_T0.txt"

    # fp_img=r'C:\Users\LIU\Desktop\01\ROI\images\N34MP01_0001_O0.jpg'
    # fp_mask=r'C:\Users\LIU\Desktop\01\ROI\masks\N34MP01_0001_O0.jpg'
    # fp_txt=r'C:\Users\LIU\Desktop\01\ROI\txt\N34MP01_0001_O0.txt'

    # imgstack(imgread(fp_img),imgread(fp_mask))
    # print(txtread(fp_txt))

    imgstack(imgread(fp_mask),makepoint(imgread(fp_mask),txtread(fp_txt))[0])
    imgstack(imgread(fp_img),makepoint(imgread(fp_mask),txtread(fp_txt))[1])
    # makepoint(imgread(fp_mask),txtread(fp_txt))