# -*- coding: utf-8 -*-
#feimengjuan 
# ÀûÓÃpythonÊµÏÖ¶àÖÖ·½·¨À´ÊµÏÖÍ¼ÏñÊ¶±ð 

import cv2 
import numpy as np 
#from matplotlib import pyplot as plt 

# ×î¼òµ¥µÄÒÔ»Ò¶ÈÖ±·½Í¼×÷ÎªÏàËÆ±È½ÏµÄÊµÏÖ 
def classify_gray_hist(image1,image2,size = (256,256)): 
    # ÏÈ¼ÆËãÖ±·½Í¼ 
    # ¼¸¸ö²ÎÊý±ØÐëÓÃ·½À¨ºÅÀ¨ÆðÀ´ 
    # ÕâÀïÖ±½ÓÓÃ»Ò¶ÈÍ¼¼ÆËãÖ±·½Í¼£¬ËùÒÔÊÇÊ¹ÓÃµÚÒ»¸öÍ¨µÀ£¬ 
    # Ò²¿ÉÒÔ½øÐÐÍ¨µÀ·ÖÀëºó£¬µÃµ½¶à¸öÍ¨µÀµÄÖ±·½Í¼ 
    # bins È¡Îª16 
    image1 = cv2.resize(image1,size) 
    image2 = cv2.resize(image2,size) 
    hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0]) 
    hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0]) 
    # ¿ÉÒÔ±È½ÏÏÂÖ±·½Í¼ 
    plt.plot(range(256),hist1,'r') 
    plt.plot(range(256),hist2,'b') 
    plt.show() 
    # ¼ÆËãÖ±·½Í¼µÄÖØºÏ¶È 
    degree = 0 
    for i in range(len(hist1)): 
        if hist1[i] != hist2[i]: 
            degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i])) 
        else: 
            degree = degree + 1 
    degree = degree/len(hist1) 
    return degree 

# ¼ÆËãµ¥Í¨µÀµÄÖ±·½Í¼µÄÏàËÆÖµ 
def calculate(image1,image2): 
    hist1 = cv2.calcHist([image1],[0],None,[256],[0.0,255.0]) 
    hist2 = cv2.calcHist([image2],[0],None,[256],[0.0,255.0]) 
    # ¼ÆËãÖ±·½Í¼µÄÖØºÏ¶È 
    degree = 0 
    for i in range(len(hist1)): 
        if hist1[i] != hist2[i]: 
            degree = degree + (1 - abs(hist1[i]-hist2[i])/max(hist1[i],hist2[i])) 
        else: 
            degree = degree + 1 
    degree = degree/len(hist1) 
    return degree 

# Í¨¹ýµÃµ½Ã¿¸öÍ¨µÀµÄÖ±·½Í¼À´¼ÆËãÏàËÆ¶È 
def classify_hist_with_split(image1,image2,size = (256,256)): 
    # ½«Í¼Ïñresizeºó£¬·ÖÀëÎªÈý¸öÍ¨µÀ£¬ÔÙ¼ÆËãÃ¿¸öÍ¨µÀµÄÏàËÆÖµ 
    image1 = cv2.resize(image1,size) 
    image2 = cv2.resize(image2,size) 
    sub_image1 = cv2.split(image1) 
    sub_image2 = cv2.split(image2) 
    sub_data = 0 
    for im1,im2 in zip(sub_image1,sub_image2): 
        sub_data += calculate(im1,im2) 
    sub_data = sub_data/3 
    return sub_data 

# Æ½¾ù¹þÏ£Ëã·¨¼ÆËã 
def classify_aHash(image1,image2): 
    image1 = cv2.resize(image1,(8,8)) 
    image2 = cv2.resize(image2,(8,8)) 
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY) 
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY) 
    hash1 = getHash(gray1) 
    hash2 = getHash(gray2) 
    return Hamming_distance(hash1,hash2) 

def classify_pHash(image1,image2): 
    image1 = cv2.resize(image1,(32,32)) 
    image2 = cv2.resize(image2,(32,32)) 
    gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY) 
    gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY) 
    # ½«»Ò¶ÈÍ¼×ªÎª¸¡µãÐÍ£¬ÔÙ½øÐÐdct±ä»» 
    dct1 = cv2.dct(np.float32(gray1)) 
    dct2 = cv2.dct(np.float32(gray2)) 
    # È¡×óÉÏ½ÇµÄ8*8£¬ÕâÐ©´ú±íÍ¼Æ¬µÄ×îµÍÆµÂÊ 
    # Õâ¸ö²Ù×÷µÈ¼ÛÓÚc++ÖÐÀûÓÃopencvÊµÏÖµÄÑÚÂë²Ù×÷ 
    # ÔÚpythonÖÐ½øÐÐÑÚÂë²Ù×÷£¬¿ÉÒÔÖ±½ÓÕâÑùÈ¡³öÍ¼Ïñ¾ØÕóµÄÄ³Ò»²¿·Ö 
    dct1_roi = dct1[0:8,0:8] 
    dct2_roi = dct2[0:8,0:8] 
    hash1 = getHash(dct1_roi) 
    hash2 = getHash(dct2_roi) 
    return Hamming_distance(hash1,hash2) 

# ÊäÈë»Ò¶ÈÍ¼£¬·µ»Øhash 
def getHash(image): 
    avreage = np.mean(image) 
    hash = [] 
    for i in range(image.shape[0]): 
        for j in range(image.shape[1]): 
            if image[i,j] > avreage: 
                hash.append(1) 
            else: 
                hash.append(0) 
    return hash 

# ¼ÆËãººÃ÷¾àÀë 
def Hamming_distance(hash1,hash2): 
    num = 0 
    for index in range(len(hash1)): 
        if hash1[index] != hash2[index]: 
            num += 1 
    return num 

def isDiff(img1,img2):
    #degree = classify_gray_hist(img1,img2) 
    #degree = classify_hist_with_split(img1,img2) 
    degree = classify_aHash(img1,img2) 
    #degree = classify_pHash(img1,img2) 
    #cv2.waitKey(0)
    if degree>=7:
        return False;
    else:
        return True;

if __name__ == '__main__': 
    img1 = cv2.imread('image22.jpg') 
    #cv2.imshow('img1',img1) 
    img2 = cv2.imread('image21.jpg') 
    #cv2.imshow('img2',img2)
    #degree = classify_gray_hist(img1,img2) 
    #degree = classify_hist_with_split(img1,img2) 
    degree = classify_aHash(img1,img2) 
    #degree = classify_pHash(img1,img2) 
    #print degree 
    cv2.waitKey(0) 
