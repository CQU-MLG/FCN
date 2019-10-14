# -*- coding: utf-8 -*-
"""
Created on Lab 7.5  2019
@author: J-CheN
"""

# 导入数据增强工具
import Augmentor
import cv2 as cv
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
aug_testpics = r"E:\jxg\SLic\img\training"
aug_testlabel_pics =r"E:\jxg\SLic\train\img_label"
aug_testlabels = r"E:\jxg\SLic\train\labels"

images_dir = r"E:\jxg\SLic\crop_train36_1\image"
label_image_dir = r"E:\jxg\SLic\crop_train36_1\labels"

re_labelimage = r"E:\jxg\SLic\crop_train16_1\re_labelimage"

piaoju_img_path = r"E:\jxg\piaoju\images"
piaoju_label_path = r"E:\jxg\piaoju\labels"


def pics_aug(save_img_dir):
    # 确定原始图像存储路径以及掩码文件存储路径
    p = Augmentor.Pipeline(save_img_dir)
    p.ground_truth(aug_testlabels)

    # 图像旋转： 按照概率0.8执行，最大左旋角度10，最大右旋角度10
    p.rotate(probability=0.4, max_left_rotation=1, max_right_rotation=1)

    # 图像左右互换： 按照概率0.5执行
    p.flip_left_right(probability=0.5)
    '''
    # 图像放大缩小： 按照概率0.8执行，面积为原始图0.85倍
    p.zoom_random(probability=0.3, percentage_area=0.9)
    '''
    # 最终扩充的数据样本数
    p.sample(100)
    '''
    p.rotate(probability=0.2, max_left_rotation=2, max_right_rotation=2)  # 旋转
    #p.flip_left_right(probability=0.5)  # 按概率左右翻转
    p.zoom_random(probability=0.2, percentage_area=0.9)  # 随即将一定比例面积的图形放大至全图
    p.skew_top_bottom(probability=0.6,magnitude=0.2)
    #p.shear(probability=1, max_shear_left=15, max_shear_right=15)
    #p.zoom(probability=0.5, min_factor=1.2, max_factor=1.5)
    #p.flip_top_bottom(probability=0.6)  # 按概率随即上下翻转
    #p.random_distortion(probability=0.6, grid_width=10, grid_height=10, magnitude=20)  # 小块变形
    p.sample(1400)'''

def move_pics(img_dir,save_img_dir):
    for img_name in os.listdir(img_dir):
        print(img_name)
        img_path =os.path.join(img_dir,img_name)
        re_imgname = img_name.split(".")[0]+".png"
        image = cv.imread(img_path)
        cv.imwrite(os.path.join(save_img_dir,re_imgname),image,[int(cv.IMWRITE_PNG_COMPRESSION), 5])


def img_name_process(path):
    i = 0
    for img_name in os.listdir(path):
        i =i+1
        print(i,"****",img_name)
        if "_original_" in img_name:
            image = cv.imread(os.path.join(path,img_name))
            re_imgname = img_name.split("_original_")[1]
            print(re_imgname)
            cv.imwrite(r"E:\jxg\SLic\crop_train64\image"+"\\"+re_imgname,image,[int(cv.IMWRITE_PNG_COMPRESSION), 5])
        if "groundtruth_(1)_image_" in img_name:
            image = cv.imread(os.path.join(path,img_name))
            re_imgname = img_name.split("groundtruth_(1)_image_")[1]
            print(re_imgname)
            cv.imwrite(r"E:\jxg\SLic\crop_train64\lables"+"\\"+re_imgname,image)
if __name__ == '__main__':
    print("hello")
    save_img_dir = r"E:\jxg\SLic\crop_train64\image"
    #move_pics(aug_testpics,save_img_dir)
    #pics_aug(save_img_dir)
    img_name_process(r"E:\jxg\SLic\crop_train64\output")