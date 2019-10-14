## 作者
2017级硕士研究生 J-CheN丶 

## 介绍
该模型是FCN经典网络模型、数据预处理及读取

## 文件介绍
augment是一个语义分割增强的.py文件。其中的pics_aug（）函数实现了图像语义分割中的数据增强。
## 使用方法
根据hand_FCN.py文件新建相应的训练和预测文件夹，将训练集和测试集加入到响应文件夹。FCN主干网VGG-16模型的参数自己添加即可，运行hand_FCN.py即可运行。
augment根据pics_aug（）函数的参数为原图片文件夹路径、p.ground_truth()参数为ground_truth的路径。p.sample()参数为要增强的图片数目。
## 工具依赖
numpy、tensorflow、scipy、os、random
augment.py中需要Augmentor，PIL
