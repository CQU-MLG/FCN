#__author__ = 'charlie'
import numpy as np
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob

import TensorflowUtils as utils

# DATA_URL = 'http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016.zip'
DATA_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'


def read_dataset(data_dir):
    pickle_filename = "MIT_SceneParsing.pickle"
    #Data_zoo/hand_print/
    pickle_filepath = os.path.join(data_dir, pickle_filename)
    if not os.path.exists(pickle_filepath):
        # 不存在文件 则下载
        utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        #  # Data_zoo / MIT_SceneParsing / ADEChallengeData2016
        SceneParsing_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        #result =   {training: [{image: 图片全路径， annotation:标签全路径， filename:图片名字}] [][]
        #            validation:[{image:图片全路径， annotation:标签全路径， filename:图片名字}] [] []}
        #SceneParsing_folder = "hand_printImanges"
        result = create_image_lists(os.path.join(data_dir, SceneParsing_folder))

        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        print("result",result)
        training_records = result['training']
        validation_records = result['validation']
        del result
        print("training_records,validation_records",training_records,validation_records)
        #training: [{image: 图片全路径， annotation: 标签全路径， filename: 图片名字}]
    return training_records, validation_records

'''
  返回一个字典:
  image_list{ 
           "training":[{'image': image_full_name, 'annotation': annotation_file, 'image_filename': },......],
           "validation":[{'image': image_full_name, 'annotation': annotation_file, 'filename': filename},......]
           }
'''

def create_image_lists(image_dir):
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    # 训练集和验证集 分别制作
    directories = ['training', 'validation']
    image_list = {}
    for directory in directories:
        file_list = []
        image_list[directory] = []
        # 获取images directory下目录下所有的图片名
        print("image_dir",image_dir)
        file_glob = os.path.join(image_dir, "images", directory, '*.' + 'png')
        # 加入文件列表  包含所有图片文件全路径+文件名字  如 Data_zoo/MIT_SceneParsing/ADEChallengeData2016/images/training/hi.jpg
        file_list.extend(glob.glob(file_glob))
        if not file_list:
            print('No files found')
        else:
            for f in file_list:
                filename = os.path.splitext(f.split("\\")[-1])[0]
                annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                if os.path.exists(annotation_file):
                    #  image:图片全路径， annotation:标签全路径， filename:图片名字
                    record = {'image': f, 'annotation': annotation_file, 'filename': filename}
                    image_list[directory].append(record)
                else:
                    print("Annotation file not found for %s - Skipping" % filename)
        # 对图片列表进行洗牌
        random.shuffle(image_list[directory])
        # 包含图片文件的个数
        no_of_images = len(image_list[directory])
        print ('No. of %s files: %d' % (directory, no_of_images))

    return image_list

