from __future__ import print_function
import os, time, pickle
import numpy as np
import tensorflow.compat.v1 as tf

#path='E:/BaiduNetdiskDownload/Div150Adhoc_test/'
path='E:/BaiduNetdiskDownload/Div400_train/'
fileList = os.listdir(path)
# for file in fileList:
#     if len(os.listdir(path + file)) !=300:
#         print(file,len(os.listdir(path + file)))
#
# incandescent_lightbulb 275
# Mont_Saint_Michel_tides 208
# Pingxi_Sky_Lantern_Festival 176
# spilled_milk 299
# surveillance_cameras 299



f = open('E:/BaiduNetdiskDownload/poiNameCorrespondences_devset.txt')
label=[]
for line in f.readlines():
    #print(line)
    label.append(' '.join(line.split()[:-1]))
#print(label)


#resnet_model_path = 'E:/BaiduNetdiskDownload/model/resnet_v1_50.ckpt'
import tensorflow.keras.preprocessing as preprocessing
import tensorflow as tf
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def extract_resnet(img_path, model):
    img = preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    dense1_output = model.predict(img)
    #print(dense1_output.shape)
    #print(dense1_output[0][0][0])  # 2048
    return dense1_output[0][0][0]

def extract_main(path, fileList):
    model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet')
    dense_layer = tf.keras.layers.AvgPool2D(pool_size=(7,7))(model.output)
    model = tf.keras.Model(inputs=model.input, outputs=dense_layer)

    image_feature = []
    for file in fileList:
        fi = os.listdir(path+file)
        for i in fi:
            image_feature.append(list(extract_resnet(path+file+'/'+i, model)))
        print(file, 'extract end')

    #print(image_feature)
    print(len(image_feature),len(image_feature[0]))
    with open('./data/div400/train_img_feats.pkl', 'wb') as f:
        pickle.dump(image_feature, f, pickle.HIGHEST_PROTOCOL)


extract_main(path, fileList)



# import csv
# import os
# import numpy as np
# path = 'E:/BaiduNetdiskDownload/descCNN/'
# fileList = os.listdir(path)
# for file in fileList:
#     print(path+file)
#     csv_file = csv.reader(open(path+file))
#     for item in csv_file:
#         #print(np.array(item, dtype=int64))
#         print(item)
#         print(len(item))
#         break
#     break