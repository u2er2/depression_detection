import os
import pickle
import cv2
import pandas as pd
import numpy as np

# path = "/home/ctong/repro-SKS/DNN/output_dir/vocab.pkl"

# file_list = {}
# for file in os.listdir(path):
#     f = open(os.path.join(path, file), 'rb')
#     f = pickle.load(f)
#     file_list[file.split('-')[-1].split('.')[0]] = f

# print(file_list)

# # 展示一幅图
# img = file_list['train']['image_data'][0]
# cv2.imwrite("img.png", img)
data=pd.read_pickle('/home/ctong/repro-SKS/DNN/output_dir/vocab.pkl')
print(data)
