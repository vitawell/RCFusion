# -*- coding: UTF-8 -*-
from PIL import Image
import os

##批量裁剪文件夹中的图片
# 需要裁剪的图片文件夹
path_img = '/Users/Vita/PycharmProjects/rcfusion-master/ocid_dataset/train_rgb'
img_dir = os.listdir(path_img)
# print(img_dir)

'''
（左上角坐标(x,y)，右下角坐标（x+w，y+h）
'''

# for i in range(len(img_dir)):
#     #根据图片名称提取id,方便重命名
#     id = int((img_dir[i].split('.')[0]).split('_')[1])
#     img = Image.open(path_img + img_dir[i])

for file in img_dir:
    (filen, extension) = os.path.splitext(file)  # 如果文件后缀是png则继续
    if extension == '.png':
        # imgname = os.path.split(file)

        img = Image.open(path_img + '/' + file)

        print(filen)

        size_img = img.size
        # print(size_img) # （1920，1080）
        x = 0
        y = 0
        #这里需要均匀裁剪几张，就除以根号下多少，这里我需要裁剪25张，根号25=5（5*5）
        w = int(size_img[0] / 5)
        h = int(size_img[1] / 5)
        for k in range(5):
            for v in range(5):
                region = img.crop((x + k * w, y + v * h, x + w * (k + 1), y + h * (v + 1)))

                #保存图片的位置以及图片名称
                region.save('/Users/Vita/PycharmProjects/rcfusion-master/ocid_dataset/newtrainrgb/' + filen + '_crop' + '%d%d' % (k, v) + '.png')

