# -*- coding: UTF-8 -*-
import os

# 生成包括所有图片名和标签1的txt文件
def generate(dir,label):
    files = os.listdir(dir)
    files.sort()
    print('input :',dir)
    print('start...')
    listText = open(dir+'/'+'list.txt','w')
    for file in files:
        (filenum, extension) = os.path.splitext(file)  # 如果文件后缀是png则继续
        if extension == '.png':
            fileType = os.path.split(file)
            if fileType[1] == '.txt':
                continue
            name = file + ' ' + str(int(label)) +'\n'
            listText.write(name)
    listText.close()
    print('down!')

# 重命名图片
def rere(dir):
    filenames = os.listdir(dir)
    for file in filenames:
        # print(name)  # 输出图片名称
        (filenum, extension) = os.path.splitext(file)  # 去掉文件后缀
        print(filenum)
        filen = filenum.split('_',2)[2]  # 按'_'分割两次，取第三个作为新文件名
        print(filen)
        os.rename( os.path.join(dir, file), os.path.join(dir,filen + '.png'))  # 替换文件名

if __name__ == '__main__':
    #generate('/Users/Vita/PycharmProjects/rcfusion-master/ocid_dataset/val_rgb',1)
    rere('/Users/Vita/Desktop/SfM/8.18 frames')
