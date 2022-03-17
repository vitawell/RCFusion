import os
# 生成包括所有图片名和标签1的txt文件

def generate(dir,label):
    files = os.listdir(dir)
    files.sort()
    print('input :',dir)
    print('start...')
    listText = open(dir+'/'+'list.txt','w')
    for file in files:
        fileType = os.path.split(file)
        if fileType[1] == '.txt':
            continue
        name = file + ' ' + str(int(label)) +'\n'
        listText.write(name)
    listText.close()
    print('down!')

if __name__ == '__main__':
    generate('/Users/Vita/PycharmProjects/rcfusion-master/ocid_dataset/val_rgb',1)
