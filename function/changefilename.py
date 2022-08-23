# TODO:用于更改xml中filename这个节点下的数据方便上传到服务器进行训练

import os


def rename():
    # 原始图片路径
    path = r'E:\cap_images_4\0'
    # 获取该路径下所有图片
    filelist = os.listdir(path)
    a = 1
    for files in filelist:
        # 原始路径
        Olddir = os.path.join(path, files)

        # if os.path.isdir(Olddir):
        #	continue
        # 将图片名切片,比如 xxx.bmp 切成xxx和.bmp
        # xxx
        filename = os.path.splitext(files)[0]
        # .bmp
        filetype = os.path.splitext(files)[1]
        # 需要存储的路径 a 是需要定义修改的文件名
        Newdir = os.path.join(path, 'test'+str(a) + filetype)
        os.rename(Olddir, Newdir)
        a += 1


rename()

