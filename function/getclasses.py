# TODO:得到xml文件中所有的类别
import xml.etree.ElementTree as ET
import os
import glob
def GetAnnotBoxLoc(file_dir):
    labelName= set([])
    files=glob.glob(file_dir)
    for i in range(len(files)):
        AnotPath=files[i]
        tree = ET.ElementTree(file=AnotPath)
        root = tree.getroot()
        ObjectSet = root.findall('object/name')
        for Object in ObjectSet:
            labelName.add(Object.text)
    return list(labelName)

path=r'\boot\bamboo\train\lables\*.xml'

s=GetAnnotBoxLoc(path)
print(s)