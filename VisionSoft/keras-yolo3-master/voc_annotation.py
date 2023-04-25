# 导入相应的库文件
import xml.etree.ElementTree as ET # 解析xml文件
from os import getcwd # 获取文件工作路径
import os
import random
trainval_percent    = 0.9
train_percent       = 0.9
def generateImageSetTxt():
    random.seed(0)
    print("Generate txt in ImageSets.")
    xmlfilepath = "../data/Annotations"
    saveBasePath = "../data/ImageSets/Main"
    temp_xml = os.listdir(xmlfilepath)
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)

    num = len(total_xml)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("train size", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in list:
        name = total_xml[i][:-4] + '.xml\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")

def voc_annotation(basepath_data,basepath_keras):
    sets=[('vision', 'train'), ('vision', 'val'), ('vision', 'test'),('vision','trainval')]
    # 变量名，这里使用的主要是为了存储生成txt文件名，比较方便
    with open(basepath_keras + 'model_data/myclasses.txt', "r") as f:
        classes_txt = f.read()
    classes = []
    for classer_str in classes_txt.split("\n"):
        classes.append(classer_str)# 类别

    def convert_annotation(label, image_id, list_file):
        in_file = open(basepath_data + 'Annotations/%s'%(image_id),encoding='UTF-8')  # 打开相应目录下的xml文件

        tree=ET.parse(in_file) # 解析xml树

        root = tree.getroot() # 根节点

        for obj in root.iter('object'):  # 根据根节点，找子节点‘object’,每一个object子节点存放的是关于佩戴安全帽与否的详细注释
            difficult = obj.find('Difficult').text
            cls = obj.find('name').text  # 'name'节点，找已设定类别
            if cls not in classes or int(difficult)==1:  # 其他类别不考虑，直接跳出此次巡航进行下一次
                continue
            cls_id = classes.index(cls) # 将标签数值化，'0'hat,'1'person
            xmlbox = obj.find('bndbox')  # 每个'object'的矩形框的具体位置,左上角和右下角两个顶点（四个坐标）
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
            list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id)) # 将坐标点左上角横坐标，左上角纵坐标，
            #右下角横坐标，右下角纵坐标，标志0或1以逗号分隔，写入txt

    wd = getcwd() # 获取当前路径

    for label, image_set in sets:  # 循环，读取每个xml文件的信息
        image_dir = basepath_data + "ImageSets/Main/%s.txt"%image_set
        print(image_dir)
        image_ids = open(image_dir,encoding='UTF-8').read().strip().split()
        list_file = open(basepath_data + 'ImageSets/Main/%s_%s.txt'%(label, image_set), 'w')
        print(image_set)
        for image_id in image_ids:
            list_file.write(basepath_data + 'JPEGImages/%s.jpg'%(image_id[:-4])) # 将xml对应的图片名以相对路径的形式存储至txt，
            convert_annotation(label, image_id, list_file) # 同时只保留每个图片中bounding box 的坐标和对应的标签
            list_file.write('\n')
        list_file.close()

if __name__ == "__main__":
    generateImageSetTxt()
    voc_annotation('../data/',"")


