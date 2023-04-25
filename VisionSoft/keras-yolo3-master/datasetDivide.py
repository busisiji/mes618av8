import os
import random

def datasetDivide(trainval_percent,basepath_data):


    # trainval_percent = 0.95   # 训练集：验证集：测试集=8：1：1
    tran_percent = 8.0/9

    xmlpath = basepath_data + 'Annotations'  # xml文件路径
    txtsavepath = basepath_data + 'ImageSets/Main' # txt文件存储路径

    xmllist = os.listdir(xmlpath)  # 获取当前路径下所有xml文件

    num = len(xmllist)
    listnum = range(num)

    trainvalnum = int(num*trainval_percent)
    trainnum = int(trainvalnum * tran_percent)

    trainval_dataset = random.sample(listnum, trainvalnum)   # 随机划分
    train_dataset = random.sample(trainval_dataset, trainnum)

    ftrainval = open(txtsavepath + "/trainval.txt", "w")
    ftrain = open(txtsavepath + "/train.txt", "w")
    fval = open(txtsavepath + "/val.txt", "w")
    ftest = open(txtsavepath + "/test.txt", "w")

    for i in listnum:
        # name = xmllist[i].split('.')[0]
        name = xmllist[i]
        if i in trainval_dataset:
            ftrainval.write(name + '\n')
            if i in train_dataset:
                ftrain.write(name + '\n')
            else:
                fval.write(name + '\n')
        else:
            ftest.write(name + '\n')


    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

if __name__ == '__main__':
    datasetDivide(0.9,'../data/')