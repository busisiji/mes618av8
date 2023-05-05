import datasetDivide
import voc_annotation
import kmeans
import json
def dataready(trainval_percent,userId):
    basepath_data = '../../../' + userId + '/VisionSoft/data/'
    basepath_keras = '../../../' + userId + '/VisionSoft/keras-yolo3-master/'
    datasetDivide.datasetDivide(trainval_percent,basepath_data)
    voc_annotation.voc_annotation(basepath_data,basepath_keras)
    kmeans.kmeans(basepath_data,basepath_keras)


if __name__ == "__main__":
    dataready(0.9,"VisionMaker")
