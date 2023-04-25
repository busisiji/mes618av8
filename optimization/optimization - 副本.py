import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import redis

print("hello")
try:
    print("hello")
    YuReFengLi = 0 if (int(sys.argv[1]) == 0) else 1
    YuReWenDu = 0 if (int(sys.argv[2]) == 0) else 1
    YuReShiJian = 0 if (int(sys.argv[3]) == 0) else 1
    GanZaoFengLi = 0 if (int(sys.argv[4]) == 0) else 1
    GanZaoWenDu = 0 if (int(sys.argv[5]) == 0) else 1
    GanZaoShiJian = 0 if (int(sys.argv[6]) == 0) else 1
    LengQueFengLi = 0 if (int(sys.argv[7]) == 0) else 1
    UserId = str(sys.argv[8])
    flag = [YuReFengLi, YuReWenDu, YuReShiJian, GanZaoFengLi, GanZaoWenDu, GanZaoShiJian, LengQueFengLi]
    chooselist = ["YuReFengLi", "YuReWenDu", "YuReShiJian", "GanZaoFengLi", "GanZaoWenDu", "GanZaoShiJian",
                  "LengQueFengLi"]
    # client = pymongo.MongoClient(host = '127.0.0.1',port = 27017)
    # db = client.SanXiang2021
    # collection = db.Opyimization

    r = redis.Redis(host="127.0.0.1", port=6379)
except Exception as e:
    print(e)
# result = collection.find({'industry': '机床', 'location': '滚动轴承'})

class filerw:
    def _init_(self):
        pass

    def choose_list(self):
        chooselist = [1*YuReFengLi,2*YuReWenDu,3*YuReShiJian,4*GanZaoFengLi,5*GanZaoWenDu,6*GanZaoShiJian,7*LengQueFengLi,8]
        chooselist = [x for x in chooselist if x != 0]
        for i in range(len(chooselist)):
            chooselist[i] = chooselist[i] - 1
        return chooselist
    def read_data(self, filestring):
        # chooselist = self.choose_list()
        # adv_data = pd.read_csv(filestring,usecols = chooselist)
        adv_data = pd.read_csv(filestring)
        r.lpush(UserId + "_Optimization", "\r\nhead:" + str(adv_data.head()) + "\r\nShape:" + str(adv_data.shape))
        # print('head:', adv_data.head(), '\nShape:', adv_data.shape)
        return adv_data

    def rebuild_data(self, data):
        new_adv_data = data.iloc[1:, 1:]
        r.lpush(UserId + "_Optimization", "\r\nhead:" + str(new_adv_data.head()) + "\r\nShape:" + str(new_adv_data.shape))
        # print('head:', new_adv_data.head(), '\nShape:', new_adv_data.shape)
        return new_adv_data

    def describe_data(self, data):
        r.lpush(UserId + "_Optimization",str(data.describe()))
        r.lpush(UserId + "_Optimization", str(data[data.isnull() == True].count()))
        # print(data.describe())
        # print(data[data.isnull() == True].count())

    def write_data(self, data, filestring):
        data0 = data[0]
        data1 = data[1]
        data2 = data[2]
        data3 = data[3]
        data4 = data[4]
        data5 = data[5]
        data6 = data[6]
        data7 = data[7]
        data8 = data[8]
        data9 = data[9]
        data10 = data[10]
        data11 = data[11]
        data12 = data[12]
        data13 = data[13]
        data14 = data[14]
        data15 = data[15]
        data16 = data[16]
        out_data = pd.DataFrame(
            {"YuReFengLi": [data0], "YuReWenDu": [data1], "YuReShiJian": [data2], "ZhiLiFengLi": [data3],
             "ZhiLiWenDu": [data4], "PenWuShiJian": [data5], "PenWuJianGeShiJian": [data6], "PenWuCiShu": [data7],
             "PenWuYaLi": [data8], "PenWuShiFengLi": [data9], "GanZaoFengLi": [data10], "GanZaoWenDu": [data11],
             "GanZaoShiJian": [data12], "LengQueFengLi": [data13], "LengQueShiJian": [data14],
             "ShaiXuanShiJian": [data15],
             "ChengPinLv": [data16]})
        out_data.to_csv(filestring, index=False)


class Linear:
    def _init_(self):
        pass

    def getTrainTestData(self, trainData, trainTarget):
        X_train, X_test, Y_train, Y_test = train_test_split(trainData, trainTarget, test_size=0.4, random_state=0)
        return X_train, X_test, Y_train, Y_test

    def getModel(self, X_train, Y_train):
        model = LinearRegression()
        model.fit(X_train, Y_train)
        a = model.intercept_
        b = model.coef_
        return model, a, b

    def getScore(self, model, X_test, Y_test):
        score = model.score(X_test, Y_test)
        Y_pred = model.predict(X_test)
        return score, Y_pred


class Plot:
    def _init_(self):
        pass

    def PlotScatter(self, data):
        xvars = []
        for i in range(len(chooselist)):
            if flag[i] != 0:
                xvars.append(chooselist[i])
        sns.pairplot(data,
                     x_vars=xvars,
                     y_vars=["ChengPinLv"], height=7, aspect=0.8, kind="reg")
        plt.savefig('../../' + UserId + '/optimization/pairplot.png')

    def Figure(self, x_lenth, Y_data):
        plt.figure()
        plt.plot(range(x_lenth), Y_data/100.0, 'b', label="成品率")
        plt.legend(loc="upper right")  # 显示图中的标签
        plt.xlabel("测试集编号")
        plt.ylabel('成品率')
        plt.savefig("fig.png")


def Matdot(mat1, mat2):
    multi = np.dot(mat1, mat2)
    return multi


def optimization():
    r = redis.Redis(host="127.0.0.1", port=6379)
    adv = filerw()
    adv_data = adv.read_data("ProcessData.csv")
    new_data = adv.rebuild_data(adv_data)
    adv.describe_data(new_data)
    r.lpush(UserId+"_Optimization","new : " + str(new_data), "\r\nold : " + str(adv_data))
    # print("new", new_data, "old", adv_data)

    pic = Plot()
    pic.PlotScatter(adv_data)

    linear = Linear()
    X_train, X_test, Y_train, Y_test = linear.getTrainTestData(new_data.iloc[:, :16], new_data.ChengPinLv)
    r.lpush(UserId + "_Optimization", "\r\n原始数据特征:"+ str(new_data.iloc[:, :16].shape) + "\r\n训练数据特征:" + str(X_train.shape) + "\r\n测试数据特征:"+ str(X_test.shape))
    r.lpush(UserId + "_Optimization", "\r\n原始数据标签:"+ str(new_data.ChengPinLv.shape) + "\r\n训练数据标签:" + str(Y_train.shape) + "\r\n测试数据标签:", str(Y_test.shape))
    # print("原始数据特征:", new_data.iloc[:, :16].shape, ",训练数据特征:", X_train.shape, ",测试数据特征:", X_test.shape)
    # print("原始数据标签:", new_data.ChengPinLv.shape, ",训练数据标签:", Y_train.shape, ",测试数据标签:", Y_test.shape)

    model, b, k = linear.getModel(X_train, Y_train)
    # print("最佳拟合线:截距", b, ",回归系数：", k)
    r.lpush(UserId + "_Optimization",("\r\n最佳拟合线:截距" + str(b) + ",回归系数：" + str(k)))
    score, Y_pred = linear.getScore(model, X_test, Y_test)
    r.lpush(UserId + "_Optimization", ("\r\n拟合得分：" + str(score)))
    r.lpush(UserId + "_Optimization", "END")
    # print(score)
    # best_data = [280, 46, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    # best_out = Matdot(best_data, k)
    # print(best_out)
    # best_data.append(best_out)
    # print(best_data)
    # adv.write_data(best_data, "out.csv")
YuReFengLi = 1
YuReWenDu = 0
YuReShiJian = 1
GanZaoFengLi = 0
GanZaoWenDu = 1
GanZaoShiJian = 0
LengQueFengLi = 1
UserId = "VisionMaker"


#
# if __name__ == '__main__':
    # print("hello")
    # try:
    #     print("hello")
    #     YuReFengLi = 0 if (int(sys.argv[1]) == 0) else 1
    #     YuReWenDu = 0 if (int(sys.argv[2]) == 0) else 1
    #     YuReShiJian = 0 if (int(sys.argv[3]) == 0) else 1
    #     GanZaoFengLi = 0 if (int(sys.argv[4]) == 0) else 1
    #     GanZaoWenDu = 0 if (int(sys.argv[5]) == 0) else 1
    #     GanZaoShiJian = 0 if (int(sys.argv[6]) == 0) else 1
    #     LengQueFengLi = 0 if (int(sys.argv[7]) == 0) else 1
    #     UserId = str(sys.argv[8])
    #     flag = [YuReFengLi, YuReWenDu, YuReShiJian, GanZaoFengLi, GanZaoWenDu, GanZaoShiJian, LengQueFengLi]
    #     chooselist = ["YuReFengLi", "YuReWenDu", "YuReShiJian", "GanZaoFengLi", "GanZaoWenDu", "GanZaoShiJian",
    #                   "LengQueFengLi"]
    #     # client = pymongo.MongoClient(host = '127.0.0.1',port = 27017)
    #     # db = client.SanXiang2021
    #     # collection = db.Opyimization
    #
    #     r = redis.Redis(host="127.0.0.1", port=6379)
    # except Exception as e:
    #     print(e)
    # # result = collection.find({'industry': '机床', 'location': '滚动轴承'})
optimization()
