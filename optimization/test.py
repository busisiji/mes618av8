import redis
YuanFenZhongLiang=0
YuReFengLi = 30
YuReWenDu = 10
YuReShiJian = 0
GanZaoFengLi = 20
GanZaoWenDu = 10
GanZaoShiJian = 10
LengQueFengLi = 20
UserId = "VisionMaker"
r =redis.Redis(host="127.0.0.1",port=6379,db=5)
# r.lpush("processOptimization",("YuReFengLi : " + str(YuReFengLi), "\r\nYuReWenDu : " + str(YuReWenDu)))
# r.lpush(UserId + "_Optimization", ("\r\n拟合得分：" + str(0.3)))
r.set("processOptimization",("YuanFenZhongLiang:" + str(YuanFenZhongLiang)))
r.set("processOptimization",("YuReFengLi:" + str(YuReFengLi)))
r.set("processOptimization",("YuReWenDu:" + str(YuReFengLi)))