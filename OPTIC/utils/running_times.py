

import time

# time.clock()默认单位为s
# 获取开始时间
start = time.perf_counter()
'''
代码开始
'''
sum = 0
for i in range(100):
    for j in range(100):
        sum = sum + i + j
print("sum = ", sum)
'''
代码结束
'''
# 获取结束时间
end = time.perf_counter()
# 计算运行时间
runTime = end - start
runTime_ms = runTime * 1000
# 输出运行时间
print("运行时间：", runTime, "秒")
print("运行时间：", runTime_ms, "毫秒")


# from torchsummaryX import summary
# summary(net, input)