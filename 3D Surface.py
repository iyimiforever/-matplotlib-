"""
化学反应动力学曲线的绘制
A+B->C->D
compound_A 原料
compound_B 辅料
compound_C 产物
compound_D 副产物
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model as lm
import sklearn.metrics as sm
from mpl_toolkits.mplot3d import axes3d

#
# # 读取数据
# files_A = ['80_degrees_Celsius.csv',
#            '90_degrees_Celsius.csv',
#            '100_degrees_Celsius.csv']
# files_A_list=[]
# for file in files_A:
#     data=pd.read_csv(file)
#     name=file.split('.')[0]
#     files_A_list.append((data,name))
#
#
# # 浓度随时间的变化曲线
# for i, item in enumerate(files_A_list):
#     plt.figure("Concentration-Time"+item[1], facecolor="yellow")
#
#     # 获取当前坐标轴
#     ax = plt.gca()
#     ax.spines['top'].set_color('none')
#     ax.spines['right'].set_color('none')
#     plt.xlim(0, max(item[0]['time']))
#     plt.ylim(0, 7)
#     # 设置刻度定位器
#     maj_loc = eval("plt.AutoLocator()")
#     ax.xaxis.set_major_locator(maj_loc)
#     plt.xlabel("time/min", fontsize=14)
#     plt.ylabel("c/mol*L-1", fontsize=14)
#
#     plt.title(item[1], fontsize=16)
#     plt.grid(linestyle=":")
#     plt.plot(item[0]['time'],item[0]['A'],color="dodgerblue",label="compound A")
#     plt.plot(item[0]['time'],item[0]['B'],color="orange",label="compound B")
#
# # 线性回归结果列表 R2，斜率，截距
# linear_regression_result=[[],[],[]]
#
# for i, item in enumerate(files_A_list):
#     # ln(浓度)随时间的变化曲线
#     plt.figure("ln(Concentration)-Time"+item[1], facecolor="yellow")
#
#     # 获取当前坐标轴
#     ax = plt.gca()
#     ax.spines['top'].set_color('none')
#     ax.spines['right'].set_color('none')
#     plt.xlim(0, max(item[0]['time']))
#     # 设置刻度定位器
#     maj_loc = eval("plt.AutoLocator()")
#     ax.xaxis.set_major_locator(maj_loc)
#     plt.xlabel("time/min", fontsize=14)
#     plt.ylabel("ln(c)", fontsize=14)
#
#     plt.title(item[1], fontsize=16)
#     plt.grid(linestyle=":")
#     plt.scatter(item[0]['time'][1:],np.log(item[0]['A'])[1:],s=70,color="dodgerblue",label="compound A")
#     plt.scatter(item[0]['time'][1:],np.log(item[0]['B'])[1:],s=70,color="orange",label="compound B")
#
#     # A
#     # 创建线性回归模型
#     model = lm.LinearRegression()
#     # 训练模型
#     x=item[0]['time'][1:].values.reshape(-1,1)
#     y=np.log(item[0]['A']).reshape(-1,1)[1:]
#     model.fit(x,y)
#     # 预测
#     pred_y=model.predict(x)
#     # # 评估训练结果误差
#     R2=sm.r2_score(y,pred_y)
#     linear_regression_result[i].append((R2,model.coef_.tolist()[0][0], model.intercept_.tolist()[0]))
#     # print("R2：",R2)
#     # print("线性模型的斜率与截距：", model.coef_, model.intercept_)
#     # 绘制回归线
#     plt.plot(x, pred_y, c='blue', label='Regression A')
#
#     # B
#     # 创建线性回归模型
#     model = lm.LinearRegression()
#     # 训练模型
#     x=item[0]['time'][1:].values.reshape(-1,1)
#     y=np.log(item[0]['B']).reshape(-1,1)[1:]
#     model.fit(x,y)
#     # 预测
#     pred_y=model.predict(x)
#     # # 评估训练结果误差
#     R2=sm.r2_score(y,pred_y)
#     linear_regression_result[i].append((R2,model.coef_.tolist()[0][0], model.intercept_.tolist()[0]))
#     # print("R2",R2)
#     # print("线性模型的斜率与截距：", model.coef_, model.intercept_)
#     # 绘制回归线
#     plt.plot(x, pred_y, c='orange', label='Regression B')
#
#     # A*B
#     model = lm.LinearRegression()
#     # 训练模型
#     x=item[0]['time'][1:].values.reshape(-1,1)
#     y=np.log(item[0]['B']).reshape(-1,1)[1:]+np.log(item[0]['A']).reshape(-1,1)[1:]
#     model.fit(x,y)
#     # 预测
#     pred_y=model.predict(x)
#     # # 评估训练结果误差
#     R2=sm.r2_score(y,pred_y)
#     linear_regression_result[i].append((R2,model.coef_.tolist()[0][0], model.intercept_.tolist()[0]))
#     # print("R2",R2)
#     # print("线性模型的斜率与截距：", model.coef_, model.intercept_)
#     # 绘制回归线
#     plt.plot(x, pred_y, c='yellow', label='Regression C')
#
# # print(linear_regression_result)
# # [[(0.9759816512511589, -0.0036237140295617377, 1.3620512765758315),
# # (0.9985436138196061, -0.002551451152775102, 1.714421974640958),
# # (0.9892957005687226, -0.006175165182336841, 3.0764732512167896)],
# # [(0.9884269018783889, -0.007701479623791293, 1.2689171279341829),
# # (0.9973275894395159, -0.005837842156531827, 1.6430723959733136),
# # (0.9955462367226635, -0.013539321780323119, 2.911989523907496)],
# # [(0.9270754838711863, -0.008481019309214409, 0.8188477303474255),
# # (0.9916598741725188, -0.011367040996225767, 1.7207662768640417),
# # (0.9926304366042502, -0.019848060305440177, 2.5396140072114672)]]
#
#
# # 根据Arrhenius方程，温度越高，反应速率越快，到达反应终点所用的时间越短，方程式如下：
# # k=k0 * e * (- Ea / (R * T))
# # 在上式中，k0为指前因子，单位为min-1；R为摩尔气体常数，取8.314 J mol-1 K-1；Ea为反应活化能，单位为J mol-1。将上式两边取自然对数：
# # lnk = -Ea / (R * T) + lnk0
# # 以各温度下的lnk对1/T进行作图，由直线的斜率和截距就可以计算得到反应活化能Ea及指前因子k0，最终确立第一步反应动力学方程：
#
# k_t_dict={}
# # ln(k)-1/t关系图 A
# k=[linear_regression_result[0][0][1],linear_regression_result[1][0][1],linear_regression_result[2][0][1]]
# t=[80+273.15,90+273.15,100+273.15]
# t=np.array(t)
# k=np.array(k)
# x=1/t.reshape(-1,1)
# y=np.log(-k)
#
# # plt.figure("ln(k)-1/t A", facecolor="yellow")
# #
# # # 获取当前坐标轴
# # ax = plt.gca()
# # ax.spines['top'].set_color('none')
# # ax.spines['right'].set_color('none')
# # # 设置刻度定位器
# # maj_loc = eval("plt.AutoLocator()")
# # ax.xaxis.set_major_locator(maj_loc)
# # plt.xlabel("1/t", fontsize=14)
# # plt.ylabel("ln(k)", fontsize=14)
# # plt.xlim(0.0026, 0.0029)
# # plt.ylim(-6.5, -4)
# #
# # plt.title('ln(k)-1/t', fontsize=16)
# # plt.grid(linestyle=":")
# # plt.scatter(x, y, s=70, color="dodgerblue")
#
# model = lm.LinearRegression()
# # 训练模型
# model.fit(x,y)
# # 预测
# pred_y=model.predict(x)
# # # 评估训练结果误差
# R2=sm.r2_score(y,pred_y)
# k_t_dict['A']=R2
#
#
# # ln(k)-1/t关系图 A*B
# k=[linear_regression_result[0][2][1],linear_regression_result[1][2][1],linear_regression_result[2][2][1]]
# t=[80+273.15,90+273.15,100+273.15]
# t=np.array(t)
# k=np.array(k)
# x=1/t.reshape(-1,1)
# y=np.log(-k)
# # plt.figure("ln(k)-1/t B", facecolor="yellow")
# #
# # # 获取当前坐标轴
# # ax = plt.gca()
# # ax.spines['top'].set_color('none')
# # ax.spines['right'].set_color('none')
# # # 设置刻度定位器
# # maj_loc = eval("plt.AutoLocator()")
# # ax.xaxis.set_major_locator(maj_loc)
# # plt.xlabel("1/t", fontsize=14)
# # plt.ylabel("ln(k)", fontsize=14)
# # plt.xlim(0.0026, 0.0029)
# # plt.ylim(-6.5, -4)
# #
# # plt.title('ln(k)-1/t', fontsize=16)
# # plt.grid(linestyle=":")
# # plt.scatter(x, y, s=70, color="dodgerblue")
#
# model = lm.LinearRegression()
# # 训练模型
# model.fit(x,y)
# # 预测
# pred_y=model.predict(x)
# # # 评估训练结果误差
# R2=sm.r2_score(y,pred_y)
# k_t_dict['A*B']=R2
#
#
#
# ln(k)-1/t关系图 B
# k=[linear_regression_result[0][1][1],linear_regression_result[1][1][1],linear_regression_result[2][1][1]]
# t=[80+273.15,90+273.15,100+273.15]
# t=np.array(t)
# k=np.array(k)
# x=1/t.reshape(-1,1)
# y=np.log(-k)
# # plt.figure("ln(k)-1/t B", facecolor="yellow")
# #
# # # 获取当前坐标轴
# # ax = plt.gca()
# # ax.spines['top'].set_color('none')
# # ax.spines['right'].set_color('none')
# # # 设置刻度定位器
# # maj_loc = eval("plt.AutoLocator()")
# # ax.xaxis.set_major_locator(maj_loc)
# # plt.xlabel("1/t", fontsize=14)
# # plt.ylabel("ln(k)", fontsize=14)
# # plt.xlim(0.0026, 0.0029)
# # plt.ylim(-6.5, -4)
# #
# # plt.title('ln(k)-1/t', fontsize=16)
# # plt.grid(linestyle=":")
# # plt.scatter(x, y, s=70, color="dodgerblue")
#
# model = lm.LinearRegression()
# # 训练模型
# model.fit(x,y)
# # 预测
# pred_y=model.predict(x)
# # # 评估训练结果误差
# R2=sm.r2_score(y,pred_y)
# k_t_dict['B']=R2
# print('斜率',model.coef_[0])
# print('截距',model.intercept_.tolist())
#
#
# # print(k_t_dict)
#
# # ln(b)-1/k is the best.
# # 代入得到
# # r1 = 7.0*10**9*np.exp(-84104/(R*T))*（c of compound B）  # c=7.21138
#
# # 读取数据
# files_B = ['2_100_degrees_Celsius.csv',
#            '2_93_degrees_Celsius.csv',
#            '2_82_degrees_Celsius.csv',
#            '2_76_5_degrees_Celsius.csv']
# files_B_list=[]
# for file in files_B:
#     data=pd.read_csv(file)
#     name=file.split('.')[:-1]
#     files_B_list.append((data,name))
# # print(files_B_list)
#
# linear_regression_result_2=[]
# # 浓度随时间的变化曲线
# for i, item in enumerate(files_B_list):
#     plt.figure("m-t"+str(i), facecolor="yellow")
#
#     # 获取当前坐标轴
#     ax = plt.gca()
#     ax.spines['top'].set_color('none')
#     ax.spines['right'].set_color('none')
#     # plt.xlim(0, max(item[0]['time']))
#     # plt.ylim(0, 7)
#     # 设置刻度定位器
#     maj_loc = eval("plt.AutoLocator()")
#     ax.xaxis.set_major_locator(maj_loc)
#     plt.xlabel("time/min", fontsize=14)
#     plt.ylabel("m/g", fontsize=14)
#
#     plt.title('m-'+item[1][0], fontsize=16)
#     plt.grid(linestyle=":")
#     plt.plot(item[0]['time'],item[0]['D'],color="dodgerblue")
#
#     # 创建线性回归模型
#     model = lm.LinearRegression()
#     # 训练模型
#     x=item[0]['time'].values.reshape(-1,1)
#     y=np.log(item[0]['D']).reshape(-1,1)
#     model.fit(x,y)
#     # 预测
#     pred_y=model.predict(x)
#     # # 评估训练结果误差
#     R2=sm.r2_score(y,pred_y)
#     # print(model.coef_.tolist())
#     # print(model.intercept_.tolist())
#     linear_regression_result_2.append((R2,model.coef_.tolist()[0][0], model.intercept_.tolist()[0]))
#     # print("R2",R2)
#     # print("线性模型的斜率与截距：", model.coef_, model.intercept_)
#     # 绘制回归线
#     plt.figure("ln(m)-t" + str(i), facecolor="yellow")
#     plt.title('ln(m)-'+item[1][0], fontsize=16)
#     plt.xlabel("time/min", fontsize=14)
#     plt.ylabel("ln(m)/g", fontsize=14)
#     plt.plot(x, pred_y, c='orange')
#
# # print(linear_regression_result_2)
# # [(0.9842339097656717, 0.0033703785824381336, -1.2433389275908178),
# # (0.9981815623532668, 0.0030889673753539963, -1.1291749216123506),
# # (0.894821321300925, 0.0009812568864036924, -1.0969809007582907),
# # (0.9769137218955833, 0.00041842662206528314, -1.0955905641595454)]
#
#
# # ln(k)-1/t关系图 D
# t = [100+273.15,93+273.15,82+273.15,76.5+273.15]
# k = [item[1] for item in linear_regression_result_2]
# # 第二组数据误差较大，移除
# t.pop(1)
# k.pop(1)
# t=np.array(t)
# k=np.array(k)
# x=1/t
# y=np.log(k)
#
# plt.figure("ln(k)-1/t D", facecolor="yellow")
#
# # 获取当前坐标轴
# ax = plt.gca()
# ax.spines['top'].set_color('none')
# ax.spines['right'].set_color('none')
# # 设置刻度定位器
# maj_loc = eval("plt.AutoLocator()")
# ax.xaxis.set_major_locator(maj_loc)
# plt.xlabel("1/t", fontsize=14)
# plt.ylabel("ln(k)", fontsize=14)
# plt.xlim(0.00265, 0.0029)
# plt.ylim(-8.0, -5.5)
#
# plt.title('ln(k)-1/t', fontsize=16)
# plt.grid(linestyle=":")
# plt.scatter(x, y, marker='o',s=70, color="dodgerblue")
#
# model = lm.LinearRegression()
# # 训练模型
# model.fit(x.reshape(-1,1),y)
# # 预测
# pred_y=model.predict(x.reshape(-1,1))
# # # 评估训练结果误差
# R2=sm.r2_score(y,pred_y)
# print('R2',R2)
# print('斜率',model.coef_[0])
# print('截距',model.intercept_.tolist())

# 代入得到
# r2 = 3.4*10**10*np.exp(-96068/(R*T))


# Finally

# r1 = 7.0*10**9*np.exp(-84104/(R*T))*(c of compound B)
# r2 = 3.4*10**10*np.exp(-96068/(R*T))  # %

# 三维曲面的绘制
plt.figure("3D Surface", facecolor="lightgreen")

ax3d = plt.gca(projection="3d")
plt.title('3D Surface', fontsize=16)
temp, time=np.meshgrid(np.linspace(50,150,100),np.meshgrid(0,300,300))
z1=7.21138-7.21138*np.exp(-7*10**9*np.exp(-84104/8.314/(273.15+temp))*time)

for te in temp:
    ax3d.plot_surface(temp,time,z1,cstride=20,rstride=20,cmap='jet')

ax3d.set_xlabel("temp")
ax3d.set_ylabel("time")
ax3d.set_zlabel("c")


plt.legend()
plt.show()
