import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)

data1 = pd.read_csv('data_code.csv',encoding='gbk')  # 导入data_code.csv文件
data2 = pd.read_csv('data_gp.csv')  # 导入data_gp.csv文件

name = data1['农产品名称映射值']
loc = data1['市场名称映射值']
name = name.to_numpy()
loc = loc.to_numpy()

for k in range(name.size):
    data3 = data2.loc[data2['农产品名称映射值'] == name[k]]  # 从data_gp.csv中找到需要预测的农产品的训练数据
    data3 = data3.loc[data3['市场名称映射值'] == loc[k]]
    data3 = pd.DataFrame.drop_duplicates(data3,subset='数据发布时间',keep='last')  # 按时间去重
    data3 = data3.sort_values(by = '数据发布时间',ascending=True)  # 按日期排序

    data3['数据发布时间'] = pd.to_datetime(data3['数据发布时间'])  # 时间标准化
    x = data3['数据发布时间'].dt.day  # 提取发布时间的日期
    y = data3['平均交易价格']

    def unique_cols(df):  # 判断平均价格是否一直保持不变
        a = df.to_numpy()
        return(a[0] == a[1:]).all(0)

    if unique_cols(y) == 1:  # 如果是，则预测价格也保持不变
        y = y.to_numpy()
        y_pre = y[0]
        print(name[k],'产品预测价格为:',y_pre)
        print('*************************')

    else:  # 不是则进行价格预测
        x = x.astype(float)
        y = y.astype(float)

        x1 = preprocessing.minmax_scale(x)
        y1 = preprocessing.minmax_scale(y)

        x1 = np.array(x1)  # 将时间列转化为一维数组
        y1 = np.array(y1)  # 平均交易价格设为y，并将该列转化为一维数组


        # 划分训练集和测试集
        x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size=0.2)

        # 模型训练
        def sigmoid(x):  # 网络激活函数
            return 1/(1+np.exp(-x))

        yita = 0.15  # 学习速率
        out_in = np.array([0.0,0,0,0,-1])  #输出层的输入
        w_mid = np.zeros([2,4])  #隐层神经元的权值&阈值
        w_out = np.zeros([5])  # 输出层神经元的权值&阈值
        delta_w_out = np.zeros([5])
        delta_w_mid = np.zeros([2,4])
        Err = []
        for j in range(1000):
            error = []
            for it in range(x_train.size):
                net_in = np.append(x_train[it], -1)  # 网络输入
                real = y_train[it]

                for i in range(4):
                    out_in[i] = sigmoid(sum(net_in * w_mid[:, i])) #从输入到隐层的传输过程

                res = sigmoid(sum(out_in * w_out))  # 模型预测值
                error.append(abs(real - res))


                delta_w_out = yita*res*(1-res)*(real-res)*out_in  # 输出层权值的修正量
                delta_w_out[4] = - yita*res*(1-res)*(real-res)  # 输出层阈值的修正量
                w_out = w_out + delta_w_out  # 更新

                for i in range(4):
                    delta_w_mid[:, i] = yita*out_in[i]*(1-out_in[i])*w_out[i]*res*(1-res)*(real-res)*net_in  # 中间层神经元的权值修正量
                    delta_w_mid[1,i] = -yita*out_in[i]*(1-out_in[i])*w_out[i]*res*(1-res)*(real-res)  #中间层神经元的阈值修正量
                w_mid = w_mid + delta_w_mid # 更新
            Err.append(np.mean(error))  # 计算平均误差

        # 代入测试集进行模型评估
        error_te = []
        for it in range(x_test.size):
            net_in = np.append(x_test[it], -1)  # 网络输入
            real = y_test[it]

            for i in range(4):
                out_in[i] = sigmoid(sum(net_in * w_mid[:, i]))  # 从输入到隐层的传输过程

            res = sigmoid(sum(out_in * w_out))  # 模型预测值

            error_te.append(abs(real - res))
        print('模型测试集平均误差为：',np.mean(error_te))

        # 价格预测
        x_true = np.array(range(32))
        x_true = x_true[1:32]
        x_true = preprocessing.minmax_scale(x_true)
        for it in range(31):
            net_in = np.append(x_true[it], -1)  # 网络输入
            for i in range(4):
                out_in[i] = sigmoid(sum(net_in * w_mid[:, i])) #从输入到隐层的传输过程
            res = sigmoid(sum(out_in * w_out))  # 模型预测值
            y_pre = res * (y.max()-y.min())+y.min()  #标准化还原
            print(name[k],'产品在',it+1,'号的预测价格为',y_pre)

        print('*************************')