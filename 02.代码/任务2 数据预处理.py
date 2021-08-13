import pandas as pd
import numpy as np
from pandas import to_datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# 读取farming2.csv及product_market.csv文件
data=pd.read_csv('C:\\Users\\LYF\\Desktop\\farming2.csv',encoding='gbk')
data2=pd.read_csv('C:\\Users\\LYF\\Desktop\\product_market.csv')
# 查看数据长度
data.shape
# 去除重复行
data.drop_duplicates(subset=None, keep='first', inplace=True)
# 查看去重后数据长度
data.shape
# 删除无关行
data.drop(data.columns[[0,2,4,5,6,7,11]],axis=1,inplace=True)
# 查看farming2.csv文件农产品映射值和市场名称映射值组合种类
data_code=data.drop_duplicates(subset=data.columns[[0,1]], keep='first', inplace=False)
data_code.shape
# 查看product_market.csv文件农产品映射值和市场名称映射值组合种类
data2_code=data2.drop_duplicates(subset=['农产品名称映射值','市场名称映射值'], keep='first', inplace=False)
data2_code.shape
#查找并导出相同农产品映射值
  #合并2个表中的组合种类
datacd_sum=pd.concat([data_code,data2_code])
  #保留了第一次出现的相同项和不同项
datacd_same=datacd_sum.drop_duplicates(subset=['农产品名称映射值','市场名称映射值'],keep='first', inplace=False)                    
datacd_same.shape
#保留了不同项
datacd_norepeat=datacd_sum.drop_duplicates(subset=['农产品名称映射值','市场名称映射值'],keep=False, inplace=False)                                          
datacd_norepeat.shape
#相同＋2*不同
datacd_sum2=pd.concat([datacd_same,datacd_norepeat])
#相同＋2*不同-2*不同
data_cd=datacd_sum2.drop_duplicates(subset=['农产品名称映射值','市场名称映射值'],keep=False, inplace=False)
data_cd.to_csv('C:\\Users\\LYF\\Desktop\\data_code.csv')
#对多个同一时间的同一商品价格取平均，以平均值代替多组数据
data_copy=data
data_copy.groupby(['市场名称映射值','农产品名称映射值','数据发布时间'])[['平均交易价格','最高交易价格','最低交易价格']].mean()
data_group=data_copy.groupby(['市场名称映射值','农产品名称映射值','数据发布时间'])[['平均交易价格','最高交易价格','最低交易价格']].mean()
#平铺数据
data_group=data_group.reset_index()   
#测试是否替换成功
data.drop_duplicates(subset=['市场名称映射值','数据发布时间','农产品名称映射值'],keep='first',inplace=True)
data.shape
#若平均值为0，最高最低值不为0，将最高最低值取平均后填补平均值
#选取数据取平均
data_pj=data_group.loc[(data_group['平均交易价格'] == 0)& (data_group['最高交易价格']!= 0)&(data_group['最低交易价格']!= 0)]
data_pj['平均交易价格']=(data_pj['最高交易价格']+data_pj['最低交易价格'])/2
data_pj
#与原数据合并，删除未填补的数据
data_gpsum=pd.concat([data_pj,data_group])
data_gp=data_gpsum.drop_duplicates(subset=['市场名称映射值','数据发布时间','农产品名称映射值'],keep='first')
data_gp
#测试是否成功填充
data_pjtext=data_gp.loc[(data_gp['平均交易价格'] == 0)& (data_gp['最高交易价格']!= 0)&(data_gp['最低交易价格']!= 0)]
data_pjtext.shape
#时间标准化
data['数据发布时间']=pd.to_datetime(data['数据发布时间'],format="%Y/%m/%d ")
#检测是否标准化成功
data.dtypes
