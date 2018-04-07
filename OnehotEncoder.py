# encoding: utf-8

'''

@author: liangchi

@contact: bnu_llc@163.com

@software: pycharm

@file: OnehotEncoder.py

@time: 2018/4/7 下午3:51

'''

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd

'''
LabelEncoder创建标签的整数编码，OneHotEncoder用于创建整数编码值的one hot编码
'''


'''
LabelEncoder能够接收不规则的特征列，并将其转化为从0到n-1的整数值（假设一共有n种不同的类别)
'''
def encoder2label(data, variable):
    le = LabelEncoder()
    new_feature = '{}_label'.format(variable)
    data[new_feature] = le.fit_transform(data[variable])
    return data


'''
onehot能够将数字label变成onehot稀疏化处理
'''
def label2onehot(data, variable):
    onehot_handdle = OneHotEncoder( dtype=np.int32, sparse=False, handle_unknown='error')
    variable_array = data[variable].values.reshape(-1, 1)
    res_array =onehot_handdle.fit_transform(variable_array)
    col_index = [ '{}_onehot_{}'.format(variable, i)  for i in range(res_array.shape[1])]
    one_hot = pd.DataFrame(res_array, columns= col_index )
    res_data = pd.concat([data, one_hot],axis = 1)
    return res_data

if __name__ == '__main__':
    data = pd.read_csv('test.csv')
    #print OneHotEncode(df)
    encoder2label(data,'device')
    print label2onehot(data,'device_label')