import os
import json
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def read_data():
    label_label = pd.read_csv('appearance.csv')
    df = pd.DataFrame(label_label)
    data = df.values
    size = data.shape[0]
    # print(size)
    # print(data[0:5])
    beauty = {}
    appearance = {}
    appearance_defect = {}
    affinity = {}
    clothes = {}
    body_shape = {}
    ## 取有子维度得分的数据，保存成字典，键为视频号，值为得分的list，便于查找与做成label
    for i in range(0,size):
        if np.isnan(data[i,1]) != True:
            beauty.update({data[i,0]:[data[i,1]]})
        if np.isnan(data[i,2]) != True:
            appearance.update({data[i,0]:[data[i,2]]})
        if np.isnan(data[i,3]) != True:
            appearance_defect.update({data[i,0]:[data[i,3]]})
        if np.isnan(data[i,4]) != True:
            affinity.update({data[i,0]:[data[i,4]]})
        if np.isnan(data[i,5]) != True:
            clothes.update({data[i,0]:[data[i,5]]})
        if np.isnan(data[i,6]) != True:
            body_shape.update({data[i,0]:[data[i,6]]})
    print('beauty_size: ',len(beauty))
    print(list(beauty.items())[0:5])
    print('appearance_size: ', len(appearance))
    print(list(appearance.items())[0:5])
    print('appearance_defect_size: ', len(appearance_defect))
    print(list(appearance_defect.items())[0:5])
    print('affinity_size: ', len(affinity))
    print(list(affinity.items())[0:5])
    print('clothes_size: ', len(clothes))
    print(list(clothes.items())[0:5])
    print('body_shape_size: ', len(body_shape))
    print(list(body_shape.items())[0:5])
    return beauty, appearance, appearance_defect, affinity, clothes, body_shape

def generate_data_set():
    beauty, appearance, appearance_defect, affinity, clothes, body_shape = read_data()
    load_feature = pd.read_csv('result.csv')
    df = pd.DataFrame(load_feature)
    data = df.values
    size = data.shape[0]
    x_feature = {}
    for i in range(0,size):
        x_feature.update({data[i,0]:data[i,1:].tolist()})
    x_beauty = []
    y_beauty = []
    x_appearance = []
    y_appearance = []
    x_appearance_defect = []
    y_appearance_defect = []
    x_affinity = []
    y_affinity = []
    x_clothes = []
    y_clothes = []
    x_body_shape = []
    y_body_shape = []
    for key in beauty:
        x_beauty.append(x_feature[key])
        y_beauty.append(beauty[key])
    for key in appearance:
        x_appearance.append(x_feature[key])
        y_appearance.append(appearance[key])
    for key in appearance_defect:
        x_appearance_defect.append(x_feature[key])
        y_appearance_defect.append(appearance_defect[key])
    for key in affinity:
        x_affinity.append(x_feature[key])
        y_affinity.append(affinity[key])
    for key in clothes:
        x_clothes.append(x_feature[key])
        y_clothes.append(clothes[key])
    for key in body_shape:
        x_body_shape.append(x_feature[key])
        y_body_shape.append(body_shape[key])
    x_beauty_train, x_beauty_test, y_beauty_train, y_beauty_test = \
        train_test_split(x_beauty, y_beauty, test_size=0.25, random_state=1)
    x_appearance_train, x_appearance_test, y_appearance_train, y_appearance_test = \
        train_test_split(x_appearance, y_appearance, test_size=0.25, random_state=1)
    x_appearance_defect_train, x_appearance_defect_test, y_appearance_defect_train, y_appearance_defect_test = \
        train_test_split(x_appearance_defect, y_appearance_defect, test_size=0.25, random_state=1)
    x_affinity_train, x_affinity_test, y_affinity_train, y_affinity_test = \
        train_test_split(x_affinity, y_affinity, test_size=0.25, random_state=1)
    x_clothes_train, x_clothes_test, y_clothes_train, y_clothes_test = \
        train_test_split(x_clothes, y_clothes, test_size=0.25, random_state=1)
    x_body_shape_train, x_body_shape_test, y_body_shape_train, y_body_shape_test = \
        train_test_split(x_body_shape, y_body_shape, test_size=0.25, random_state=1)
    return x_beauty_train, x_beauty_test, y_beauty_train, y_beauty_test, \
           x_appearance_train, x_appearance_test, y_appearance_train, y_appearance_test,\
           x_appearance_defect_train, x_appearance_defect_test, y_appearance_defect_train, y_appearance_defect_test,\
           x_affinity_train, x_affinity_test, y_affinity_train, y_affinity_test,\
           x_clothes_train, x_clothes_test, y_clothes_train, y_clothes_test,\
           x_body_shape_train, x_body_shape_test, y_body_shape_train, y_body_shape_test

def regression():
    x_beauty_train, x_beauty_test, y_beauty_train, y_beauty_test, \
    x_appearance_train, x_appearance_test, y_appearance_train, y_appearance_test, \
    x_appearance_defect_train, x_appearance_defect_test, y_appearance_defect_train, y_appearance_defect_test, \
    x_affinity_train, x_affinity_test, y_affinity_train, y_affinity_test, \
    x_clothes_train, x_clothes_test, y_clothes_train, y_clothes_test, \
    x_body_shape_train, x_body_shape_test, y_body_shape_train, y_body_shape_test = generate_data_set()
    ## 形象气质
    lr_beauty = linear_model.LinearRegression()
    lr_beauty.fit(x_beauty_train,y_beauty_train)
    print('lr_beauty_coef: ',lr_beauty.coef_)
    print('lr_beauty_intercept: ', lr_beauty.intercept_)
    y_beauty_pred = lr_beauty.predict(x_beauty_test)
    print('mean_absolute_error: ',mean_absolute_error(y_beauty_test,y_beauty_pred))
    print('mean_squared_error: ',mean_squared_error(y_beauty_test,y_beauty_pred))
    print('r2_score: ',r2_score(y_beauty_test,y_beauty_pred))
    ## 仪容仪表
    lr_appearance = linear_model.LinearRegression()
    lr_appearance.fit(x_appearance_train,y_appearance_train)
    print('lr_appearance_coef: ',lr_appearance.coef_)
    print('lr_appearance_intercept: ', lr_appearance.intercept_)
    y_appearance_pred = lr_appearance.predict(x_appearance_test)
    print('mean_absolute_error: ',mean_absolute_error(y_appearance_test,y_appearance_pred))
    print('mean_squared_error: ',mean_squared_error(y_appearance_test,y_appearance_pred))
    print('r2_score: ',r2_score(y_appearance_test,y_appearance_pred))
    ## 形象缺陷
    lr_appearance_defect = linear_model.LinearRegression()
    lr_appearance_defect.fit(x_appearance_defect_train,y_appearance_defect_train)
    print('lr_appearance_defect_coef: ',lr_appearance_defect.coef_)
    print('lr_appearance_defect_intercept: ', lr_appearance_defect.intercept_)
    y_appearance_defect_pred = lr_beauty.predict(x_appearance_defect_test)
    print('mean_absolute_error: ',mean_absolute_error(y_appearance_defect_test,y_appearance_defect_pred))
    print('mean_squared_error: ',mean_squared_error(y_appearance_defect_test,y_appearance_defect_pred))
    print('r2_score: ',r2_score(y_appearance_defect_test,y_appearance_defect_pred))
    ## 形象亲和
    lr_affinity = linear_model.LinearRegression()
    lr_affinity.fit(x_affinity_train,y_affinity_train)
    print('lr_affinity_coef: ',lr_affinity.coef_)
    print('lr_affinity_intercept: ', lr_affinity.intercept_)
    y_affinity_pred = lr_affinity.predict(x_affinity_test)
    print('mean_absolute_error: ',mean_absolute_error(y_affinity_test,y_affinity_pred))
    print('mean_squared_error: ',mean_squared_error(y_affinity_test,y_affinity_pred))
    print('r2_score: ',r2_score(y_affinity_test,y_affinity_pred))
    ## 面试着装
    lr_clothes = linear_model.LinearRegression()
    lr_clothes.fit(x_clothes_train,y_clothes_train)
    print('lr_clothes_coef: ',lr_clothes.coef_)
    print('lr_clothes_intercept: ', lr_clothes.intercept_)
    y_clothes_pred = lr_clothes.predict(x_clothes_test)
    print('mean_absolute_error: ',mean_absolute_error(y_clothes_test,y_clothes_pred))
    print('mean_squared_error: ',mean_squared_error(y_clothes_test,y_clothes_pred))
    print('r2_score: ',r2_score(y_clothes_test,y_clothes_pred))
    ## 身材外形
    lr_body_shape = linear_model.LinearRegression()
    lr_body_shape.fit(x_body_shape_train,y_body_shape_train)
    print('lr_body_shape_coef: ',lr_body_shape.coef_)
    print('lr_body_shape_intercept: ', lr_body_shape.intercept_)
    y_body_shape_pred = lr_body_shape.predict(x_body_shape_test)
    print('mean_absolute_error: ',mean_absolute_error(y_body_shape_test,y_body_shape_pred))
    print('mean_squared_error: ',mean_squared_error(y_body_shape_test,y_body_shape_pred))
    print('r2_score: ',r2_score(y_body_shape_test,y_body_shape_pred))


if __name__ == '__main__':
    # read_data()
    generate_data_set()