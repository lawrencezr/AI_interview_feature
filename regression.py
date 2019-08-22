import os
import json
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model, svm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.externals import joblib
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, scale


def preprocess(x):
    return 0.05 * x


# def merge_list(a,b):
#     a.append(b[-1])
#     return a

beauty_size = 0
appearance_size = 0
appearance_defect_size = 0
affinity_size = 0
clothes_size = 0
body_shape_size = 0


def read_data():
    label_label = pd.read_csv('train/appearance.csv')
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
    for i in range(0, size):
        if np.isnan(data[i, 1]) != True:
            beauty.update({data[i, 0]: [data[i, 1]]})
        if np.isnan(data[i, 2]) != True:
            appearance.update({data[i, 0]: [data[i, 2]]})
        if np.isnan(data[i, 3]) != True:
            appearance_defect.update({data[i, 0]: [data[i, 3]]})
        if np.isnan(data[i, 4]) != True:
            affinity.update({data[i, 0]: [data[i, 4]]})
        if np.isnan(data[i, 5]) != True:
            clothes.update({data[i, 0]: [data[i, 5]]})
        if np.isnan(data[i, 6]) != True:
            body_shape.update({data[i, 0]: [data[i, 6]]})

    print('beauty_size: ', len(beauty))
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
    load_feature = pd.read_csv('train/result.csv')
    print(np.isnan(load_feature).any() == False)
    df = pd.DataFrame(load_feature)
    data = df.values
    size = data.shape[0]
    x_feature = {}
    for i in range(0, size):
        x_feature.update({data[i, 0]: data[i, 1:].tolist()})
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
        x = list(map(preprocess,x_feature[key]))
        tmp = []
        tmp.extend(x[:1])  # 颜值
        tmp.extend(x[4:])  # 面色、微笑程度、正装
        # x = merge_list(x,x_feature[key])
        # x_beauty.append(x_feature[key])
        x_beauty.append(tmp)
        y_beauty.append(beauty[key])
    for key in appearance:
        x = list(map(preprocess,x_feature[key]))
        # x = merge_list(x,x_feature[key])
        tmp = []
        tmp.extend(x[:2])  # 颜值、黑眼圈
        tmp.extend(x[3:4])  # 痤疮
        tmp.extend(x[5:6])  # 正装
        x_appearance.append(tmp)
        y_appearance.append(appearance[key])
    for key in appearance_defect:
        x = list(map(preprocess,x_feature[key]))
        # x = merge_list(x,x_feature[key])
        tmp = []
        tmp.extend(x[:1])  # 颜值
        tmp.extend(x[2:3])  # 色斑
        x_appearance_defect.append(tmp)
        y_appearance_defect.append(appearance_defect[key])
    for key in affinity:
        x = list(map(preprocess,x_feature[key]))
        # x = merge_list(x,x_feature[key])
        # tmp = []
        tmp = []
        tmp.extend(x[:1])
        tmp.extend(x[-1:])
        # tmp.extend(x_feature[key][2:4])  # 色斑、痤疮
        x_affinity.append(tmp)
        y_affinity.append(affinity[key])
    for key in clothes:
        x = list(map(preprocess,x_feature[key]))
        # x = merge_list(x,x_feature[key])
        tmp = []
        tmp.extend(x[2:3])  # 色斑
        tmp.extend(x[5:6])  # 正装
        x_clothes.append(tmp)
        y_clothes.append(clothes[key])
    for key in body_shape:
        # x = list(map(preprocess,x_feature[key]))
        # x = merge_list(x,x_feature[key])
        x_body_shape.append(x_feature[key])
        y_body_shape.append(body_shape[key])
    x_beauty_train, x_beauty_test, y_beauty_train, y_beauty_test = \
        train_test_split(x_beauty, y_beauty, test_size=0.25, random_state=1)
    print(x_beauty[0:5])
    print(y_beauty[0:5])
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
           x_appearance_train, x_appearance_test, y_appearance_train, y_appearance_test, \
           x_appearance_defect_train, x_appearance_defect_test, y_appearance_defect_train, y_appearance_defect_test, \
           x_affinity_train, x_affinity_test, y_affinity_train, y_affinity_test, \
           x_clothes_train, x_clothes_test, y_clothes_train, y_clothes_test, \
           x_body_shape_train, x_body_shape_test, y_body_shape_train, y_body_shape_test


def relation():
    beauty, appearance, appearance_defect, affinity, clothes, body_shape = read_data()
    load_feature = pd.read_csv('train/result.csv')
    print(np.isnan(load_feature).any() == False)
    df = pd.DataFrame(load_feature)
    data = df.values
    size = data.shape[0]
    x_feature = {}
    for i in range(0, size):
        x_feature.update({data[i, 0]: data[i, 1:].tolist()})
    x_beauty = []
    y_appearance = []
    y_beauty = []
    x_appearance = []
    x_appearance_defect = []
    y_appearance_defect = []
    x_affinity = []
    y_affinity = []
    x_clothes = []
    y_clothes = []
    x_body_shape = []
    y_body_shape = []
    for key in beauty:
        # x = list(map(preprocess,x_feature[key][:-1]))
        # x = merge_list(x,x_feature[key])
        x_beauty.append(x_feature[key])
        y_beauty.append(beauty[key])
    for key in appearance:
        # x = list(map(preprocess,x_feature[key][:-1]))
        # x = merge_list(x,x_feature[key])
        x_appearance.append(x_feature[key])
        y_appearance.append(appearance[key])
    for key in appearance_defect:
        # x = list(map(preprocess,x_feature[key][:-1]))
        # x = merge_list(x,x_feature[key])
        x_appearance_defect.append(x_feature[key])
        y_appearance_defect.append(appearance_defect[key])
    for key in affinity:
        # x = list(map(preprocess,x_feature[key][:-1]))
        # x = merge_list(x,x_feature[key])
        x_affinity.append(x_feature[key])
        y_affinity.append(affinity[key])
    for key in clothes:
        # x = list(map(preprocess,x_feature[key][:-1]))
        # x = merge_list(x,x_feature[key])
        x_clothes.append(x_feature[key])
        y_clothes.append(clothes[key])
    for key in body_shape:
        # x = list(map(preprocess,x_feature[key][:-1]))
        # x = merge_list(x,x_feature[key])
        x_body_shape.append(x_feature[key])
        y_body_shape.append(body_shape[key])
    x_beauty = scale(x_beauty)
    x_appearance = scale(x_appearance)
    x_appearance_defect = scale(x_appearance_defect)
    x_affinity = scale(x_affinity)
    x_clothes = scale(x_clothes)
    # x_body_shape = scale(x_body_shape)
    df_beauty = pd.DataFrame({'score': [y_beauty[i][0] for i in range(0, len(y_beauty))],
                              'beauty': [x_beauty[i][0] for i in range(0, len(x_beauty))],
                              'darkCircle': [x_beauty[i][1] for i in range(0, len(x_beauty))],
                              'stain': [x_beauty[i][2] for i in range(0, len(x_beauty))],
                              'acne': [x_beauty[i][3] for i in range(0, len(x_beauty))],
                              'health': [x_beauty[i][4] for i in range(0, len(x_beauty))],
                              'suit': [x_beauty[i][5] for i in range(0, len(x_beauty))],
                              'smile': [x_beauty[i][6] for i in range(0, len(x_beauty))]})
    df_appearance = pd.DataFrame({'score': [y_appearance[i][0] for i in range(0, len(y_appearance))],
                                  'beauty': [x_appearance[i][0] for i in range(0, len(x_appearance))],
                                  'darkCircle': [x_appearance[i][1] for i in range(0, len(x_appearance))],
                                  'stain': [x_appearance[i][2] for i in range(0, len(x_appearance))],
                                  'acne': [x_appearance[i][3] for i in range(0, len(x_appearance))],
                                  'health': [x_appearance[i][4] for i in range(0, len(x_appearance))],
                                  'suit': [x_appearance[i][5] for i in range(0, len(x_appearance))],
                                  'smile': [x_beauty[i][6] for i in range(0, len(x_appearance))]})
    df_appearance_defect = pd.DataFrame(
        {'score': [y_appearance_defect[i][0] for i in range(0, len(y_appearance_defect))],
         'beauty': [x_appearance_defect[i][0] for i in range(0, len(x_appearance_defect))],
         'darkCircle': [x_appearance_defect[i][1] for i in range(0, len(x_appearance_defect))],
         'stain': [x_appearance_defect[i][2] for i in range(0, len(x_appearance_defect))],
         'acne': [x_appearance_defect[i][3] for i in range(0, len(x_appearance_defect))],
         'health': [x_appearance_defect[i][4] for i in range(0, len(x_appearance_defect))],
         'suit': [x_appearance_defect[i][5] for i in range(0, len(x_appearance_defect))],
         'smile': [x_beauty[i][6] for i in range(0, len(x_appearance_defect))]})
    df_affinity = pd.DataFrame({'score': [y_affinity[i][0] for i in range(0, len(y_affinity))],
                                'beauty': [x_affinity[i][0] for i in range(0, len(x_affinity))],
                                'darkCircle': [x_affinity[i][1] for i in range(0, len(x_affinity))],
                                'stain': [x_affinity[i][2] for i in range(0, len(x_affinity))],
                                'acne': [x_affinity[i][3] for i in range(0, len(x_affinity))],
                                'health': [x_affinity[i][4] for i in range(0, len(x_affinity))],
                                'suit': [x_affinity[i][5] for i in range(0, len(x_affinity))],
                                'smile': [x_beauty[i][6] for i in range(0, len(x_affinity))]})
    df_clothes = pd.DataFrame({'score': [y_clothes[i][0] for i in range(0, len(y_clothes))],
                               'beauty': [x_clothes[i][0] for i in range(0, len(x_clothes))],
                               'darkCircle': [x_clothes[i][1] for i in range(0, len(x_clothes))],
                               'stain': [x_clothes[i][2] for i in range(0, len(x_clothes))],
                               'acne': [x_clothes[i][3] for i in range(0, len(x_clothes))],
                               'health': [x_clothes[i][4] for i in range(0, len(x_clothes))],
                               'suit': [x_clothes[i][5] for i in range(0, len(x_clothes))],
                               'smile': [x_beauty[i][6] for i in range(0, len(x_clothes))]})
    df_feature = pd.DataFrame({'beauty': [x_clothes[i][0] for i in range(0, len(x_clothes))],
                               'darkCircle': [x_clothes[i][1] for i in range(0, len(x_clothes))],
                               'stain': [x_clothes[i][2] for i in range(0, len(x_clothes))],
                               'acne': [x_clothes[i][3] for i in range(0, len(x_clothes))],
                               'health': [x_clothes[i][4] for i in range(0, len(x_clothes))],
                               'suit': [x_clothes[i][5] for i in range(0, len(x_clothes))],
                               'smile': [x_beauty[i][6] for i in range(0, len(x_clothes))]})
    print('------------spearman_beauty-------------')
    print(df_beauty.corr('spearman'))
    print('------------spearman_appearance-------------')
    print(df_appearance.corr('spearman'))
    print('------------spearman_appearance_defect-------------')
    print(df_appearance_defect.corr('spearman'))
    print('------------spearman_affinity-------------')
    print(df_affinity.corr('spearman'))
    print('------------spearman_clothes-------------')
    print(df_clothes.corr('spearman'))
    print('------------spearman_feature-------------')
    print(df_feature.corr('spearman'))
    print('------------pearson_feature-------------')
    print(df_feature.corr('pearson'))
    print('------------kendall_feature-------------')
    print(df_feature.corr('kendall'))
    # print(df_beauty.corr())
    # print(df_beauty.corr('kendall'))


def regression():
    x_beauty_train, x_beauty_test, y_beauty_train, y_beauty_test, \
    x_appearance_train, x_appearance_test, y_appearance_train, y_appearance_test, \
    x_appearance_defect_train, x_appearance_defect_test, y_appearance_defect_train, y_appearance_defect_test, \
    x_affinity_train, x_affinity_test, y_affinity_train, y_affinity_test, \
    x_clothes_train, x_clothes_test, y_clothes_train, y_clothes_test, \
    x_body_shape_train, x_body_shape_test, y_body_shape_train, y_body_shape_test = generate_data_set()
    print(x_beauty_test)
    ## 多项式线性回归效果更差
    quadratic = PolynomialFeatures(degree=3)
    x_beauty_train = quadratic.fit_transform(x_beauty_train)
    x_appearance_train = quadratic.fit_transform(x_appearance_train)
    x_appearance_defect_train = quadratic.fit_transform(x_appearance_defect_train)
    x_affinity_train = quadratic.fit_transform(x_affinity_train)
    x_clothes_train = quadratic.fit_transform(x_clothes_train)
    x_body_shape_train = quadratic.fit_transform(x_body_shape_train)
    x_beauty_test = quadratic.fit_transform(x_beauty_test)
    x_appearance_test = quadratic.fit_transform(x_appearance_test)
    x_appearance_defect_test = quadratic.fit_transform(x_appearance_defect_test)
    x_affinity_test = quadratic.fit_transform(x_affinity_test)
    x_clothes_test = quadratic.fit_transform(x_clothes_test)
    x_body_shape_test = quadratic.fit_transform(x_body_shape_test)
    n_estimators = 10
    max_sample = 0.5
    max_features = 0.5
    bootstrap = False
    bootstrap_features = False
    # x_beauty_train = np.array(x_beauty_train)
    # x_beauty_test= np.array(x_beauty_test)
    # y_beauty_train= np.array(y_beauty_train)
    # y_beauty_test= np.array(y_beauty_test)
    ## 形象气质
    print('-------------形象气质-------------')
    lr_beauty = linear_model.Ridge()
    # lr_beauty = BaggingRegressor(RandomForestRegressor(),n_estimators=n_estimators,max_samples=max_sample,max_features=max_features,
    #                              bootstrap=bootstrap,bootstrap_features=bootstrap_features,random_state=1,verbose=0)
    # lr_beauty = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    # x_beauty_train = scale(x_beauty_train)
    # x_beauty_test = scale(x_beauty_test)
    # x_beauty_train = list(map(lambda x: x / 20.0, x_beauty_train))
    # x_beauty_test = list(map(lambda x: x / 20.0, x_beauty_test))
    # print(x_beauty_train)
    lr_beauty.fit(x_beauty_train, y_beauty_train)
    joblib.dump(lr_beauty, 'model/lr_beauty_feature.m')
    print('lr_beauty_coef: ', lr_beauty.coef_)
    print('lr_beauty_intercept: ', lr_beauty.intercept_)
    y_beauty_train_pred = lr_beauty.predict(x_beauty_train)
    y_beauty_test_pred = lr_beauty.predict(x_beauty_test)

    print('mean_absolute_error trian: {:.3f} test: {:.3f}'.
          format(mean_absolute_error(y_beauty_train, y_beauty_train_pred),
                 mean_absolute_error(y_beauty_test, y_beauty_test_pred)))
    print('mean_squared_error trian: {:.3f} test: {:.3f}'.
          format(mean_squared_error(y_beauty_train, y_beauty_train_pred),
                 mean_squared_error(y_beauty_test, y_beauty_test_pred)))
    print('r2_score: trian: {:.3f} test: {:.3f}'.
          format(r2_score(y_beauty_train, y_beauty_train_pred),
                 r2_score(y_beauty_test, y_beauty_test_pred)))
    print('score: {:.3f}'.format(lr_beauty.score(x_beauty_test, y_beauty_test)))
    ## 仪容仪表
    print('-------------仪容仪表-------------')
    lr_appearance = linear_model.LinearRegression()
    # lr_appearance = BaggingRegressor(RandomForestRegressor(),n_estimators=n_estimators,max_samples=max_sample,max_features=max_features,
    #                              bootstrap=bootstrap,bootstrap_features=bootstrap_features,random_state=1,verbose=1)
    # lr_appearance = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    # x_appearance_train = scale(x_appearance_train)
    # x_appearance_test = scale(x_appearance_test)
    lr_appearance.fit(x_appearance_train, y_appearance_train)
    joblib.dump(lr_appearance, 'model/lr_appearance_feature.m')
    print('lr_appearance_coef: ', lr_appearance.coef_)
    print('lr_appearance_intercept: ', lr_appearance.intercept_)
    y_appearance_train_pred = lr_appearance.predict(x_appearance_train)
    y_appearance_test_pred = lr_appearance.predict(x_appearance_test)
    print('mean_absolute_error trian: {:.3f} test: {:.3f}'.
          format(mean_absolute_error(y_appearance_train, y_appearance_train_pred),
                 mean_absolute_error(y_appearance_test, y_appearance_test_pred)))
    print('mean_squared_error trian: {:.3f} test: {:.3f}'.
          format(mean_squared_error(y_appearance_train, y_appearance_train_pred),
                 mean_squared_error(y_appearance_test, y_appearance_test_pred)))
    print('r2_score trian: {:.3f} test: {:.3f}'.
          format(r2_score(y_appearance_train, y_appearance_train_pred),
                 r2_score(y_appearance_test, y_appearance_test_pred)))
    print('score: {:.3f}'.format(lr_appearance.score(x_appearance_test, y_appearance_test)))
    ## 形象缺陷
    print('-------------形象缺陷-------------')
    lr_appearance_defect = linear_model.LinearRegression()
    # lr_appearance_defect = BaggingRegressor(RandomForestRegressor(),n_estimators=n_estimators,max_samples=max_sample,max_features=max_features,
    #                              bootstrap=bootstrap,bootstrap_features=bootstrap_features,random_state=1,verbose=1)
    # lr_appearance_defect = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    # x_appearance_defect_train = scale(x_appearance_defect_train)
    # x_appearance_defect_test = scale(x_appearance_defect_test)
    lr_appearance_defect.fit(x_appearance_defect_train, y_appearance_defect_train)
    joblib.dump(lr_appearance_defect, 'model/lr_appearance_defect_feature.m')
    print('lr_appearance_defect_coef: ', lr_appearance_defect.coef_)
    print('lr_appearance_defect_intercept: ', lr_appearance_defect.intercept_)
    y_appearance_defect_train_pred = lr_appearance_defect.predict(x_appearance_defect_train)
    y_appearance_defect_test_pred = lr_appearance_defect.predict(x_appearance_defect_test)
    print('mean_absolute_error trian: {:.3f} test: {:.3f}'.
          format(mean_absolute_error(y_appearance_defect_train, y_appearance_defect_train_pred),
                 mean_absolute_error(y_appearance_defect_test, y_appearance_defect_test_pred)))
    print('mean_squared_error trian: {:.3f} test: {:.3f}'.
          format(mean_squared_error(y_appearance_defect_train, y_appearance_defect_train_pred),
                 mean_squared_error(y_appearance_defect_test, y_appearance_defect_test_pred)))
    print('r2_score trian: {:.3f} test: {:.3f}'.
          format(r2_score(y_appearance_defect_train, y_appearance_defect_train_pred),
                 r2_score(y_appearance_defect_test, y_appearance_defect_test_pred)))
    print('score: {:.3f}'.format(lr_appearance_defect.score(x_appearance_defect_test, y_appearance_defect_test)))
    ## 形象亲和
    print('-------------形象亲和-------------')
    lr_affinity = linear_model.Ridge()
    # lr_affinity = BaggingRegressor(RandomForestRegressor(), n_estimators=n_estimators, max_samples=max_sample,
    #                                         max_features=max_features,
    #                                         bootstrap=bootstrap, bootstrap_features=bootstrap_features, random_state=1,
    #                                         verbose=1)
    # lr_affinity = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    # x_affinity_train = scale(x_affinity_train)
    # x_affinity_test = scale(x_affinity_test)
    # x_affinity_train = list(map(lambda x: x / 20.0, x_affinity_train))
    # x_affinity_test = list(map(lambda x: x / 20.0, x_affinity_test))
    lr_affinity.fit(x_affinity_train, y_affinity_train)
    joblib.dump(lr_affinity, 'model/lr_affinity_feature.m')
    print('lr_affinity_coef: ', lr_affinity.coef_)
    print('lr_affinity_intercept: ', lr_affinity.intercept_)
    y_affinity_train_pred = lr_affinity.predict(x_affinity_train)
    y_affinity_test_pred = lr_affinity.predict(x_affinity_test)
    print('mean_absolute_error trian: {:.3f} test: {:.3f}'.
          format(mean_absolute_error(y_affinity_train, y_affinity_train_pred),
                 mean_absolute_error(y_affinity_test, y_affinity_test_pred)))
    print('mean_squared_error trian: {:.3f} test: {:.3f}'.
          format(mean_squared_error(y_affinity_train, y_affinity_train_pred),
                 mean_squared_error(y_affinity_test, y_affinity_test_pred)))
    print('r2_score trian: {:.3f} test: {:.3f}'.
          format(r2_score(y_affinity_train, y_affinity_train_pred),
                 r2_score(y_affinity_test, y_affinity_test_pred)))
    print('score: {:.3f}'.format(lr_affinity.score(x_affinity_test, y_affinity_test)))
    ## 面试着装
    print('-------------面试着装-------------')
    lr_clothes = linear_model.Ridge()
    # lr_clothes = BaggingRegressor(RandomForestRegressor(), n_estimators=n_estimators, max_samples=max_sample,
    #                                         max_features=max_features,
    #                                         bootstrap=bootstrap, bootstrap_features=bootstrap_features, random_state=1,
    #                                         verbose=1)
    # lr_clothes = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    # x_clothes_train = scale(x_clothes_train)
    # x_clothes_test = scale(x_clothes_test)
    # x_clothes_train = list(map(lambda x:x / 20.0, x_clothes_train))
    # x_clothes_test = list(map(lambda x:x / 20.0, x_clothes_test))
    lr_clothes.fit(x_clothes_train, y_clothes_train)
    joblib.dump(lr_clothes, 'model/lr_clothes_feature.m')
    print('lr_clothes_coef: ', lr_clothes.coef_)
    print('lr_clothes_intercept: ', lr_clothes.intercept_)
    y_clothes_train_pred = lr_clothes.predict(x_clothes_train)
    y_clothes_test_pred = lr_clothes.predict(x_clothes_test)
    print('mean_absolute_error trian: {:.3f} test: {:.3f}'.
          format(mean_absolute_error(y_clothes_train, y_clothes_train_pred),
                 mean_absolute_error(y_clothes_test, y_clothes_test_pred)))
    print('mean_squared_error trian: {:.3f} test: {:.3f}'.
          format(mean_squared_error(y_clothes_train, y_clothes_train_pred),
                 mean_squared_error(y_clothes_test, y_clothes_test_pred)))
    print('r2_score trian: {:.3f} test: {:.3f}'.
          format(r2_score(y_clothes_train, y_clothes_train_pred),
                 r2_score(y_clothes_test, y_clothes_test_pred)))
    print('score: {:.3f}'.format(lr_clothes.score(x_clothes_test, y_clothes_test)))
    ## 身材外形
    print('-------------身材外形-------------')
    lr_body_shape = linear_model.LinearRegression()
    # lr_body_shape = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
    lr_body_shape.fit(x_body_shape_train, y_body_shape_train)
    joblib.dump(lr_body_shape, 'model/lr_body_shape.m')
    print('lr_body_shape_coef: ', lr_body_shape.coef_)
    print('lr_body_shape_intercept: ', lr_body_shape.intercept_)
    y_body_shape_train_pred = lr_body_shape.predict(x_body_shape_train)
    y_body_shape_test_pred = lr_body_shape.predict(x_body_shape_test)
    print('mean_absolute_error trian: {:.3f} test: {:.3f}'.
          format(mean_absolute_error(y_body_shape_train, y_body_shape_train_pred),
                 mean_absolute_error(y_body_shape_test, y_body_shape_test_pred)))
    print('mean_squared_error trian: {:.3f} test: {:.3f}'.
          format(mean_squared_error(y_body_shape_train, y_body_shape_train_pred),
                 mean_squared_error(y_body_shape_test, y_body_shape_test_pred)))
    print('r2_score trian: {:.3f} test: {:.3f}'.
          format(r2_score(y_body_shape_train, y_body_shape_train_pred),
                 r2_score(y_body_shape_test, y_body_shape_test_pred)))
    print('score: {:.3f}'.format(lr_body_shape.score(x_body_shape_test, y_body_shape_test)))
    ## 分布分析
    beauty_count = 0
    appearance_count = 0
    appearance_defect_count = 0
    affinity_count = 0
    clothes_count = 0
    beauty_error = []
    appearance_error = []
    apearance_defect_error = []
    affinity_error = []
    clothes_error = []
    global beauty_size
    beauty_size = len(y_beauty_test)
    global appearance_size
    appearance_size = len(y_appearance_test)
    global appearance_defect_size
    appearance_defect_size = len(y_appearance_defect_test)
    global affinity_size
    affinity_size = len(y_affinity_test)
    global clothes_size
    clothes_size = len(y_clothes_test)
    # global body_shape_size
    # body_shape_size = len(body_shape)
    for i in range(len(y_beauty_test)):
        error = y_beauty_test[i] - y_beauty_test_pred[i]
        beauty_error.append(error)
        if abs(error) > 1:
            beauty_count += 1
    for i in range(len(y_appearance_test)):
        error = y_appearance_test[i] - y_appearance_test_pred[i]
        appearance_error.append(error)
        if abs(error) > 1:
            appearance_count += 1
    for i in range(len(y_appearance_defect_test)):
        error = y_appearance_defect_test[i] - y_appearance_defect_test_pred[i]
        apearance_defect_error.append(error)
        if abs(error) > 1:
            appearance_defect_count += 1
    for i in range(len(y_affinity_test)):
        error = y_affinity_test[i] - y_affinity_test_pred[i]
        affinity_error.append(error)
        if abs(error) > 1:
            affinity_count += 1
    for i in range(len(y_clothes_test)):
        error = y_clothes_test[i] - y_clothes_test_pred[i]
        clothes_error.append(error)
        if abs(error) > 1:
            clothes_count += 1
    dimensions = ['形象气质', '仪容仪表', '形象缺陷', '形象亲和', '面试着装']
    loss_rate = [beauty_count / beauty_size, appearance_count / appearance_size,
                 appearance_defect_count / appearance_defect_size, affinity_count / affinity_size,
                 clothes_count / clothes_size]
    df_beauty = pd.DataFrame(
        {'beauty_pred': np.array(y_beauty_test_pred).ravel(), 'beauty_true': np.array(y_beauty_test).ravel()})
    df_appearance = pd.DataFrame({'appearance_pred': np.array(y_appearance_test_pred).ravel(),
                                  'appearance_true': np.array(y_appearance_test).ravel()})
    df_appearance_defect = pd.DataFrame({'appearance_defect_pred': np.array(y_appearance_defect_test_pred).ravel(),
                                         'appearance_defect_test': np.array(y_appearance_defect_test).ravel()})
    df_affinity = pd.DataFrame(
        {'affinity_pred': np.array(y_affinity_test_pred).ravel(), 'y_affinity_test': np.array(y_affinity_test).ravel()})
    df_clothes = pd.DataFrame(
        {'clothes_pred': np.array(y_clothes_test_pred).ravel(), 'y_clothes_test': np.array(y_clothes_test).ravel()})
    df_loss = pd.DataFrame({'dimensions': dimensions, 'loss_rate': loss_rate})
    df_beauty.to_csv('beauty.csv', index=False)
    df_appearance.to_csv('appearance.csv', index=False)
    df_appearance_defect.to_csv('appearance_defect.csv', index=False)
    df_affinity.to_csv('affinity.csv', index=False)
    df_clothes.to_csv('clothes.csv', index=False)
    df_loss.to_csv('analysis/loss.csv', index=False)


if __name__ == '__main__':
    # read_data()
    # generate_data_set()
    regression()
    # relation()
