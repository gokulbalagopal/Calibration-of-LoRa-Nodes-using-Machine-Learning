# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import sklearn.gaussian_process as gp
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import matplotlib.dates as mdates




df = pd.read_csv("calibrate.csv")
df.info()
summ = df.iloc[:,1:].describe()
df.hist(bins=50,figsize=(20,15))
plt.show()
df["dateTime"] = pd.to_datetime(df["dateTime"])
df["dateTime"] = df["dateTime"].dt.date
y_grimm = []
y_loRa = []
y_Palas = []
x = []
col_name = list(df)

x.append("dateTime")
for i in col_name:
    if "_grimm" in i:
        y_grimm.append(i)
    if "_loRa" in i:
        x.append(i)
    if "Palas" in i:
        y_Palas.append(i)

#Data frames with grim target values
        #Create a dictionary called grim and store dataframes with each grim value as the key

grimm = {}
for i in y_grimm:
    grim_cols = x + [i]
    grimm[i[:-len("_grimm")]] = df[grim_cols]

Palas = {}
for i in y_Palas:
    Palas_cols = x + [i]
    Palas[i[:-len("Palas")]] = df[Palas_cols]
    



for k,v in enumerate(grimm):

    X = grimm[v].drop([v+"_grimm"],axis = 1)
    y = grimm[v][v+"_grimm"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state= 40)
    print("x train",X_train.shape)
    print("x test",X_test.shape)
    print("y train",y_train.shape)
    print("y test",y_test.shape)
    X.to_csv("X"+v+".csv")
    y.to_csv("y"+v+".csv")
#----------------------------------Supervised Learning-----------------------------------------------# 

#--------------Linear Regression for grimm------------------#
    lm = linear_model.LinearRegression()
    lm.fit( X_train.iloc[:,1:],y_train)
    predict_train = lm.predict(X_train.iloc[:,1:])
    predict_test = lm.predict(X_test.iloc[:,1:])
    print("first 5 predictions for each grim test set",predict_test[0:5])
    
    r2_score_train = round(metrics.r2_score(y_train, predict_train),2) 
    corr_coef_train = np.sqrt(r2_score_train)
    r2_score_test = round(metrics.r2_score(y_test, predict_test),2)
    corr_coef_test = np.sqrt(r2_score_test)
    error_train =  y_train - predict_train
    error_test = y_test - predict_test
    bins = np.linspace(-10, 10, 100)
    
    plt.hist(error_train, bins, label='train', color='green')
    plt.legend(loc='upper right')
    plt.show()
    plt.hist(error_train, bins, alpha=0.5, label='test', color='red')
    plt.legend(loc='upper right')
    plt.show()
    
    df_y = pd.DataFrame({'Actual': y_test, 'Predicted': predict_test})
    df_y_1 =df_y.head(25)
    df_y_1.plot(kind='bar',figsize=(16,10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    
    # Scatter plots,Error histograms MSE,RMSE, Time Series
    train_label = "Training Data R = " + str(r2_score_train)
    test_label ="Testing Data R = " + str(r2_score_test)
    plt.plot(y_train, y_train, linewidth=2, label ="1:1" )
    plt.scatter(y_train,predict_train, color='red',label = train_label)
    plt.scatter(y_test,predict_test, color='green',label = test_label)

    if y_test.name == "pm10_grimm":
        y_test.name = "pm 10_grimm" 
    if y_test.name == "pm1_grimm":
        y_test.name = "pm 1_grimm"
    if y_test.name == "pm2_5_grimm":
        y_test.name = "pm 2.5_grimm"  
    xlabel = "Actual "+ y_test.name[:-len("_grimm")]
    ylabel = "Estimated "+ y_test.name[:-len("_grimm")]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    tname = "Scatter Plot for " + y_test.name[:-len("_grimm")]
    plt.title(tname)
    plt.legend(loc="upper left")
    plt.show()
    print('Mean Absolute Error for Training set:', metrics.mean_absolute_error(y_train, predict_train))  
    print('Mean Squared Error for Training set:', metrics.mean_squared_error(y_train, predict_train))  
    print('Root Mean Squared Error for Training set:', np.sqrt(metrics.mean_squared_error(y_train, predict_train)))    
    print('Mean Absolute Error for Testset:', metrics.mean_absolute_error(y_test, predict_test))  
    print('Mean Squared Error for Testset:', metrics.mean_squared_error(y_test, predict_test))  
    print('Root Mean Squared Error for Testset:', np.sqrt(metrics.mean_squared_error(y_test, predict_test)))
    
# #    #-------Time Series for Linear Regression------#
    # sns.set()
    # y_test = pd.DataFrame(y_test.to_numpy())
    # time_train  = pd.DataFrame(X_train["dateTime"].to_numpy())
    # predict_test = pd.DataFrame(predict_test)
    # time_test  = pd.DataFrame(X_test["dateTime"].to_numpy())
    # df_timeseries_test = pd.concat([time_test,y_test],axis = 1)
    # df_timeseries_predict = pd.concat([time_test,predict_test],axis = 1)
    # df_timeseries_test.columns =["dateTime","test"]
    # df_timeseries_predict.columns =["dateTime","predict"]
    # df_timeseries_test = df_timeseries_test.sort_values(by='dateTime',ascending=True)
    # df_timeseries_test = df_timeseries_test.set_index('dateTime')
    # df_timeseries_predict = df_timeseries_predict.sort_values(by='dateTime',ascending=True)
    # df_timeseries_predict = df_timeseries_predict.set_index('dateTime')
    # df_timeseries_test.plot()
    # df_timeseries_predict.plot()
    # plt.plot(df_timeseries_test, linewidth = 2, label ="Actual",color = "red")
    # plt.plot(df_timeseries_predict,linewidth = 2,label = "Estimated", color = "green")
    # plt.xlabel("Date")
    # plt.ylabel(y_train.name)
    # plt.title("Time Series for linear Regression")
    # plt.xticks(rotation=45)
    # plt.show()
    # myFmt = DateFormatter('%Y-%m-%d')
    # # ax.xaxis.set_major_formatter(myFmt)
    
    # ## Rotate date labels automatically
    # fig.autofmt_xdate()
    # plt.legend(loc="upper left")
    # plt.show()
    
    
#--------------Neural Network for grimm------------------# 
    mlp = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    mlp.fit(X_train.iloc[:,1:],y_train)
    predict_train = mlp.predict(X_train.iloc[:,1:])
    predict_test = mlp.predict(X_test.iloc[:,1:])
    
    r2_score_train = round(metrics.r2_score(y_train, predict_train),2) 
    corr_coef_train = np.sqrt(r2_score_train)
    r2_score_test = round(metrics.r2_score(y_test, predict_test),2)
    corr_coef_test = np.sqrt(r2_score_test)
    error_train =  y_train - predict_train
    error_test = y_test - predict_test
    
    train_label = "Training Data R = " + str(r2_score_train)
    test_label ="Testing Data R = " + str(r2_score_test)
    plt.plot(y_train, y_train, linewidth=2, label ="1:1" )
    plt.scatter(y_train,predict_train, color='red',label = train_label)
    plt.scatter(y_test,predict_test, color='green',label = test_label)
    
    if y_test.name == "pm10_grimm":
        y_test.name = "pm 10_grimm" 
    if y_test.name == "pm1_grimm":
        y_test.name = "pm 1_grimm"
    if y_test.name == "pm2_5_grimm":
        y_test.name = "pm 2.5_grimm"  
    xlabel = "Actual "+ y_test.name[:-len("_grimm")]
    ylabel = "Estimated "+ y_test.name[:-len("_grimm")]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    tname = "Scatter Plot for " + y_test.name[:-len("_grimm")]
    plt.title(tname)
    plt.legend(loc="upper left")
    plt.show()
    
    # #-------Time Series for Neural Network------#
    # fig, ax = plt.subplots()
    # y_test = pd.DataFrame(y_test.to_numpy())
    # predict_test = pd.DataFrame(predict_test)
    # time_test  = pd.DataFrame(X_test["dateTime"].to_numpy())
    # df_timeseries_test = pd.concat([time_test,y_test],axis = 1)
    # df_timeseries_predict = pd.concat([time_test,predict_test],axis = 1)
    # df_timeseries_test.columns =["dateTime","test"]
    # df_timeseries_predict.columns =["dateTime","predict"]
    # df_timeseries_test = df_timeseries_test.sort_values(by='dateTime',ascending=True)
    # df_timeseries_predict = df_timeseries_predict.sort_values(by='dateTime',ascending=True)
    # ax.plot(df_timeseries_test.iloc[:,0],df_timeseries_test.iloc[:,1], linewidth = 2, label ="Actual",color = "red")
    # ax.plot(df_timeseries_predict.iloc[:,0],df_timeseries_predict.iloc[:,1], linewidth = 2,label = "Estimated", color = "green")
    # plt.xlabel("Date")
    # plt.ylabel(y_train.name)
    # plt.title("Time Series for Neural Network")
    # myFmt = DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(myFmt)
    
    # ## Rotate date labels automatically
    # fig.autofmt_xdate()
    # plt.legend(loc="upper left")
    # plt.show()
    
    
#--------------Support Vector Regression for grimm------------------#
    svr = SVR(kernel='rbf')
    svr.fit(X_train.iloc[:,1:],y_train)
    predict_train = svr.predict(X_train.iloc[:,1:])
    predict_test = svr.predict(X_test.iloc[:,1:])
    print("first 5 predictions for each grim test set",predict_test[0:5])
    
    r2_score_train = round(metrics.r2_score(y_train, predict_train),2) 
    corr_coef_train = np.sqrt(r2_score_train)
    r2_score_test = round(metrics.r2_score(y_test, predict_test),2)
    corr_coef_test = np.sqrt(r2_score_test)
    error_train =  y_train - predict_train
    error_test = y_test - predict_test
    
    train_label = "Training Data R = " + str(r2_score_train)
    test_label ="Testing Data R = " + str(r2_score_test)
    plt.plot(y_train, y_train, linewidth=2, label ="1:1" )
    plt.scatter(y_train,predict_train, color='red',label = train_label)
    plt.scatter(y_test,predict_test, color='green',label = test_label)
    
    if y_test.name == "pm10_grimm":
        y_test.name = "pm 10_grimm" 
    if y_test.name == "pm1_grimm":
        y_test.name = "pm 1_grimm"
    if y_test.name == "pm2_5_grimm":
        y_test.name = "pm 2.5_grimm"  
    xlabel = "Actual "+ y_test.name[:-len("_grimm")]
    ylabel = "Estimated "+ y_test.name[:-len("_grimm")]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    tname = "Scatter Plot for " + y_test.name[:-len("_grimm")]
    plt.title(tname)
    plt.legend(loc="upper left")
    plt.show()
    
    # #-------Time Series for Support Vector Regression ------#
    # fig, ax = plt.subplots()
    # y_test = pd.DataFrame(y_test.to_numpy())
    # predict_test = pd.DataFrame(predict_test)
    # time_test  = pd.DataFrame(X_test["dateTime"].to_numpy())
    # df_timeseries_test = pd.concat([time_test,y_test],axis = 1)
    # df_timeseries_predict = pd.concat([time_test,predict_test],axis = 1)
    # df_timeseries_test.columns =["dateTime","test"]
    # df_timeseries_predict.columns =["dateTime","predict"]
    # df_timeseries_test = df_timeseries_test.sort_values(by='dateTime',ascending=True)
    # df_timeseries_predict = df_timeseries_predict.sort_values(by='dateTime',ascending=True)
    # ax.plot(df_timeseries_test.iloc[:,0],df_timeseries_test.iloc[:,1], linewidth = 2, label ="Actual",color = "red")
    # ax.plot(df_timeseries_predict.iloc[:,0],df_timeseries_predict.iloc[:,1], linewidth = 2,label = "Estimated", color = "green")
    # plt.xlabel("Date")
    # plt.ylabel(y_train.name)
    # plt.title("Time Series for Support Vector Regression")
    # myFmt = DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(myFmt)
    
    # ## Rotate date labels automatically
    # fig.autofmt_xdate()
    # plt.legend(loc="upper left")
    # plt.show()

#--------------Gaussian Process Regression for grimm------------------#
    kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
    model.fit(X_train.iloc[:,1:],y_train)
    params = model.kernel_.get_params()
    predict_train = model.predict(X_train.iloc[:,1:])
    predict_test = model.predict(X_test.iloc[:,1:])
    MSE = ((predict_test - y_test.to_numpy())**2).mean()
    print("first 5 predictions for each grim test set",predict_test[0:5]) 
    print("Mean Square Error after gaussian process",MSE)
    
    r2_score_train = round(metrics.r2_score(y_train, predict_train),2) 
    corr_coef_train = np.sqrt(r2_score_train)
    r2_score_test = round(metrics.r2_score(y_test, predict_test),2)
    corr_coef_test = np.sqrt(r2_score_test)
    error_train =  y_train - predict_train
    error_test = y_test - predict_test
    
    train_label = "Training Data R = " + str(r2_score_train)
    test_label ="Testing Data R = " + str(r2_score_test)
    plt.plot(y_train, y_train, linewidth=2, label ="1:1" )
    plt.scatter(y_train,predict_train, color='red',label = train_label)
    plt.scatter(y_test,predict_test, color='green',label = test_label)
   
    if y_test.name == "pm10_grimm":
        y_test.name = "pm 10_grimm" 
    if y_test.name == "pm1_grimm":
        y_test.name = "pm 1_grimm"
    if y_test.name == "pm2_5_grimm":
        y_test.name = "pm 2.5_grimm"  
    xlabel = "Actual "+ y_test.name[:-len("_grimm")]
    ylabel = "Estimated "+ y_test.name[:-len("_grimm")]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    tname = "Scatter Plot for " + y_test.name[:-len("_grimm")]
    plt.title(tname)
    plt.legend(loc="upper left")
    plt.show()
    # #-------Time Series for Guassian Process Regression ------#
    # fig, ax = plt.subplots()
    # y_test = pd.DataFrame(y_test.to_numpy())
    # predict_test = pd.DataFrame(predict_test)
    # time_test  = pd.DataFrame(X_test["dateTime"].to_numpy())
    # df_timeseries_test = pd.concat([time_test,y_test],axis = 1)
    # df_timeseries_predict = pd.concat([time_test,predict_test],axis = 1)
    # df_timeseries_test.columns =["dateTime","test"]
    # df_timeseries_predict.columns =["dateTime","predict"]
    # df_timeseries_test = df_timeseries_test.sort_values(by='dateTime',ascending=True)
    # df_timeseries_predict = df_timeseries_predict.sort_values(by='dateTime',ascending=True)
    # ax.plot(df_timeseries_test.iloc[:,0],df_timeseries_test.iloc[:,1], linewidth = 2, label ="Actual",color = "red")
    # ax.plot(df_timeseries_predict.iloc[:,0],df_timeseries_predict.iloc[:,1], linewidth = 2,label = "Estimated", color = "green")
    # plt.xlabel("Date")
    # plt.ylabel(y_train.name)
    # plt.title("Time Series for Guassian Process Regression")
    # myFmt = DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(myFmt)
    
    # ## Rotate date labels automatically
    # fig.autofmt_xdate()
    # plt.legend(loc="upper left")
    # plt.show()
#--------------Decision Trees for grimm------------------#
    decision_tree_model = DecisionTreeRegressor(random_state = 42)
    decision_tree_model.fit(X_train.iloc[:,1:],y_train)
    predict_train = decision_tree_model.predict(X_train.iloc[:,1:])
    predict_test = decision_tree_model.predict(X_test.iloc[:,1:])
    
    r2_score_train = round(metrics.r2_score(y_train, predict_train),2) 
    corr_coef_train = np.sqrt(r2_score_train)
    r2_score_test = round(metrics.r2_score(y_test, predict_test),2)
    corr_coef_test = np.sqrt(r2_score_test)
    error_train =  y_train - predict_train
    error_test = y_test - predict_test
    
    train_label = "Training Data R = " + str(r2_score_train)
    test_label ="Testing Data R = " + str(r2_score_test)
    plt.plot(y_train, y_train, linewidth=2, label ="1:1" )
    plt.scatter(y_train,predict_train, color='red',label = train_label)
    plt.scatter(y_test,predict_test, color='green',label = test_label)
    
    if y_test.name == "pm10_grimm":
        y_test.name = "pm 10_grimm" 
    if y_test.name == "pm1_grimm":
        y_test.name = "pm 1_grimm"
    if y_test.name == "pm2_5_grimm":
        y_test.name = "pm 2.5_grimm"  
    xlabel = "Actual "+ y_test.name[:-len("_grimm")]
    ylabel = "Estimated "+ y_test.name[:-len("_grimm")]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    tname = "Scatter Plot for " + y_test.name[:-len("_grimm")]
    plt.title(tname)
    plt.legend(loc="upper left")
    plt.show()
    #-------Time Series for Decision Tree Regression ------#
    # fig, ax = plt.subplots()
    # y_test = pd.DataFrame(y_test.to_numpy())
    # predict_test = pd.DataFrame(predict_test)
    # time_test  = pd.DataFrame(X_test["dateTime"].to_numpy())
    # df_timeseries_test = pd.concat([time_test,y_test],axis = 1)
    # df_timeseries_predict = pd.concat([time_test,predict_test],axis = 1)
    # df_timeseries_test.columns =["dateTime","test"]
    # df_timeseries_predict.columns =["dateTime","predict"]
    # df_timeseries_test = df_timeseries_test.sort_values(by='dateTime',ascending=True)
    # df_timeseries_predict = df_timeseries_predict.sort_values(by='dateTime',ascending=True)
    # ax.plot(df_timeseries_test.iloc[:,0],df_timeseries_test.iloc[:,1], linewidth = 2, label ="Actual",color = "red")
    # ax.plot(df_timeseries_predict.iloc[:,0],df_timeseries_predict.iloc[:,1], linewidth = 2,label = "Estimated", color = "green")
    # plt.xlabel("Date")
    # plt.ylabel(y_train.name)
    # plt.title("Time Series for Decision Tree Regression")
    # myFmt = DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(myFmt)
    
    # ## Rotate date labels automatically
    # fig.autofmt_xdate()
    # plt.legend(loc="upper left")
    # plt.show()
#--------------Random Forest for grimm------------------#
    random_forest_regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
    random_forest_regressor.fit(X_train.iloc[:,1:], y_train)
    predict_train = random_forest_regressor.predict(X_train.iloc[:,1:])
    predict_test = random_forest_regressor.predict(X_test.iloc[:,1:])
    
    r2_score_train = round(metrics.r2_score(y_train, predict_train),2) 
    corr_coef_train = np.sqrt(r2_score_train)
    r2_score_test = round(metrics.r2_score(y_test, predict_test),2)
    corr_coef_test = np.sqrt(r2_score_test)
    error_train =  y_train - predict_train
    error_test = y_test - predict_test
    
    train_label = "Training Data R = " + str(r2_score_train)
    test_label ="Testing Data R = " + str(r2_score_test)
    plt.plot(y_train, y_train, linewidth=2, label ="1:1" )
    plt.scatter(y_train,predict_train, color='red',label = train_label)
    plt.scatter(y_test,predict_test, color='green',label = test_label)
   
    if y_test.name == "pm10_grimm":
        y_test.name = "pm 10_grimm" 
    if y_test.name == "pm1_grimm":
        y_test.name = "pm 1_grimm"
    if y_test.name == "pm2_5_grimm":
        y_test.name = "pm 2.5_grimm"  
    xlabel = "Actual "+ y_test.name[:-len("_grimm")]
    ylabel = "Estimated "+ y_test.name[:-len("_grimm")]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    tname = "Scatter Plot for " + y_test.name[:-len("_grimm")]
    plt.title(tname)
    plt.legend(loc="upper left")
    plt.show()
    # #-------Time Series for Random Forest Regression ------#
    # fig, ax = plt.subplots()
    # y_test = pd.DataFrame(y_test.to_numpy())
    # predict_test = pd.DataFrame(predict_test)
    # time_test  = pd.DataFrame(X_test["dateTime"].to_numpy())
    # df_timeseries_test = pd.concat([time_test,y_test],axis = 1)
    # df_timeseries_predict = pd.concat([time_test,predict_test],axis = 1)
    # df_timeseries_test.columns =["dateTime","test"]
    # df_timeseries_predict.columns =["dateTime","predict"]
    # df_timeseries_test = df_timeseries_test.sort_values(by='dateTime',ascending=True)
    # df_timeseries_predict = df_timeseries_predict.sort_values(by='dateTime',ascending=True)
    # ax.plot(df_timeseries_test.iloc[:,0],df_timeseries_test.iloc[:,1], linewidth = 2, label ="Actual",color = "red")
    # ax.plot(df_timeseries_predict.iloc[:,0],df_timeseries_predict.iloc[:,1], linewidth = 2,label = "Estimated", color = "green")
    # plt.xlabel("Date")
    # plt.ylabel(y_train.name)
    # plt.title("Time Series for Random Forest Regression")
    # myFmt = DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(myFmt)
    
    ## Rotate date labels automatically
    # fig.autofmt_xdate()
    # plt.legend(loc="upper left")
    # plt.show()
    # Evaluating the Algorithm
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predict_test))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predict_test))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predict_test)))
    label = y_test.name[:-len("_grimm")]
    ls_lab = X_train.iloc[:,1:].columns
    ls_feat_lab = []
    for i in ls_lab:
        if "_loRa" in i:
            print(i[:-len("_loRa")])
            ls_feat_lab.append(i[:-len("_loRa")])    
    feat_importances = pd.Series(random_forest_regressor.feature_importances_, index = ls_feat_lab)
    feat_importances = feat_importances.drop(labels = ["P2_ratio","P2_lpo","P1_ratio","P1_lpo"])
    feat_importances.nsmallest(13).plot(kind='barh',title = label )
    plt.show()
#"""new_x = []    
#for i in x:
#    new_x.append(i[:-len("_loRa")])
#"""
#"""new_y_grimm = []    
#for i in y_grimm:
#    new_y_grimm.append(i[:-len("_grimm")])"""


#----------------Unsupervised Learning--------------------#
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE    
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

    #------------------ Kmeans ---------------------#
df_kmeans = X.iloc[:,1:]
range_n_clusters = list (range(2,10))
print ("Number of clusters from 2 to 9: \n", range_n_clusters)
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict(df_kmeans)
    centers = clusterer.cluster_centers_

    score = silhouette_score(df_kmeans, preds)
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
kmeans = KMeans(n_clusters=2,precompute_distances="auto", n_jobs=-1)
df_kmeans['clusters'] = kmeans.fit_predict(df_kmeans)
### Run PCA on the data and reduce the dimensions in pca_num_components dimensions
pca_num_components = 2
reduced_data_kmeans = PCA(n_components=pca_num_components).fit_transform(df_kmeans)
result_kmeans = pd.DataFrame(reduced_data_kmeans,columns=['pca1','pca2'])

sns.scatterplot(x="pca1",y="pca2", hue= df_kmeans['clusters'],data = result_kmeans)
plt.title('K-means Clustering with 2 dimensions')
plt.show()

    #------------------ Heirarchial ---------------------#
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    
    linkage_matrix = np.column_stack([model.children_, model.distances_,counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
data_heirarchial =X.iloc[:,1:]
cluster_dist = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
cluster_dist.fit(data_heirarchial)

plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(cluster_dist, truncate_mode='level', p=2)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

    #-------------Gaussian Mixture Model----------------#
data_gmm = X.iloc[:,1:]
gmm = GaussianMixture(n_components=2).fit(data_gmm)
data_gmm["labels"] = gmm.predict(data_gmm)

### Run PCA on the data and reduce the dimensions in pca_num_components dimensions
pca_num_components = 2
reduced_data_gmm = PCA(n_components=pca_num_components).fit_transform(data_gmm)
results_gmm = pd.DataFrame(reduced_data_gmm,columns=['pca1','pca2'])

sns.scatterplot(x="pca1",y="pca2", hue= data_gmm['labels'],data = results_gmm)
plt.title('Gaussian Mixture with 2 dimensions')
plt.show()
    #-----------Hidden Markov Models---------#
data_hmm = X.iloc[:,1:]
hiddenMarkov= hmm.GaussianHMM(2, "full")
hiddenMarkov.fit(data_hmm)
data_hmm["labels"] = hiddenMarkov.predict(data_hmm)
pca_num_components = 2
reduced_data_hmm = PCA(n_components=pca_num_components).fit_transform(data_hmm)
results_hmm = pd.DataFrame(reduced_data_hmm,columns=['pca1','pca2'])

sns.scatterplot(x="pca1",y="pca2", hue= data_hmm['labels'],data = results_hmm)
plt.title('Hidden Markov with 2 dimensions')
plt.show()




















#---------------------------palas--------------------------#
for k,v in enumerate(Palas):

    X = Palas[v].drop([v+"Palas"],axis = 1)
    y = grimm[v][v+"_grimm"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state= 40)
    print("x train",X_train.shape)
    print("x test",X_test.shape)
    print("y train",y_train.shape)
    print("y test",y_test.shape)
    X.to_csv("X"+v+".csv")
    y.to_csv("y"+v+".csv")
#----------------------------------Supervised Learning-----------------------------------------------# 

#--------------Linear Regression for grimm------------------#
    lm = linear_model.LinearRegression()
    lm.fit( X_train.iloc[:,1:],y_train)
    predict_train = lm.predict(X_train.iloc[:,1:])
    predict_test = lm.predict(X_test.iloc[:,1:])
    print("first 5 predictions for each grim test set",predict_test[0:5])
    
    r2_score_train = round(metrics.r2_score(y_train, predict_train),2) 
    corr_coef_train = np.sqrt(r2_score_train)
    r2_score_test = round(metrics.r2_score(y_test, predict_test),2)
    corr_coef_test = np.sqrt(r2_score_test)
    error_train =  y_train - predict_train
    error_test = y_test - predict_test
    bins = np.linspace(-10, 10, 100)
    
    plt.hist(error_train, bins, label='train', color='green')
    plt.legend(loc='upper right')
    plt.show()
    plt.hist(error_train, bins, alpha=0.5, label='test', color='red')
    plt.legend(loc='upper right')
    plt.show()
    
    df_y = pd.DataFrame({'Actual': y_test, 'Predicted': predict_test})
    df_y_1 =df_y.head(25)
    df_y_1.plot(kind='bar',figsize=(16,10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    
    # Scatter plots,Error histograms MSE,RMSE, Time Series
    train_label = "Training Data R = " + str(r2_score_train)
    test_label ="Testing Data R = " + str(r2_score_test)
    plt.plot(y_train, y_train, linewidth=2, label ="1:1" )
    plt.scatter(y_train,predict_train, color='red',label = train_label)
    plt.scatter(y_test,predict_test, color='green',label = test_label)

    if y_test.name == "pm10_grimm":
        y_test.name = "pm 10_grimm" 
    if y_test.name == "pm1_grimm":
        y_test.name = "pm 1_grimm"
    if y_test.name == "pm2_5_grimm":
        y_test.name = "pm 2.5_grimm"  
    xlabel = "Actual "+ y_test.name[:-len("_grimm")]
    ylabel = "Estimated "+ y_test.name[:-len("_grimm")]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    tname = "Scatter Plot for " + y_test.name[:-len("_grimm")]
    plt.title(tname)
    plt.legend(loc="upper left")
    plt.show()
    print('Mean Absolute Error for Training set:', metrics.mean_absolute_error(y_train, predict_train))  
    print('Mean Squared Error for Training set:', metrics.mean_squared_error(y_train, predict_train))  
    print('Root Mean Squared Error for Training set:', np.sqrt(metrics.mean_squared_error(y_train, predict_train)))    
    print('Mean Absolute Error for Testset:', metrics.mean_absolute_error(y_test, predict_test))  
    print('Mean Squared Error for Testset:', metrics.mean_squared_error(y_test, predict_test))  
    print('Root Mean Squared Error for Testset:', np.sqrt(metrics.mean_squared_error(y_test, predict_test)))
    
# #    #-------Time Series for Linear Regression------#
    # sns.set()
    # y_test = pd.DataFrame(y_test.to_numpy())
    # time_train  = pd.DataFrame(X_train["dateTime"].to_numpy())
    # predict_test = pd.DataFrame(predict_test)
    # time_test  = pd.DataFrame(X_test["dateTime"].to_numpy())
    # df_timeseries_test = pd.concat([time_test,y_test],axis = 1)
    # df_timeseries_predict = pd.concat([time_test,predict_test],axis = 1)
    # df_timeseries_test.columns =["dateTime","test"]
    # df_timeseries_predict.columns =["dateTime","predict"]
    # df_timeseries_test = df_timeseries_test.sort_values(by='dateTime',ascending=True)
    # df_timeseries_test = df_timeseries_test.set_index('dateTime')
    # df_timeseries_predict = df_timeseries_predict.sort_values(by='dateTime',ascending=True)
    # df_timeseries_predict = df_timeseries_predict.set_index('dateTime')
    # df_timeseries_test.plot()
    # df_timeseries_predict.plot()
    # plt.plot(df_timeseries_test, linewidth = 2, label ="Actual",color = "red")
    # plt.plot(df_timeseries_predict,linewidth = 2,label = "Estimated", color = "green")
    # plt.xlabel("Date")
    # plt.ylabel(y_train.name)
    # plt.title("Time Series for linear Regression")
    # plt.xticks(rotation=45)
    # plt.show()
    # myFmt = DateFormatter('%Y-%m-%d')
    # # ax.xaxis.set_major_formatter(myFmt)
    
    # ## Rotate date labels automatically
    # fig.autofmt_xdate()
    # plt.legend(loc="upper left")
    # plt.show()
    
    
#--------------Neural Network for grimm------------------# 
    mlp = MLPRegressor(
    hidden_layer_sizes=(10,),  activation='relu', solver='adam', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    mlp.fit(X_train.iloc[:,1:],y_train)
    predict_train = mlp.predict(X_train.iloc[:,1:])
    predict_test = mlp.predict(X_test.iloc[:,1:])
    
    r2_score_train = round(metrics.r2_score(y_train, predict_train),2) 
    corr_coef_train = np.sqrt(r2_score_train)
    r2_score_test = round(metrics.r2_score(y_test, predict_test),2)
    corr_coef_test = np.sqrt(r2_score_test)
    error_train =  y_train - predict_train
    error_test = y_test - predict_test
    
    train_label = "Training Data R = " + str(r2_score_train)
    test_label ="Testing Data R = " + str(r2_score_test)
    plt.plot(y_train, y_train, linewidth=2, label ="1:1" )
    plt.scatter(y_train,predict_train, color='red',label = train_label)
    plt.scatter(y_test,predict_test, color='green',label = test_label)
    
    if y_test.name == "pm10_grimm":
        y_test.name = "pm 10_grimm" 
    if y_test.name == "pm1_grimm":
        y_test.name = "pm 1_grimm"
    if y_test.name == "pm2_5_grimm":
        y_test.name = "pm 2.5_grimm"  
    xlabel = "Actual "+ y_test.name[:-len("_grimm")]
    ylabel = "Estimated "+ y_test.name[:-len("_grimm")]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    tname = "Scatter Plot for " + y_test.name[:-len("_grimm")]
    plt.title(tname)
    plt.legend(loc="upper left")
    plt.show()
    
    # #-------Time Series for Neural Network------#
    # fig, ax = plt.subplots()
    # y_test = pd.DataFrame(y_test.to_numpy())
    # predict_test = pd.DataFrame(predict_test)
    # time_test  = pd.DataFrame(X_test["dateTime"].to_numpy())
    # df_timeseries_test = pd.concat([time_test,y_test],axis = 1)
    # df_timeseries_predict = pd.concat([time_test,predict_test],axis = 1)
    # df_timeseries_test.columns =["dateTime","test"]
    # df_timeseries_predict.columns =["dateTime","predict"]
    # df_timeseries_test = df_timeseries_test.sort_values(by='dateTime',ascending=True)
    # df_timeseries_predict = df_timeseries_predict.sort_values(by='dateTime',ascending=True)
    # ax.plot(df_timeseries_test.iloc[:,0],df_timeseries_test.iloc[:,1], linewidth = 2, label ="Actual",color = "red")
    # ax.plot(df_timeseries_predict.iloc[:,0],df_timeseries_predict.iloc[:,1], linewidth = 2,label = "Estimated", color = "green")
    # plt.xlabel("Date")
    # plt.ylabel(y_train.name)
    # plt.title("Time Series for Neural Network")
    # myFmt = DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(myFmt)
    
    # ## Rotate date labels automatically
    # fig.autofmt_xdate()
    # plt.legend(loc="upper left")
    # plt.show()
    
    
#--------------Support Vector Regression for grimm------------------#
    svr = SVR(kernel='rbf')
    svr.fit(X_train.iloc[:,1:],y_train)
    predict_train = svr.predict(X_train.iloc[:,1:])
    predict_test = svr.predict(X_test.iloc[:,1:])
    print("first 5 predictions for each grim test set",predict_test[0:5])
    
    r2_score_train = round(metrics.r2_score(y_train, predict_train),2) 
    corr_coef_train = np.sqrt(r2_score_train)
    r2_score_test = round(metrics.r2_score(y_test, predict_test),2)
    corr_coef_test = np.sqrt(r2_score_test)
    error_train =  y_train - predict_train
    error_test = y_test - predict_test
    
    train_label = "Training Data R = " + str(r2_score_train)
    test_label ="Testing Data R = " + str(r2_score_test)
    plt.plot(y_train, y_train, linewidth=2, label ="1:1" )
    plt.scatter(y_train,predict_train, color='red',label = train_label)
    plt.scatter(y_test,predict_test, color='green',label = test_label)
    
    if y_test.name == "pm10_grimm":
        y_test.name = "pm 10_grimm" 
    if y_test.name == "pm1_grimm":
        y_test.name = "pm 1_grimm"
    if y_test.name == "pm2_5_grimm":
        y_test.name = "pm 2.5_grimm"  
    xlabel = "Actual "+ y_test.name[:-len("_grimm")]
    ylabel = "Estimated "+ y_test.name[:-len("_grimm")]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    tname = "Scatter Plot for " + y_test.name[:-len("_grimm")]
    plt.title(tname)
    plt.legend(loc="upper left")
    plt.show()
    
    # #-------Time Series for Support Vector Regression ------#
    # fig, ax = plt.subplots()
    # y_test = pd.DataFrame(y_test.to_numpy())
    # predict_test = pd.DataFrame(predict_test)
    # time_test  = pd.DataFrame(X_test["dateTime"].to_numpy())
    # df_timeseries_test = pd.concat([time_test,y_test],axis = 1)
    # df_timeseries_predict = pd.concat([time_test,predict_test],axis = 1)
    # df_timeseries_test.columns =["dateTime","test"]
    # df_timeseries_predict.columns =["dateTime","predict"]
    # df_timeseries_test = df_timeseries_test.sort_values(by='dateTime',ascending=True)
    # df_timeseries_predict = df_timeseries_predict.sort_values(by='dateTime',ascending=True)
    # ax.plot(df_timeseries_test.iloc[:,0],df_timeseries_test.iloc[:,1], linewidth = 2, label ="Actual",color = "red")
    # ax.plot(df_timeseries_predict.iloc[:,0],df_timeseries_predict.iloc[:,1], linewidth = 2,label = "Estimated", color = "green")
    # plt.xlabel("Date")
    # plt.ylabel(y_train.name)
    # plt.title("Time Series for Support Vector Regression")
    # myFmt = DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(myFmt)
    
    # ## Rotate date labels automatically
    # fig.autofmt_xdate()
    # plt.legend(loc="upper left")
    # plt.show()

#--------------Gaussian Process Regression for grimm------------------#
    kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
    model.fit(X_train.iloc[:,1:],y_train)
    params = model.kernel_.get_params()
    predict_train = model.predict(X_train.iloc[:,1:])
    predict_test = model.predict(X_test.iloc[:,1:])
    MSE = ((predict_test - y_test.to_numpy())**2).mean()
    print("first 5 predictions for each grim test set",predict_test[0:5]) 
    print("Mean Square Error after gaussian process",MSE)
    
    r2_score_train = round(metrics.r2_score(y_train, predict_train),2) 
    corr_coef_train = np.sqrt(r2_score_train)
    r2_score_test = round(metrics.r2_score(y_test, predict_test),2)
    corr_coef_test = np.sqrt(r2_score_test)
    error_train =  y_train - predict_train
    error_test = y_test - predict_test
    
    train_label = "Training Data R = " + str(r2_score_train)
    test_label ="Testing Data R = " + str(r2_score_test)
    plt.plot(y_train, y_train, linewidth=2, label ="1:1" )
    plt.scatter(y_train,predict_train, color='red',label = train_label)
    plt.scatter(y_test,predict_test, color='green',label = test_label)
   
    if y_test.name == "pm10_grimm":
        y_test.name = "pm 10_grimm" 
    if y_test.name == "pm1_grimm":
        y_test.name = "pm 1_grimm"
    if y_test.name == "pm2_5_grimm":
        y_test.name = "pm 2.5_grimm"  
    xlabel = "Actual "+ y_test.name[:-len("_grimm")]
    ylabel = "Estimated "+ y_test.name[:-len("_grimm")]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    tname = "Scatter Plot for " + y_test.name[:-len("_grimm")]
    plt.title(tname)
    plt.legend(loc="upper left")
    plt.show()
    # #-------Time Series for Guassian Process Regression ------#
    # fig, ax = plt.subplots()
    # y_test = pd.DataFrame(y_test.to_numpy())
    # predict_test = pd.DataFrame(predict_test)
    # time_test  = pd.DataFrame(X_test["dateTime"].to_numpy())
    # df_timeseries_test = pd.concat([time_test,y_test],axis = 1)
    # df_timeseries_predict = pd.concat([time_test,predict_test],axis = 1)
    # df_timeseries_test.columns =["dateTime","test"]
    # df_timeseries_predict.columns =["dateTime","predict"]
    # df_timeseries_test = df_timeseries_test.sort_values(by='dateTime',ascending=True)
    # df_timeseries_predict = df_timeseries_predict.sort_values(by='dateTime',ascending=True)
    # ax.plot(df_timeseries_test.iloc[:,0],df_timeseries_test.iloc[:,1], linewidth = 2, label ="Actual",color = "red")
    # ax.plot(df_timeseries_predict.iloc[:,0],df_timeseries_predict.iloc[:,1], linewidth = 2,label = "Estimated", color = "green")
    # plt.xlabel("Date")
    # plt.ylabel(y_train.name)
    # plt.title("Time Series for Guassian Process Regression")
    # myFmt = DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(myFmt)
    
    # ## Rotate date labels automatically
    # fig.autofmt_xdate()
    # plt.legend(loc="upper left")
    # plt.show()
#--------------Decision Trees for grimm------------------#
    decision_tree_model = DecisionTreeRegressor(random_state = 42)
    decision_tree_model.fit(X_train.iloc[:,1:],y_train)
    predict_train = decision_tree_model.predict(X_train.iloc[:,1:])
    predict_test = decision_tree_model.predict(X_test.iloc[:,1:])
    
    r2_score_train = round(metrics.r2_score(y_train, predict_train),2) 
    corr_coef_train = np.sqrt(r2_score_train)
    r2_score_test = round(metrics.r2_score(y_test, predict_test),2)
    corr_coef_test = np.sqrt(r2_score_test)
    error_train =  y_train - predict_train
    error_test = y_test - predict_test
    
    train_label = "Training Data R = " + str(r2_score_train)
    test_label ="Testing Data R = " + str(r2_score_test)
    plt.plot(y_train, y_train, linewidth=2, label ="1:1" )
    plt.scatter(y_train,predict_train, color='red',label = train_label)
    plt.scatter(y_test,predict_test, color='green',label = test_label)
    
    if y_test.name == "pm10_grimm":
        y_test.name = "pm 10_grimm" 
    if y_test.name == "pm1_grimm":
        y_test.name = "pm 1_grimm"
    if y_test.name == "pm2_5_grimm":
        y_test.name = "pm 2.5_grimm"  
    xlabel = "Actual "+ y_test.name[:-len("_grimm")]
    ylabel = "Estimated "+ y_test.name[:-len("_grimm")]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    tname = "Scatter Plot for " + y_test.name[:-len("_grimm")]
    plt.title(tname)
    plt.legend(loc="upper left")
    plt.show()
    #-------Time Series for Decision Tree Regression ------#
    # fig, ax = plt.subplots()
    # y_test = pd.DataFrame(y_test.to_numpy())
    # predict_test = pd.DataFrame(predict_test)
    # time_test  = pd.DataFrame(X_test["dateTime"].to_numpy())
    # df_timeseries_test = pd.concat([time_test,y_test],axis = 1)
    # df_timeseries_predict = pd.concat([time_test,predict_test],axis = 1)
    # df_timeseries_test.columns =["dateTime","test"]
    # df_timeseries_predict.columns =["dateTime","predict"]
    # df_timeseries_test = df_timeseries_test.sort_values(by='dateTime',ascending=True)
    # df_timeseries_predict = df_timeseries_predict.sort_values(by='dateTime',ascending=True)
    # ax.plot(df_timeseries_test.iloc[:,0],df_timeseries_test.iloc[:,1], linewidth = 2, label ="Actual",color = "red")
    # ax.plot(df_timeseries_predict.iloc[:,0],df_timeseries_predict.iloc[:,1], linewidth = 2,label = "Estimated", color = "green")
    # plt.xlabel("Date")
    # plt.ylabel(y_train.name)
    # plt.title("Time Series for Decision Tree Regression")
    # myFmt = DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(myFmt)
    
    # ## Rotate date labels automatically
    # fig.autofmt_xdate()
    # plt.legend(loc="upper left")
    # plt.show()
#--------------Random Forest for grimm------------------#
    random_forest_regressor = RandomForestRegressor(n_estimators = 50, random_state = 0)
    random_forest_regressor.fit(X_train.iloc[:,1:], y_train)
    predict_train = random_forest_regressor.predict(X_train.iloc[:,1:])
    predict_test = random_forest_regressor.predict(X_test.iloc[:,1:])
    
    r2_score_train = round(metrics.r2_score(y_train, predict_train),2) 
    corr_coef_train = np.sqrt(r2_score_train)
    r2_score_test = round(metrics.r2_score(y_test, predict_test),2)
    corr_coef_test = np.sqrt(r2_score_test)
    error_train =  y_train - predict_train
    error_test = y_test - predict_test
    
    train_label = "Training Data R = " + str(r2_score_train)
    test_label ="Testing Data R = " + str(r2_score_test)
    plt.plot(y_train, y_train, linewidth=2, label ="1:1" )
    plt.scatter(y_train,predict_train, color='red',label = train_label)
    plt.scatter(y_test,predict_test, color='green',label = test_label)
   
    if y_test.name == "pm10_grimm":
        y_test.name = "pm 10_grimm" 
    if y_test.name == "pm1_grimm":
        y_test.name = "pm 1_grimm"
    if y_test.name == "pm2_5_grimm":
        y_test.name = "pm 2.5_grimm"  
    xlabel = "Actual "+ y_test.name[:-len("_grimm")]
    ylabel = "Estimated "+ y_test.name[:-len("_grimm")]
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    tname = "Scatter Plot for " + y_test.name[:-len("_grimm")]
    plt.title(tname)
    plt.legend(loc="upper left")
    plt.show()
    # #-------Time Series for Random Forest Regression ------#
    # fig, ax = plt.subplots()
    # y_test = pd.DataFrame(y_test.to_numpy())
    # predict_test = pd.DataFrame(predict_test)
    # time_test  = pd.DataFrame(X_test["dateTime"].to_numpy())
    # df_timeseries_test = pd.concat([time_test,y_test],axis = 1)
    # df_timeseries_predict = pd.concat([time_test,predict_test],axis = 1)
    # df_timeseries_test.columns =["dateTime","test"]
    # df_timeseries_predict.columns =["dateTime","predict"]
    # df_timeseries_test = df_timeseries_test.sort_values(by='dateTime',ascending=True)
    # df_timeseries_predict = df_timeseries_predict.sort_values(by='dateTime',ascending=True)
    # ax.plot(df_timeseries_test.iloc[:,0],df_timeseries_test.iloc[:,1], linewidth = 2, label ="Actual",color = "red")
    # ax.plot(df_timeseries_predict.iloc[:,0],df_timeseries_predict.iloc[:,1], linewidth = 2,label = "Estimated", color = "green")
    # plt.xlabel("Date")
    # plt.ylabel(y_train.name)
    # plt.title("Time Series for Random Forest Regression")
    # myFmt = DateFormatter('%Y-%m-%d')
    # ax.xaxis.set_major_formatter(myFmt)
    
    ## Rotate date labels automatically
    # fig.autofmt_xdate()
    # plt.legend(loc="upper left")
    # plt.show()
    # Evaluating the Algorithm
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predict_test))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predict_test))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predict_test)))
    label = y_test.name[:-len("_grimm")]
    ls_lab = X_train.iloc[:,1:].columns
    ls_feat_lab = []
    for i in ls_lab:
        if "_loRa" in i:
            print(i[:-len("_loRa")])
            ls_feat_lab.append(i[:-len("_loRa")])    
    feat_importances = pd.Series(random_forest_regressor.feature_importances_, index = ls_feat_lab)
    feat_importances = feat_importances.drop(labels = ["P2_ratio","P2_lpo","P1_ratio","P1_lpo"])
    feat_importances.nsmallest(13).plot(kind='barh',title = label )
    plt.show()
#"""new_x = []    
#for i in x:
#    new_x.append(i[:-len("_loRa")])
#"""
#"""new_y_grimm = []    
#for i in y_grimm:
#    new_y_grimm.append(i[:-len("_grimm")])"""


#----------------Unsupervised Learning--------------------#
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE    
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

    #------------------ Kmeans ---------------------#
df_kmeans = X.iloc[:,1:]
range_n_clusters = list (range(2,10))
print ("Number of clusters from 2 to 9: \n", range_n_clusters)
for n_clusters in range_n_clusters:
    clusterer = KMeans(n_clusters=n_clusters)
    preds = clusterer.fit_predict(df_kmeans)
    centers = clusterer.cluster_centers_

    score = silhouette_score(df_kmeans, preds)
    print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
kmeans = KMeans(n_clusters=2,precompute_distances="auto", n_jobs=-1)
df_kmeans['labels'] = kmeans.fit_predict(df_kmeans)
### Run PCA on the data and reduce the dimensions in pca_num_components dimensions
pca_num_components = 2
reduced_data_kmeans = PCA(n_components=pca_num_components).fit_transform(df_kmeans)
result_kmeans = pd.DataFrame(reduced_data_kmeans,columns=['pca1','pca2'])

sns.scatterplot(x="pca1",y="pca2", hue= df_kmeans['labels'],data = result_kmeans)
plt.title('K-means Clustering with 2 dimensions')
plt.show()

    #------------------ Heirarchial ---------------------#
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    
    linkage_matrix = np.column_stack([model.children_, model.distances_,counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
data_heirarchial =X.iloc[:,1:]
cluster_dist = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
cluster_dist.fit(data_heirarchial)

plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(cluster_dist, truncate_mode='level', p=2)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

    #-------------Gaussian Mixture Model----------------#
data_gmm = X.iloc[:,1:]
gmm = GaussianMixture(n_components=2).fit(data_gmm)
data_gmm["labels"] = gmm.predict(data_gmm)

### Run PCA on the data and reduce the dimensions in pca_num_components dimensions
pca_num_components = 2
reduced_data_gmm = PCA(n_components=pca_num_components).fit_transform(data_gmm)
results_gmm = pd.DataFrame(reduced_data_gmm,columns=['pca1','pca2'])

sns.scatterplot(x="pca1",y="pca2", hue= data_gmm['labels'],data = results_gmm)
sns.set()
plt.title('Gaussian Mixture with 2 dimensions')
plt.show()
    #-----------Hidden Markov Models---------#
data_hmm = X.iloc[:,1:]
hiddenMarkov= hmm.GaussianHMM(2, "full")
hiddenMarkov.fit(data_hmm)
data_hmm["labels"] = hiddenMarkov.predict(data_hmm)
pca_num_components = 2
reduced_data_hmm = PCA(n_components=pca_num_components).fit_transform(data_hmm)
results_hmm = pd.DataFrame(reduced_data_hmm,columns=['pca1','pca2'])

sns.scatterplot(x="pca1",y="pca2", hue= data_hmm['labels'],data = results_hmm)
sns.set()
plt.title('Hidden Markov with 2 dimensions')
plt.show()

