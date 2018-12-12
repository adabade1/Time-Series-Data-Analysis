from matplotlib import pyplot
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas import DataFrame

# open files to write the output and the RMSE values
file = open("output.txt", 'w+')
file_rms = open("RMSE.txt", 'w+')
# Open the input file to get the data and perform a transpose on it.
data = pd.read_csv("C:\\Users\Akshata\PycharmProjects\dmProject\product_distribution_training_set",sep='\t',header=None).T
new_header = data.iloc[0]
data.columns = new_header
data = data.drop(data.index[[0]])
#print(data)
# Perform addition of all categorical values for each product.
overall_sum  = np.array(np.sum(data,axis=1).astype(float))
# print(overall_sum)
# Divide the data into training and testing set.
train=overall_sum[0:88]
test=overall_sum[88:]
# Convert the array to pandas series.
Time_Series = pd.Series(overall_sum)
# Time_Series_Train = pd.Series(train)
# Time_Series_Test = pd.Series(test)
Time_Series.index = pd.to_datetime(Time_Series.index,unit = 'D')
# Time_Series_Train.index = pd.to_datetime(Time_Series_Train.index,unit = 'D')
# Time_Series_Test.index = pd.to_datetime(Time_Series_Test.index,unit = 'D')
# print(Time_Series)
# Call ARIMA on the data set.
file.write(str(0))
file.write('\t')
history1 = [y for y in Time_Series.values]
model1 = ARIMA(history1, order=(7, 0, 0))
model_fit1 = model1.fit(disp=0)
forecast = model_fit1.predict(start=1, end=29)
print(forecast)
# Write the predicted values to output file.
for i in range(len(forecast)):
    file.write(str(int(forecast[i])))
    file.write('\t')
# Uncomment below lines to check the accuracy of the model
# predictions1 = list()
# for t1 in range(len(test)):
#     model1 = ARIMA(history1, order=(7, 0, 0))
#     model_fit1 = model1.fit(disp=0)
#     output1 = model_fit1.forecast()
#     yhat1 = output1[0]
#     predictions1.append(yhat1)
#     file.write(str(int(yhat1)))
#     file.write('\t')
#     history1.append(test[t1])
# error1 = sqrt(mean_squared_error(test, predictions1))
# print('Test RMSE: %.3f' % error1)
# pyplot.plot(test)
# pyplot.plot(predictions1, color='red')
# pyplot.title(str(cnt))
# pyplot.show()
#     print('predicted=%f' % (yhat1))
# print(predictions1)

# Predict the values of individual products for next 29 days
for cnt in range(1,101):
    file.write('\n')
    file.write('\n')
    file.write(str(int(new_header[cnt-1])))
    file.write('\t')
    overall_sum = []
    overall_sum  =  np.array(data[int(new_header[cnt-1])]).astype(float)
    # print(overall_sum)
    # overall_sum = np.array(data[158].astype(float))
    # print(overall_sum)
    train=overall_sum[0:88]
    test=overall_sum[88:]
    Time_Series = pd.Series(overall_sum)
    Time_Series_Train = pd.Series(train)
    Time_Series_Test = pd.Series(test)
    Time_Series.index = pd.to_datetime(Time_Series.index,unit = 'D')
    Time_Series_Train.index = pd.to_datetime(Time_Series_Train.index,unit = 'D')
    Time_Series_Test.index = pd.to_datetime(Time_Series_Test.index,unit = 'D')
# Call ARIMA on the data set.
    history1 = [y for y in Time_Series_Train.values]
    predictions1 = list()
    for t1 in range(len(test)):
        model1 = ARIMA(history1, order=(7, 1, 0))
        try:
            model_fit1 = model1.fit(disp=0)
        except:
            model1 = ARIMA(history1, order=(5, 1, 0))
            model_fit1 = model1.fit(disp=0)
        output1 = model_fit1.forecast()
        yhat1 = output1[0]
        predictions1.append(yhat1)
        obs1 = test[t1]
        history1.append(obs1)
# Uncomment below lines to find the residual values.
    # residuals = [test[i] - predictions1[i] for i in range(len(test))]
    # residuals = DataFrame(residuals)
    # bias = np.mean(residuals,axis=None)
    # print(bias)
    # final_prediction = [-(bias) + predictions1[i] for i in range(len(test))]
    error1 = sqrt(mean_squared_error(test, predictions1))
    # error2 = sqrt(mean_squared_error(test, final_prediction))
    print('Test RMSE: %.3f' % error1)
    # print('Test RMSE bias: %.3f' % error2)
    file_rms.write(str(int(new_header[cnt-1])))
    file_rms.write('\t')
    file_rms.write(str(error1))
    file_rms.write('\n')
    # pyplot.plot(test)
    # pyplot.plot(predictions1, color='red')
    # pyplot.title(str(cnt))
    # pyplot.show()

    history1 = [y for y in Time_Series.values]
    model1 = ARIMA(history1, order=(7, 0, 0))
    model_fit1 = model1.fit(disp=0)
    forecast = model_fit1.predict(start=1, end=29)
    print(forecast)
    for i in range(len(forecast)):
        file.write(str(int(forecast[i])))
        file.write('\t')
# Uncomment below lines to test the accuracy of the model.
    # ts1 = Time_Series.append(pd.Series(forecast), ignore_index=True)
    # pyplot.plot(ts1, color='red')
    # pyplot.show()

    # output1 = model_fit1.predict()
        # yhat1 = output1[0]
        # predictions1.append(yhat1)
        # file.write(str(int(yhat1)))
        # file.write('\t')
        # obs1 = test[t1]
        # history1.append(yhat1)
    # print("done")

        # print('predicted=%f, expected=%f' % (yhat1, obs1))
# error1 = sqrt(mean_squared_error(test, predictions1))
# print('Test RMSE: %.3f' % error1)
# print(predictions1)
# pyplot.plot(Time_Series)
# pyplot.plot(predictions1, color='red')
# pyplot.show()
