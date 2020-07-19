print("Problem 1: Predict time series revenue of a company")
print("start")
import sklearn
import matplotlib.pylab as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
import requests, pandas as pd, numpy as np
from pandas import DataFrame
from io import StringIO
import time, json
from datetime import date, datetime
import statsmodels
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from joblib import dump, load


# # Load and read data for Problem1

# In[60]:


#Import data
activity= pd.read_csv('data/activity.csv')
marketing=pd.read_csv('data/marketing.csv')
pricing=pd.read_csv('data/pricing.csv')
transaction=pd.read_csv('data/transaction.csv')


# # Preview the data

# In[4]:


#Preview data
#print(activity.head(2))
#activity.ACTIVITY.unique()
#print(marketing.head(2))
#print(pricing.head(2))
#print(transaction.head(2))


# In[5]:


#Calculate the number of loggedin, Number of New acc by date
Loggedin=activity.loc[activity['ACTIVITY'] == 'logged in']
LoggedinSummary=Loggedin.groupby(['DATE'])['USERID'].agg('count').reset_index()
LoggedinSummary.columns= ['DATE','NumberLoggedin']
CreateAcc=activity.loc[activity['ACTIVITY'] == 'created account']
CreateAccSummary=CreateAcc.groupby(['DATE'])['USERID'].agg('count').reset_index()
CreateAccSummary.columns= ['DATE','NumberNewAcc']
AccountData=pd.merge(LoggedinSummary, CreateAccSummary, on=['DATE'], how='left')

print ("Figure out the data of Sale")
TransactionSummary=transaction.groupby(['DATE'])['TOTAL'].agg('sum').reset_index()
Saledata= pd.merge(TransactionSummary, pricing, on=['DATE'], how='left')
Saledata.columns=['DATE','TotalSale','Price','PricingPolicy']
Saledata=pd.merge(Saledata, marketing, on=['DATE'], how='left')


print("Merge all data")
Fulldata=pd.merge(Saledata, AccountData, on=['DATE'], how='left')
Fulldata['DATE'] = Fulldata['DATE'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
indexed_df = Fulldata.set_index('DATE')

ts_rev= indexed_df['TotalSale']

#plt.plot(ts_rev.index.to_pydatetime(), ts_rev.values)

ts_rev_week = ts_rev.resample('W-SAT').mean()


#plt.plot(ts_rev_week.index.to_pydatetime(), ts_rev_week.values)

def check_stationarity(timeseries):
    
    #Determing rolling statistics
    rolling_mean = timeseries.rolling(window=52,center=False).mean() 
    rolling_std = timeseries.rolling(window=52,center=False).std()

    #Plot rolling statistics:
    original = plt.plot(timeseries.index.to_pydatetime(), timeseries.values, color='blue',label='Original')
    mean = plt.plot(rolling_mean.index.to_pydatetime(), rolling_mean.values, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std.index.to_pydatetime(), rolling_std.values, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dickey_fuller_test = adfuller(timeseries, autolag='AIC')
    dfresults = pd.Series(dickey_fuller_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dickey_fuller_test[4].items():
        dfresults['Critical Value (%s)'%key] = value
    print (dfresults)


# In[15]:


check_stationarity(ts_rev_week)


# In[16]:


ts_rev_week_log = np.log(ts_rev_week)


# In[17]:


check_stationarity(ts_rev_week_log)


# In[35]:


#decomposition = seasonal_decompose(ts_rev_week)

#trend = decomposition.trend
#seasonal = decomposition.seasonal
#residual = decomposition.resid

# Select the most recent weeks 
#ts_rev_week_log_select = ts_rev_week_log[-100:]

#plt.subplot(411)
#plt.plot(ts_rev_week_log_select.index.to_pydatetime(), ts_rev_week_log_select.values, label='Original')
#plt.legend(loc='best')
#plt.subplot(412)
#plt.plot(ts_rev_week_log_select.index.to_pydatetime(), trend[-100:].values, label='Trend')
#plt.legend(loc='best')
#plt.subplot(413)
#plt.plot(ts_rev_week_log_select.index.to_pydatetime(), seasonal[-100:].values,label='Seasonality')
#plt.legend(loc='best')
#plt.subplot(414)
#plt.plot(ts_rev_week_log_select.index.to_pydatetime(), residual[-100:].values, label='Residuals')
#plt.legend(loc='best')
#plt.tight_layout()

ts_rev_week_log_diff = ts_rev_week_log - ts_rev_week_log.shift()
plt.plot(ts_rev_week_log_diff.index.to_pydatetime(), ts_rev_week_log_diff.values)

ts_rev_week_log_diff.dropna(inplace=True)
check_stationarity(ts_rev_week_log_diff)

#ACF and PACF plots

lag_auto_corr = acf(ts_rev_week_log_diff, nlags=10)
lag_par_auto_corr = pacf(ts_rev_week_log_diff, nlags=10, method='ols')

#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_auto_corr)
plt.axhline(y=0,linestyle='--',color='black')
plt.axhline(y=-1.96/np.sqrt(len(ts_rev_week_log_diff)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(ts_rev_week_log_diff)),linestyle='--',color='black')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_par_auto_corr)
plt.axhline(y=0,linestyle='--',color='black')
plt.axhline(y=-1.96/np.sqrt(len(ts_rev_week_log_diff)),linestyle='--',color='black')
plt.axhline(y=1.96/np.sqrt(len(ts_rev_week_log_diff)),linestyle='--',color='black')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()


model = ARIMA(ts_rev_week_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)

model = ARIMA(ts_rev_week_log, order=(2, 1, 1))  
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_rev_week_log_diff.index.to_pydatetime(), ts_rev_week_log_diff.values)
plt.plot(ts_rev_week_log_diff.index.to_pydatetime(), results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_rev_week_log_diff)**2))


print(results_ARIMA.summary())
# plot residual errors
residuals = DataFrame(results_ARIMA.resid)
residuals.plot(kind='kde')
print(residuals.describe())


rev_predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print (rev_predictions_ARIMA_diff.head())


rev_predictions_ARIMA_diff_cumsum = rev_predictions_ARIMA_diff.cumsum()
rev_predictions_ARIMA_log = pd.Series(ts_rev_week_log.iloc[0], index=ts_rev_week_log.index)
rev_predictions_ARIMA_log = rev_predictions_ARIMA_log.add(rev_predictions_ARIMA_diff_cumsum,fill_value=0)



rev_predictions_ARIMA = np.exp(rev_predictions_ARIMA_log)
plt.plot(ts_rev_week.index.to_pydatetime(), ts_rev_week.values)
plt.plot(ts_rev_week.index.to_pydatetime(), rev_predictions_ARIMA.values)
plt.title('RMSE: %.4f'% np.sqrt(sum((rev_predictions_ARIMA-ts_rev_week)**2)/len(ts_rev_week)))


size = int(len(ts_rev_week_log) - 23)
train, test = ts_rev_week_log[0:size], ts_rev_week_log[size:len(ts_rev_week_log)]
historical = [x for x in train]
predictions = list()
predictweek=[]

print('Printing Predicted vs Expected Values...')
print('\n')
for t in range(len(test)):
    model = ARIMA(historical, order=(2,1,1))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    observed = test[t]
    historical.append(observed)
    predictweek.append(np.exp(yhat))
    print('Predicted Revernue = %f, Expected revernue = %f' % (np.exp(yhat), np.exp(observed)))

error = mean_squared_error(test, predictions)

#print('\n')
#print('Printing Mean Squared Error of Predictions...')
#print('Test MSE: %.6f' % error)

rev_predictions_series = pd.Series(predictions, index = test.index)

fig, ax = plt.subplots()
ax.set(title='Revernue by week', xlabel='Date', ylabel='Revernue')
ax.plot(ts_rev_week[-15:], 'o', label='observed')
ax.plot(np.exp(rev_predictions_series), 'g', label='rolling one-step out-of-sample forecast')
legend = ax.legend(loc='upper left')
legend.get_frame().set_facecolor('w')


#We need to 
forecast = model_fit.forecast(steps=14)[0]
result = [] 
for i in forecast:
    print(np.exp(i))
    result.append(np.exp(i)*7)
print(result)


#problem_1_output=pd.read_csv('problem-one-forecast-weeks.csv',header=None)
#problem_1_output['result']=result[1:]



#problem_1_output.to_csv('problem-one-answers.csv', header=False, index=False)


print ("Serializing metadata.....")
dump(model_fit, 'PredictRevenue.joblib')


#clf = load('PredictRevenue.joblib') 

