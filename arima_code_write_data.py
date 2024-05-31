

def random_walk_calc(positions):

	import random
	import numpy as np
	import matplotlib.pyplot as plt
 
# Probability to move up or down
	prob = [0.05, 0.95]  
 
# statically defining the starting position
#	start = 2 
#	positions = [start]
 
# creating the random points
	rr = np.random.random(1000)
	downp = rr < prob[0]
	upp = rr > prob[1]
 
 
	for idownp, iupp in zip(downp, upp):
    		down = idownp and positions[-1] > 1
    		up = iupp and positions[-1] < 4
    		positions.append(positions[-1] - down + up)
 
# plotting down the graph of the random walk in 1D
	#plt.plot(positions)
	#plt.show()
	
def function_input_for_arima(positions):

	white_noise=np.zeros(1000)
	for i in range(0,1000):
   	 white_noise[i]=positions[i]#np.sin(i*0.6)+0.1*random.randint(0,10)

#white_noise = np.random.randn(100)
	dataset=pd.DataFrame(white_noise,columns=["Acceleration"])
	

	#print(dataset)
	
	return dataset
	
def generic_function_input():

	white_noise=np.zeros(100)
	for i in range(0,100):
   	 white_noise[i]=np.sin(i*0.6)#+0.1*random.randint(0,10)

#white_noise = np.random.randn(100)
	dataset=pd.DataFrame(white_noise,columns=["Acceleration"])
	

	#print(dataset)
	
	return dataset
	
def statistics_before_arima(indexedDataset):

	#indexedDataset = dataset
	rolmean = indexedDataset.rolling(window=10).mean() #window size 12 denotes 12 months, giving rolling mean at yearly level
	rolstd = indexedDataset.rolling(window=10).std()
	print(rolmean,rolstd)
	
def autocor_and_partial_autocor(indexedDataset):
	lag_acf = acf(indexedDataset, nlags=20)
	lag_pacf = pacf(indexedDataset, nlags=20, method='ols')

#Plot ACF:
	plt.subplot(121)
	plt.plot(lag_acf)
	plt.axhline(y=0, linestyle='--', color='gray')
	plt.axhline(y=-1.96/np.sqrt(len(indexedDataset)), linestyle='--', color='gray')
	plt.axhline(y=1.96/np.sqrt(len(indexedDataset)), linestyle='--', color='gray')
	plt.title('Autocorrelation Function')            

#Plot PACF
	plt.subplot(122)
	plt.plot(lag_pacf)
	plt.axhline(y=0, linestyle='--', color='gray')
	plt.axhline(y=-1.96/np.sqrt(len(indexedDataset)), linestyle='--', color='gray')
	plt.axhline(y=1.96/np.sqrt(len(indexedDataset)), linestyle='--', color='gray')
	plt.title('Partial Autocorrelation Function')
            
	plt.tight_layout()  
	plt.show()
	
def pmdarima_model(indexedDataset):
	import pmdarima as pm

# datasetLogDiffShifting is your time series data
	auto_model = pm.auto_arima(indexedDataset, seasonal=True, stepwise=True, suppress_warnings=True)

# Print the optimal order (p, d, q)
	print("Optimal (p, d, q) order:", auto_model.order)
	
	return auto_model.order
	
def arima_model_train(p,d,q,N_model,indexedDataset):

	from statsmodels.graphics.tsaplots import plot_predict

#	p=10
#	d=0
#	q=2
	model = ARIMA(indexedDataset[:N_model], order=(p,d,q))
#	model = ARIMA(indexedDataset, order=(0,1,0))
	results_ARIMA = model.fit()

	print(results_ARIMA.summary())

	plt.plot(indexedDataset,'-x')
	plt.plot(results_ARIMA.fittedvalues, '-',color='red')
	plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - indexedDataset['Acceleration'][:N_model])**2))
	#plt.title('RSS: %.4f'%sum((results_ARIMA.fittedvalues - indexedDataset['Acceleration'])**2))
	fig, ax = plt.subplots()
	ax = indexedDataset.plot(ax=ax)
	plot_predict(results_ARIMA,N_model+1,N_end,ax=ax,dynamic=True)
	plt.show()
	print('Plotting ARIMA Optimum model')
	df1 = pd.DataFrame(results_ARIMA.fittedvalues)
	return df1

#	fig, ax = plt.subplots(1,2)
#	residuals.plot(title="Residuals", ax=ax[0])
#	residuals.plot(kind='kde', title='Density', ax=ax[1])
#	plt.show()
	
def arima_model_predict(p,d,q,N_model,N_end,indexedDataset):
	#from statsmodels.graphics.tsaplots import plot_predict
	from datetime import datetime
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	import statsmodels.api as sm
	from statsmodels.tsa.arima.model import ARIMA
	from matplotlib.pylab import rcParams
	from statsmodels.graphics.tsaplots import plot_predict

#	p=10
#	d=0
#	q=2

#	train=indexedDataset[:N_model]
	train, test = indexedDataset[0:N_model-1], indexedDataset[N_model:N_end]

	#results_ARIMA=sm.tsa.ARIMA(train, order=(0,1,0)).fit()
	results_ARIMA=ARIMA(train, order=(p,d,q)).fit()

#model = sm.tsa.ARIMA(indexedDataset, order=(2,2,0)).fit()
#results_ARIMA = model.fit()
#We have 32618(existing data) data points. 
#And we want to forecast for additional 10000 data points.
#plt.plot(indexedDataset)
#plt.plot(results_ARIMA.fittedvalues, color='red')
#plot_predict(100,110,dynamic=True,ax=results_ARIMA) 
	fig, ax = plt.subplots()
	ax = indexedDataset.plot(ax=ax)
#plot_predict(res, '1990', '2012', ax=ax)
#plt.show()
	plot_predict(results_ARIMA,N_model,N_end,ax=ax,dynamic=True)
	plt.show()
#x=results_ARIMA.forecast(steps=10)
#x
#model.fit.plot_predict

def calc_error(p,d,q):
	from statsmodels.tsa.arima.model import ARIMA
	from math import sqrt
	from sklearn.metrics import mean_squared_error
	import warnings
	from statsmodels.tools.sm_exceptions import ConvergenceWarning
	warnings.simplefilter('ignore', ConvergenceWarning)
	def parser(x):
		return datetime.strptime('190'+x, '%Y-%m')
#dataset.index = dataset.index.to_period('M')
# split into train and test sets
	X = indexedDataset.values
	#size = int(len(X) * 0.66)
	train, test = X[0:N_model-1], X[N_model:N_end]
	history = [x for x in train]
	predictions = list()
	
# walk-forward validation

	for t in range(len(test)):
 		model = ARIMA(history, order=(p,d,q))
 		model_fit = model.fit()
 		output = model_fit.forecast()
 		yhat = output[0]
 		predictions.append(yhat)
 		                
 		obs = test[t]
 		history.append(obs)
 		print('predicted=%f, expected=%f' % (yhat, obs))
        
# evaluate forecasts
	rmse = sqrt(mean_squared_error(test, predictions))
	df2=pd.DataFrame(predictions)
	
	print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
	plt.plot(test)
	plt.plot(predictions, color='red')
	plt.show()
	return df2	
if __name__ == "__main__":

	#Import Libraries 
	from datetime import datetime
	import numpy as np             #for numerical computations like log,exp,sqrt etc
	import pandas as pd            #for reading & storing data, pre-processing
	import matplotlib.pylab as plt #for visualization
	import random
#for making sure matplotlib plots are generated in Jupyter notebook itself
#matplotlib inline             
	from statsmodels.tsa.stattools import adfuller
	from statsmodels.tsa.stattools import acf, pacf
	from statsmodels.tsa.seasonal import seasonal_decompose
	from statsmodels.tsa.arima.model import ARIMA
	from matplotlib.pylab import rcParams
	import time
	start_time=time.time()
	rcParams['figure.figsize'] = 10, 6
	start = 2 
	N_model=6599
	N_end=9999
	positions = [start]
#	random_walk_calc(positions)
#	function_input_for_arima(positions)
#	generic_function_input()
	dataset=np.loadtxt("accx_80.dat")
	data_pd=pd.DataFrame(dataset,columns=["time","Acceleration"])
	data_pd2=data_pd["Acceleration"]
	indexedDataset=pd.DataFrame(data_pd2,columns=["Acceleration"])
	statistics_before_arima(indexedDataset)
	autocor_and_partial_autocor(indexedDataset)
	[p,d,q]=pmdarima_model(indexedDataset)
	p=8
	d=0
	q=1
#	arima_model_train(p,d,q,N_model,indexedDataset)
#	arima_model_predict(p,d,q,N_model,N_end,indexedDataset)
#	calc_error(p,d,q)
	y1=arima_model_train(p,d,q,N_model,indexedDataset)
	y2=calc_error(p,d,q)
	df3=pd.concat([y1, y2], ignore_index = True)
	df3.reset_index()
	
	f2=open("fitted_data.txt","w")
	f2.write(str(df3.to_string()))
	f2.close()
	end_time=time.time()
	elapsed_time=end_time-start_time
	print(elapsed_time)

