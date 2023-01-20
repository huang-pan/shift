TBD: model not winners, no expected_margin
    
    
The used car segment which generates 52 billion revenue yields just 0.03 billion or 1%, although it has a potential of 3%.

- Support Vector Machine regression against expected_margin, calculated_margin; use hypergrid parameter search; don't do, hard to put into production?
- http://www.svm-tutorial.com/2014/10/support-vector-regression-r/
- Implement genetic algo for hyperparameter search later

SVM is a machine learning method that has been developed to work with data sets that are typically high dimensional and sparse
(data set contains a small number of the training data pairs). Nonetheless, before using SVR to predict an outcome, one should
select the suitable kernel and hyperparameters rst. Therefore, an intermediary validation step is needed before the test phase.
The use of independent multinomial variables in regression analysis and SVR is not justiable, since they can cause a non-linear eect.
Therefore, the data in this format need to be converted into several dummy binary variables before they can be used in linear regression.
Normalizing here means scaling the values of those continuous variables by their minimum and maximum, to make all elements lie in
a range of [0, 1], similar to those binary variables' range. 70% training data 30% validation data Cost values are set to
2􀀀6; 2􀀀5;    ; 215, and epsilon values to 2􀀀9; 2􀀀8;    ; 29 For SVR with linear, polynomial, and RBF kernel, only a coarse gr
id search will be conducted, the minimum training set sizes are 27,500 for linear SVR, 22,000 for polynomial SVR degree 2 and 32,500 for
RBF SVR. linear best performance.

- expected value: some sort of Monte Carlo simulation

%md ### Plan of Action
#### Keeping the end Business Goals in mind
So what are we exactly trying to do here? The end goal is to increase profits for the company.<br><br>
How do we do that? There are many ways to do it. 

- The point of Exploratory Data Analysis is to prioritize which areas to target and maximize usage of available resources.
- The point of Predictive Modeling is to generate a statistical model based on all available data that maximizes profit for the company. 

#### Price Model Testing Framework:

metrics: average expected_margin, conversion rate, other metrics: number of inquiries, number sold --> calculate profit
localization: test different pricing models out by sector or zip code in region

price model calculates initial price:
0. Current pricing model, arbitrary, based on current market rates, hence the percent_to_market: this may not be the best way to price cars, too random, too time consuming, are there underlying factors that give better predictions of price? Need something that will statistically maximize profits to the company. quant models are a lot less labor intensive. This is why cross validation is important for quant models, look at RMSE of each model.
 Initial pricing: compare against similar vehicles for sale in area on ebay, craigslist, compile cohort analysis of prices: arb opps: High, average, low (post mechanic looked at): 
- Market price: Buy data or scrape websites for used car prices: Kelly Blue Book, Edmunds, NADA, Ebay, Craigslist, etc.
- Need better data: what is the market rate for the make/model/year/mileage? KBB, edmunds? What is the price the car actually sells for?
1. Simple Average Model: delta percent * market_rate, market rate pulled from ebay, craigslist, dealers, KBB, edmunds? market_rate model?
2. Linear Regression based on car feature set, evaluate model based on RMSE
3. Support Vector Regression based on car feature set, evaluate model based on RMSE
4. Other models

A. compare model performance, keep alpha and beta the same, choose best performing model
B. once best model has been found, find optimal alpha and beta, keep on updating model

- Look at entire system as closed system with feedback loop
- Current model: find market price, list below market price
- Improved model: generate initial list price based on predictive model, initial list price must be below market price, adaptive model over time, lower price over time, adjust price in response to similar vehicles listed
- Overall system metrics, empirical measurements over time: expected_margin x conversion rate = total profit: simulate different scenarios, adjust models parameters to arrive at max total
Current model + predictive model + feedback (num_days_on_market)
- percent_to_market?
- Have in input to the predictive model that takes into account average number of days to sell for Shift for the particular make/model/year/mileage, etc. Cost function, feedback loop.

Data Exploration, descriptive analytics: increase profit
1. Increase the conversion rate of Shift inquiries to Shift sellers.<br>
2. Increase the number of Shift inquiries.<br>
3. Don't do deals on cars with negative expected margin<br>
4. increase average expected margin

Initial pricing: from market.
Improved pricing: from model, with feedback from market (sell rate, number of days to sell), 
EDA, increase conversion rate: target both buyers and sellers: ID feature set with highest EM in particular market, focus on those

**NOTE**: In this notebook, let's refer to Expected Margin as what we want to calculate. Let's refer to expected_margin as the data that has been given to us - the non-zero margin of cars that have been sold by Shift.

We can do this by:
- Taking expected_margin, let us create a predictive model that determines the Expected Margin for all cars that we don't have expected_margin for.
- From this predictive model, we can A. Determine the cars with the highest Expected Margin -
- From descriptive statistics and predictive modeling, we can B. Determine which set of features (make/model/year/location, target market, etc.) has the strongest effect on Expected Margin.
- 2. Using data exploration and descriptive statistics we can identify the features that have the highest impact on expected_margin. We can rank features by importance. Target cars with highest return features.
- 1. We can increase the conversion rate of Shift inquiries to Shift sellers by A. focusing outreach on the inquiries with the highest Expected Margins. B. focusing outreach on the inquiries of cars with features that have the strongest effect on Expected Margin.
- 2. Likewise, we can increase the number of Shift inquiries by marketing to A. owners with cars that have the highest Expected Margin. B. owners with cars with features that have the strongest effect on Expected Margin.

- Create predictive model for Expected Margin and calculate RMSE of model, Conversion Rate by location & feature set. Feedback loop from RMSE, conversion rate to Expected Margin predictive model.
- Compare Expected Price with average market price

predictive model: 1. price that max imized profit 2. any particular features we should focus on?

- cohort analysis Break analysis into regions first: region_short_code/location_zip, sort by make/model/year/mileage/similar_vehicles_listed/percent_to_market/market_average_days_to_sell to expected_margin/margin. track over time, repeat biz, use information about existing sellers & buyers to market to similar sellers & buyers to increase inquiry rate
- E.g. which brand/make/model has the highest conversion rate?

- ceiling analysis: lowest hanging fruit first, get most bang for your buck: simple percent --> linear regress -> random forests --> neural networks; segmentation

- Predict missing Expected Margin values, find out what features have the highest effect on Expected Margin. Target sellers with cars with features that have highest expected_margin to raise overall profit


**Feedback Loop**<br> 
- 1. update model: expected_margin, conversion_rate: collect data, update model, compare to RMSE of model
- 2. pricing cost function: shift_num/avg_days_to_sell: lower price until min if num_days_to_sell goes up, collect data
- percent_to_market
- 3. total system metric: conversion_rate: collect data, calculate

------

%md ### Plan of Action

**NOTE**: In this notebook, let's refer to Expected Margin as what we want to calculate. Let's refer to expected_margin as the sales data that has been given to us - the non-zero margin of cars that have been sold by Shift.

In addition, normally we would want to model price. However, in Shift's case, we are modeling Expected Margin = final sales price - the wholesale price. 
Since the wholesale price is a predetermined, fixed quantity - a constant, we will end up modeling final sales price in the form of Expected Margin.

#### Keeping the end Business Goals in mind
So what are we exactly trying to do here? The end goal is to increase profits for the company by using all available data.<br><br>
How do we do that? There are many ways.<br>
For example, from the above, we can use statistics to figure out ways to increase the number of inquiries.
We can also use predictive pricing models to figure out ways to best increase the average Expected Margin and conversion rate.<br>

#### Exploratory Data Analysis
The point of Exploratory Data Analysis (EDA) is to analyze descriptive statistics of the data in order to prioritize which areas of the business to target. 
We want to maximize usage of available resources.<br>

For example, through EDA we can increase profits by figuring out which locations Shift should focus on.
Through outreach campaigns to sellers at these locations, we can increase the number of Shift inquiries.
Through marketing campaigns to buyers at these locations, we can increase the conversion rate.<br>

#### Predictive Models
The point of Predictive Modeling is to generate a pricing model that statistically maximizes profit for the company. This statistical model has 3 uses:<br>

1. Be a better, more consistent predictor of price than arbitrarily picking a percent_to_market price. 
Through this pricing model, we can identify arbitrage opportunities where cars are underpriced with respect to the market. 
We can also identify cars that are overpriced, and avoid those cars.
By focusing outreach on underpriced cars and avoiding overpriced ones, we can increase the average Expected Margin and total profit for the company.<br>

2. Predict the Expected Margin for cars that we don't have expected_margin data for. Help guide us on how to price those cars in order to maximize profit.<br>

3. From this pricing model, we can also determine if there any underlying factors/features that have a large effect on price. 
If so, we can use this information to help maximize profits for the company. 
We can target cars with these features during outreach/marketing campaigns that increase the conversion rate and number of inquiries.
This information will help us maximize profits.<br>

With any statistical model, there's a trade off between complexity and effectiveness. Some models may have the best performance, but may be too complex (optimal parameters too hard to find) to be put into production.
For our purposes, let's start with the simplest model first, and then try more complex models:

1. The simplest model we can try is the Percent to Market Model (PMM). 
This model is somewhat arbitrary. Let us determine the market rate for the car and multiple it by a percentage < 100% to arrive at a final sales price.
This may not be the best way to price cars as: A. it is very labor intensive and time consuming to gather the market rate for each make/model/year/mileage out there.
The process may also be very error prone. B. Results from pricing cars this way may be very random.
Still, this is a simple technique and may be worth evaluating.
2. Quantitative models may be a lot less labor intensive and should give more consistent results. 
For our first quantitative model, let us try multivariate linear regression based on car feature set.
3. Finally, we can try more sophisticated quantitative models like Support Vector Regression, Neural Networks, etc.

To evaluate and compare each model, the metric that we use is the Root Mean Square Error (RMSE) of the Expected Margin generated from the model and the expected_margin of the car actually sold.
The goal is to use the pricing model with the smallest RMSE. By minimizing RMSE, we can minimize any negative Expected Margin.<br>
In general, we should try to use the simplest model possible that gives us the desired results that we want. The simpler the model, the fewer sources of error in our calculations.

#### Empirical Testing Framework:
Once we have these pricing models, we need a framework to test them.<br>

Given a pricing model, let us look at metrics of how well the pricing model works in the field.
The most important metrics are expected_margin, the number of inquiries, and the number of cars sold through Shift.
We can use these metrics to calculate total profit, total profit lost, and the conversion rate.
Another useful metric would be the number of days it takes to sell a car.<br>

For the price testing framework, we can test different pricing models out by region or zip code.
In each zip code we can try out a different pricing model to see which one performs the best empirically according to the metrics above.
We can also use Cohort Analysis and break the analyses into further subsets, e.g. by Make/Model/Year/Mileage, etc. E.g. for each zip code, which make/model has the highest Expected Margin? The highest conversion rate?

We can also track the performance over time. If we look at the entire system as closed system with feedback loop, we can adjust model parameters over time to achieve optimal performance/maximum profit.<br>

-----

- For the avoidance of multicollinearity, implementing LASSO regression is not a bad idea.
- https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models
- xboost/ridge/LASSO https://www.kaggle.com/zoupet/house-prices-advanced-regression-techniques/xgboost-ridge-lasso
- numerical vs categorical features, L1/L2/ElasticNet/Xboost regression: https://www.kaggle.com/juliencs/house-prices-advanced-regression-techniques/a-study-on-regression-applied-to-the-ames-dataset http://www.datasciencecentral.com/profiles/blogs/predicting-car-prices-part-1-linear-regression
- multifactor linear regression using current expected_values to extrapolate other expected_values, collinearity, LASSO, ridge regularization

-----
pyspark reference doc: http://spark.apache.org/docs/latest/api/python/pyspark.ml.html

Shift: mleap
https://github.com/TrueCar/mleap-demo/blob/master/notebooks/MLeap%20-%20Train%20Airbnb%20Price%20Prediction%20Model%20-%20Spark%20Saturday.ipynb

- feature exploration: correlation, scatterplots with LOESS to ID most correlated features, compare aginst current exploratory data analysis, linear regression look at T-squared
- Correlation of factors with expected_margin; correlation heatmap scatter plots of most highly correlated features - see a gentel intro to apache spark for data scientists
- regression on each factor, then MV regression: measure correlation/coliinearity of each factor; correlation heatmap of all factors; scatterplot/regression of all factors with high correlation, box plot of features against expected_margin
http://spark.apache.org/docs/latest/ml-pipeline.html#example-pipeline
http://spark.apache.org/docs/latest/ml-tuning.html
https://github.com/apache/spark/blob/master/examples/src/main/python/ml/linear_regression_with_elastic_net.py
https://databricks.com/product/getting-started-guide/machine-learning
http://go.databricks.com/hubfs/notebooks/Pop._vs._Price_LR.html
https://dbc-b2fbdc71-aa1b.cloud.databricks.com/#externalnotebook/https%3A%2F%2Fdocs.cloud.databricks.com%2Fdocs%2Flatest%2Fdatabricks_guide%2Findex.html%2305%2520MLlib%2F2%2520Algorithms%2F5%2520Linear%2520Regression%2520Pipeline.html
https://github.com/apache/spark/blob/master/examples/src/main/python/ml/random_forest_classifier_example.py
https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-regression
http://stackoverflow.com/questions/32769573/cross-validation-of-random-forest-model-with-spark

trainingvalidationsplit
https://dbc-b2fbdc71-aa1b.cloud.databricks.com/#externalnotebook/https%3A%2F%2Fdocs.cloud.databricks.com%2Fdocs%2Flatest%2Fdatabricks_guide%2Findex.html%2305%2520MLlib%2F2%2520Algorithms%2F5%2520Linear%2520Regression%2520Pipeline.html

cross validation
https://docs.cloud.databricks.com/docs/latest/sample_applications/index.html#07%20Sample%20ML/MLPipeline%20Bike%20Dataset.html
https://docs.cloud.databricks.com/docs/latest/sample_applications/index.html#07%20Sample%20ML/MLPipeline%20Newsgroup%20Dataset.html
https://dbc-b2fbdc71-aa1b.cloud.databricks.com/#externalnotebook/https%3A%2F%2Fdocs.cloud.databricks.com%2Fdocs%2Flatest%2Fsample_applications%2Findex.html%2302%2520Gentle%2520Introduction%2FApache%2520Spark%2520on%2520Databricks%2520for%2520Data%2520Scientists%2520(Scala).html

-----

Works in Databricks Spark 2.0

# Imports
import pandas as pd
import numpy as np
import sklearn as sk
#from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer
from scipy.stats import skew
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.mllib.regression
#import xgboost as xgb

-----

import csv
import urllib2
url = 'https://raw.githubusercontent.com/huang-pan/Spark/master/training_data.csv'
response = urllib2.urlopen(url)
cr = csv.reader(response)

# read csv into list of lists
dataList = []
for i, row in enumerate(cr):
  if i < 5: print row
  if (row[0] != 'quote_id'):
    dataList.append(row[1:])
    
# can also read into pandas dataframe, but use Spark dataframe for now
