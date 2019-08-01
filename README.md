# Predicting Ridership DC Capital Bikeshare

![](images/capital_bikes.jpeg)

* 06-03-2019 Flatiron Data Science Fellowship Module 4 group project 

Data Source: https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset 
  *  This dataset contains the hourly and daily count of rental bikes between years 2011 and 2012 in Capital bikeshare system with the corresponding weather and seasonal information 
      * Original Source: http://capitalbikeshare.com/system-data 
      * Weather Information: http://www.freemeteo.com 
      * Holiday Schedule: http://dchr.dc.gov/page/holiday-schedule
  * 17379 rows and 15 features
      
 ## Members
- Mindy Zhou
- Joe McAllister

## Business Insights:
- Top 10 features for increasing or decreasing DC total bike rentals 
   - Method: Linear Regression
   - StatsModels 
   - sklearn RidgeCV, LassoCV, LinearRegressionCV
   - target variable: total bike rentals (cnt)
- Top 10 features for more casual users than registered users
  - casual users: daily, weekly, or monthly pass (one time use)
  - registered users: daykey, monthly, yearly membership (recurring payment) 
  - Method: Logistic Regression
  - sklearn logistic regression 
  - target variable: 1 for more casual users than registered users otherwise 0 
- Time Series Analysis: 
  - Monthly trend 
  - Seasonality 
  - Forecast model for future prediction 
  - StatsModel SARIMAX 
  - target variable: total bike rentals (cnt)
  
## Results: 
- Linear Regression on total bike rentals:

![](images/coefs.png)


  
 Google slide: https://docs.google.com/presentation/d/1qnJeEjhnirk8CnebKaHb_3ECITmTdtwuEuynRNlAu_A/edit#slide=id.g5e50a81f87_0_96

  
