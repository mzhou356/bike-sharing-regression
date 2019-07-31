#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.metrics import r2_score
from statsmodels.api import OLS
import statsmodels.api as sm


# In[2]:


# change column value to understandable string 
class DataCleaning():
    '''
    contains functions to clean and prepare the dc dataframe 
    '''
    
    def change_colvalue(self, df, colname, colvaluemap):
        '''
        inputs: 
        df: a pandas dataframe with the colnames needed to be changed 
        colname: the colname for df needed to be changed 
        colnamemap: a dictionary with original colvalue as key and
                    desired colvalue as value 
        returns:
        a list with the desired colvalue
        '''
        changed_name = df[colname].map(colvaluemap)
        return changed_name
    
    def convert_to_prop(self, df, colnames):
        '''
        inputs:
        df: pandas dataframe 
        colnames: a list of colnames for calculation proportions
                  order of names: subset1, subset2, total 
        returns:
        a dataframe with two new colnames with proportions 
        '''
        props_subset1 = []
        props_subset2 = []
        for i, row in df.iterrows():
            props_subset1.append(row[colnames[0]]/row[colnames[2]])
            props_subset2.append(row[colnames[1]]/row[colnames[2]])
        return pd.DataFrame({colnames[0]: props_subset1,colnames[1]: props_subset2})
    
    


# In[3]:


# create a class for multilinear regress
class BikeLinearRegression():
    '''
    helper function for streamline bike linear regression
    '''

    def __init__(self, df, target):
        '''
        initialize the object with the biker df and target value 
        df: pandas df 
        target: string 
        '''
        self.df = df
        self.target = target

    def target_features(self):
        '''
        split the dataframe into outcome and features

        '''
        outcome = self.df[self.target]
        features = self.df.drop(columns=[self.target])
        return outcome, features

    def stats_models(self, X_train, y_train, show_summary=False):
        '''
        perform OLS from stats model 
        return model results
        '''
        X = sm.add_constant(X_train)
        model_stats = OLS(y_train, X)
        results_stats = model_stats.fit()
        if show_summary:
            results_stats.summary()
        return results_stats

    def stats_pred(self, X_train, y_train, X_test, y_test):
        '''
        perform stats model
        perform prediction and return r2 score 
        '''
        results_stats = self.stats_models(X_train, y_train)
        X_pred = sm.add_constant(X_test)
        y_pred_stats = results_stats.predict(X_pred)
        return r2_score(y_test, y_pred_stats)

    def stats_top_bottom_features(self, X_train, y_train, colname):
        '''
        return top 10 features with coef datafarme according to coef magnitude
        bottom 5 features 
        '''
        results_stats = self.stats_models(X_train, y_train)
        coefs_df = pd.DataFrame(results_stats.params).rename(
            columns={0: colname})
        top_10_features = coefs_df[colname].abs().sort_values(
            ascending=False).index[1:11].values
        bottom_5_features = coefs_df[colname].abs().sort_values(
            ascending=False).index[-5:].values
        return coefs_df.loc[top_10_features], coefs_df.loc[bottom_5_features]

    def normality_homoscedasticity(self, X_train, y_train):
        '''
        check for normality and homosecdesticty for resid 
        '''
        results_stats = self.stats_models(X_train, y_train)
        # normality test
        sns.distplot(results_stats.resid)
        plt.show()
        # check for homoscedasticy
        X = sm.add_constant(X_train)
        sns.scatterplot(results_stats.predict(X), results_stats.resid)
        sns.lineplot(results_stats.predict(X), [
                     0 for i in range(len(X))], color='red')

    def ridge_cv(self, X_train, y_train, X_test, y_test):
        '''
        perform cross validation for ridge regression
        print results 
        return dataframe with top 10 features
        bottom 5 features
        '''
        # ridge regression , make sure best alpha in alphas
        regr_cv = RidgeCV(cv=10, alphas=np.linspace(0.1, 0.5, 10))
        model_cv = regr_cv.fit(X_train, y_train)  # cv on training set
        print('best lambda:', model_cv.alpha_)
        y_ridge_train = regr_cv.predict(X_train)
        y_ridge_test = regr_cv.predict(X_test)
        print('ridge_train:', r2_score(y_train, y_ridge_train))
        print('ridge_test:', r2_score(y_test, y_ridge_test))
        r_coef_df = pd.DataFrame(
            {'cols': self.target_features()[1].columns, 'coef_ridge': regr_cv.coef_})
        top_10_features_ridge = r_coef_df.coef_ridge.abs(
        ).sort_values(ascending=False).index[:10].values
        bottom_5_ridge = r_coef_df.coef_ridge.abs(
        ).sort_values(ascending=False).index[-5:].values
        return r_coef_df.loc[top_10_features_ridge], r_coef_df.loc[bottom_5_ridge]

    def lasso_cv(self, X_train, y_train, X_test, y_test):
        '''
        perform cross validation for lasso regression
        print results 
        return dataframe with top 10 features
        bottom 5 features 
        '''
        # lasso regression , make sure alphas
        regl_cv = LassoCV(cv=10, alphas=np.linspace(1e-04, 1e-06, 10))
        modell_cv = regl_cv.fit(X_train, y_train)  # cross val on training set
        print(modell_cv.alpha_)
        y_lasso_train = regl_cv.predict(X_train)
        y_lasso_test = regl_cv.predict(X_test)
        print('lasso_train:', r2_score(y_train, y_lasso_train))
        print('lasso_test:', r2_score(y_test, y_lasso_test))
        print('features removed:', sum(regl_cv.coef_ == 0))
        l_coef_df = pd.DataFrame(
            {'cols': self.target_features()[1].columns, 'coef_lasso': regl_cv.coef_})
        top_10_features_lasso = l_coef_df.coef_lasso.abs(
        ).sort_values(ascending=False).index[:10].values
        bottom_5_lasso = l_coef_df.coef_lasso.abs(
        ).sort_values(ascending=False).index[-5:].values
        return l_coef_df.loc[top_10_features_lasso], l_coef_df.loc[bottom_5_lasso]

    def linear_cv(self):
        '''
        perform cross validation on linear regression without regularization
        return MAE, RMSE, R2 scores over 10 fold CV

        '''
        # cross validation on linear regression model
        linreg = LinearRegression()
        # use 3 metrics mean
        # absolute error (mae)
        target, x_cols = self.target_features()
        MAE = np.mean(cross_val_score(linreg, x_cols, target,
                                      cv=10, scoring='neg_mean_absolute_error')*-1)
        print('MAE:', MAE)
        # root mean squared error (RMSE)
        RMSE = np.mean(np.sqrt((cross_val_score(linreg, x_cols, target,
                                                cv=10, scoring='neg_mean_squared_error')*-1)))
        print('RMSE:', RMSE)
        # r2 score
        r2 = np.mean(cross_val_score(
            linreg, x_cols, target, cv=10, scoring='r2'))
        print('R2:', r2)
        result_df = pd.DataFrame({'MAE': [MAE], 'RMSE': [RMSE], 'r2': [r2]})
        return result_df

    def merge_feature_dfs(self, dfs, plot = False):
        '''
        dfs: a list of dataframes
        merge features with coef into one table 
        create bar plot 
        '''
        merged_df = pd.concat(dfs, axis=1)
        if plot:
            merged_df.plot.bar(subplots=False, sharex=True)
        return merged_df


# In[ ]:




