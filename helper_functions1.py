#!/usr/bin/env python
# coding: utf-8

# In[9]:


# import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import classification_report,roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.metrics import r2_score
from statsmodels.api import OLS
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose
import itertools


# In[6]:


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
    
    


# In[7]:


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
        RMSE = np.mean(np.sqrt((cross_val_score(linreg,x_cols, target,
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


# In[8]:


class UserLogReg():
    '''
    helper function for streamline user logistic regression
    '''

    def __init__(self, df, target):
        '''
        initialize the object with the user df and target value 
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

    def logreg(self, weight, X_train, y_train, X_test,y_test):
        '''
        inputs:
        weight: class weight for logreg 
        returns:
        y_score
        fpr,tpr, thresholds 
        also auc curve 
        '''
        logreg = LogisticRegression(C=0.03359818286283781,
                                    fit_intercept=True, class_weight=weight, max_iter=150, solver='lbfgs')
        model_log = logreg.fit(X_train, y_train)
        y_pred = model_log.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        AUC = auc(fpr, tpr)
        return model_log, y_pred, fpr, tpr, thresholds, AUC

    def smote_oversampling(self, X_train, y_train):
        '''
        use smote to strategically oversample the minority class
        '''
        # initialize smote
        smote = SMOTE()
        X_train_resampled, y_train_resampled = smote.fit_sample(
            X_train, y_train)
        return X_train_resampled, y_train_resampled

    def grid_search_parameters(self, X_train, y_train):
        '''
        perform grid search pipeline to find the best hyper paramater 
        returns the best model with the best setting

        '''
        pipe = Pipeline([('classifier', LogisticRegression())])
        param_grid = [{'classifier': [LogisticRegression()], 'classifier__penalty': ['l2'], 'classifier__C': np.logspace(-4, 4, 20), 'classifier__solver': ['liblinear', 'lbfgs']}
                      ]
        # Create grid search object
        clf = GridSearchCV(pipe, param_grid=param_grid,
                           cv=5, verbose=True, n_jobs=-1)
        # Fit on data
        best_clf = clf.fit(X_train, y_train)
        return best_clf

    def confusion_table(self, real_label, pred_label):
        '''
        inputs:
        real_label: an array of class values that are true 
        pred_label: an array of predicted class values
        returns:
        a confusion matrix table
        '''
        matrix = confusion_matrix(real_label, pred_label)
        df = pd.DataFrame(matrix, index=['True_0', 'True_1'], columns=[
                          'Pred_0', 'Pred_1'])
        return df

    def plot_auc_curve(self, true_y, pred_y, cutoff=0.5):
        '''
        inputs:
        true_y: array of true_y values
        pred_y: array of pred_y values 
        cutoffs: cutoff point for classificaiton
        returns:
        auc_curve and auc area 
        '''
        fpr, tpr, thresholds = roc_curve(true_y, pred_y)
        area = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve {cutoff}')
        plt.legend()
        return area

    def coef_results(self, final_model, X):
        '''
        display coef results

        '''
        coefs = final_model[0].coef_[0]
        features = X.columns
        coefs_df = pd.DataFrame(dict(zip(features, coefs)), index=[0]).T
        top_10 = coefs_df[0].abs().sort_values(ascending=False)[:10].reset_index().rename(
            columns={'index': 'features', 0: 'coef'})
        bottom_5 = coefs_df[0].abs().sort_values(ascending=False)[-5:].reset_index().rename(
            columns={'index': 'features', 0: 'coef'})
        return top_10, bottom_5


# In[ ]:


class TimeSeries():
    '''
    helper function for timeseries analysis
    '''

    def __init__(self, df, target):
        '''
        initialize the object with the user df and target value 
        df: pandas df 
        target: string 
        '''
        self.df = df
        self.target = target
        
    def create_timeseries(self):
        '''
        returns a dataframe with just time series component
        '''
        return self.df[[self.target]]
    
    def plot_time_monthly(self, plot = False):
        '''
        create monthly sum and generate a plot
        '''
        monthly_time = self.create_timeseries().resample('m').sum()
        if plot:
            monthly_time.plot()
        return monthly_time
    
    def generate_decomp(self):
        '''
        generate decompose plots
        
        '''
        decomposition_cnt = seasonal_decompose(self.plot_time_monthly(), freq=12)
        fig = plt.figure()
        fig = decomposition_cnt.plot()
        fig.set_size_inches(15, 8)
        
    def generate_train_test(self, df):
        '''
        generate test for future 10% data
        '''
        ind = int(len(df)*0.9)
        train, test = df[:ind], df[ind:]
        return train, test 
    
    def generate_combo(self):
        '''
        create p,d,q combo 
        '''
        # Define the p, d and q parameters to take any value between 0 and 2
        p = d = q = range(0,2)
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))
        # Generate all different combinations of seasonal p, q and q triplets (use 12 for frequency)
        pdqs = [(x[0], x[1], x[2], 12) for x in pdq]
        return pdq, pdqs
    
    def model_parameter_tuning(self,combo,combos):
        '''
        find best pdq
        
        '''
        ans = []
        for comb in pdq:
            for combs in pdqs:
                try:
                    mod = sm.tsa.statespace.SARIMAX(train,
                                            order=comb,
                                            seasonal_order=combs,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

                    output = mod.fit()
                    ans.append([comb, combs, output.aic])
#                     print('ARIMA {} x {}12 : AIC Calculated ={}'.format(comb, combs, output.aic))
                except:
                    continue
        ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
        return ans_df.loc[ans_df['aic'].idxmin()]
    
    def sarimax_model(self, train, pdq, pdqs):
        '''
        create sarimax model with best pdq, pdqs
        
        '''
        ARIMA_MODEL = sm.tsa.statespace.SARIMAX(train,
                                order=pdq,
                                seasonal_order=pdqs,
                                enforce_stationarity=False,
                                enforce_invertibility=False)
        output = ARIMA_MODEL.fit()
        return output
    
    def forecast(self, model, test):
        '''
        create forecastead result and append to test table 
        '''
        # Get forecast
        pred = model.forecast(steps = len(test))
        test['forecasted'] = pred.tolist()
        return test 
    
    def plot_finalresults(self, test, train):
        '''
        plot final result with test and train 
        
        '''
        plt.plot(test.cnt.resample('m').sum().index,
         test.cnt.resample('m').sum(), label='cnt_test')
        plt.plot(test.forecasted.resample('m').sum().index,
         test.forecasted.resample('m').sum(), label = 'forecast')
        plt.plot(train.resample('m').sum().index, train.resample('m').sum(), label = 'train')
        plt.legend()
        plt.xticks(rotation = '80')
        plt.show()

