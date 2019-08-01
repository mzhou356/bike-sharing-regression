# This script loads raw data, builds predictor dataset
# runs logistic regression model and outputs ROC curve.  

#load data

hour_raw = pd.read_csv('data/hour.csv')

season_dummies = pd.get_dummies(data=hour_raw.season, prefix='season')
weathersit_dummies=pd.get_dummies(data=hour_raw.weathersit, prefix='weathersit')
weekday_dummies = pd.get_dummies(data=hour_raw.weekday, prefix='weekday')
hr_dummies = pd.get_dummies(data=hour_raw.hr, prefix='hr')

hour_raw = pd.concat([hour_raw, season_dummies, weathersit_dummies, weekday_dummies, hr_dummies], axis=1)
hour_raw.drop(columns=['season_1', 'weathersit_4', 'weekday_6','hr_0'], inplace = True)

# modify wind/cnt to normalize
hour_raw['sqr_wind'] = hour_raw.windspeed**(1/2)
hour_raw['sqr_cnt'] = hour_raw.cnt**(1/5)

#create target var for Log Reg (more casual or not)
hour_raw['higher_casual'] = [1 if hour_raw['casual'][x]/hour_raw['cnt'][x] >= .5 else 0 for x in list(range(0,len(hour_raw)))] 


logR_target = hour_raw['higher_casual']
predictor_int_log = hour_raw.drop(['instant', 'dteday', 'season', 'yr', 
                                   'mnth', 'hr', 'weathersit', 'atemp', 'casual', 
                                   'registered', 'cnt', 'hr', 'sqr_wind', 'sqr_cnt', 
                                   'higher_casual'], axis=1)


X_train, X_test, y_train, y_test = train_test_split(predictor_int_log, logR_target, test_size = 0.3, random_state=79703)

smote = SMOTE(random_state=2927)
X_train_resampled, y_train_resampled = smote.fit_sample(X_train, y_train) 
logreg = LogisticRegression(fit_intercept = True, max_iter=400, C = .01, solver ='lbfgs', random_state=1214) #Starter code
model_log = logreg.fit(X_train_resampled, y_train_resampled)

y_hat_test = logreg.predict(X_test)
y_score = model_log.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
    
print('AUC for {}: {}'.format('balanced', auc(fpr, tpr)))
lw = 2

fig = plt.figure(figsize=(7,6))
plt.plot(fpr, tpr, color='red', lw=lw, label='ROC curve {}'.format('balanced'))
plt.title('ROC Curve')
plt.xlabel('False Positive')
plt.ylabel('True Positive')

plt.savefig('images/log_roc_curve.jpg')
