import pandas as pd

def display_predictors():
	hour_raw = pd.read_csv('data/hour.csv')
# add season_dummies
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
	hour_raw['higher_casual'] = [1 if hour_raw['casual'][x]/hour_raw['cnt'][x] >= .5 else 0 for \
		x in list(range(0,len(hour_raw)))] 


	logR_target = hour_raw['higher_casual']
	predictor_int_log = hour_raw.drop(['instant', 'dteday', 
		'season', 'yr', 'mnth', 'hr', 'weathersit', 'atemp', 
		'casual', 'registered', 'cnt', 'hr', 'sqr_wind', 'sqr_cnt', 
		'higher_casual'], axis=1)
	return predictor_int_log.columns
