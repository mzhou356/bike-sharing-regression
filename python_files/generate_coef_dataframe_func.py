import numpy as np
import pandas as pd
import math

def generate_coef_df(model_log, X_train):
	log_reg_coefs = pd.DataFrame(X_train.columns, columns=['predictor'])
	log_reg_coefs['log_coefs'] = model_log.coef_.transpose()
	#log_reg_coefs['transformed_coefs'] = math.e**log_reg_coefs.log_coefs
	coef_df = log_reg_coefs.loc[(log_reg_coefs['log_coefs']>.7) ,:].sort_values(by='log_coefs', ascending=False)
	return coef_df