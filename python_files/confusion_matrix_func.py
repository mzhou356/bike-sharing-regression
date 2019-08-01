# This file exists to construct confusion matrix for logistic regression
# Probability threshold for binary confusion matrix is adjusted manually here
# from .5 to .7 as sklearn has not built in function to do this.  
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def generate_conf_matrix(the_model, X_test, y_test):
	preds = np.where(the_model.predict_proba(X_test)[:,1] > .60, 1, 0)
	cm_df = pd.DataFrame(confusion_matrix(y_test, preds), index = ['Actual False', 'Actual True'],
             columns = ['Predict False', 'Predict True'])
	ax = sns.heatmap(cm_df, annot=True, cbar=False, fmt='d')
	plt.savefig('images/confusion_matrix.jpg')
	return ax