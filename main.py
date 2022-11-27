import pandas as pd
from sklearn import metrics
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('card_transdata.csv')
s = data['fraud'].value_counts()

X=data[['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price', 'repeat_retailer', 'used_chip','used_pin_number','online_order']]  
y=data['fraud']  

nr = NearMiss()
  
X_nm,y_nm= nr.fit_resample(X,y)

X_train_nm, X_test_nm, y_train_nm, y_test_nm = train_test_split(X_nm, y_nm, test_size=0.7)

clf=RandomForestClassifier(n_estimators=100)

clf.fit(X_train_nm, y_train_nm)

y_pred=clf.predict(X_test_nm)

print('Percent of fraud is: {} %'.format(100*data[data['fraud']==1].shape[0]/data.shape[0]))

print("Accuracy: ",metrics.accuracy_score(y_test_nm, y_pred))
print("F1 Score: ",metrics.f1_score(y_test_nm,y_pred,average="macro"))
print("Specity: ",s)
