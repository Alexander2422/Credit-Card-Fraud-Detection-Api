import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics


df=pd.read_csv('card_transdata_train.csv')

X=df[['distance_from_home','distance_from_last_transaction','repeat_retailer','used_chip','used_pin_number',
        'online_order']].values
y=df[['fraud']]

y=np.array(y['fraud'])


X_train,X_test,y_train,y_test= train_test_split(X,y, train_size=0.2)
knn=neighbors.KNeighborsClassifier(n_neighbors=25,weights='distance')
model=knn.fit(X_train,y_train)

y_pred=knn.predict(X_test)

accuracy=metrics.accuracy_score(y_test,y_pred)
#print('Accuracy =',accuracy)

pickle_out = open ("classifier.pkl","wb")
pickle.dump(model,pickle_out)
pickle_out.close()
