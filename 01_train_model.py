
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
import pickle

fileName = 'data/model.sav'

cubeDF = pd.read_parquet('data/cubeDS.parquet')

X = cubeDF.drop(['Move_number','Run','Move'],axis=1)
y = cubeDF['Move']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

rfc = RandomForestClassifier()
parametros = {'n_estimators':[20,50,100],
            'criterion':['gini','entropy'],
            'min_samples_split':[2,3]}

grid_search = GridSearchCV(estimator = rfc,param_grid=parametros,cv = 2)
grid_search.fit(X_train, y_train)

melhores_param = grid_search.best_params_

rfc = RandomForestClassifier(**melhores_param)
rfc.fit(X_train, y_train)

with open(fileName,'wb') as file:
    pickle.dump(rfc, file)


