import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np


data_dict = pickle.load(open('data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier(n_estimators=120, random_state=42)

model.fit(x_train, y_train)

# Predict on test set
y_predict = model.predict(x_test)

score = accuracy_score(y_test, y_predict)
print('Initial RandomForestClassifier accuracy: {:.2f}%'.format(score * 100))

# Further optimize the model using GridSearchCV for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 150],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(x_train, y_train)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Predict on test set with the best model
y_predict_best = best_model.predict(x_test)

# Evaluate best model
score_best = accuracy_score(y_test, y_predict_best)
print('Optimized RandomForestClassifier accuracy: {:.2f}%'.format(score_best * 100))

# Save best model to file
with open('model.pickle', 'wb') as f:
    pickle.dump(best_model, f)