#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor





data_path = 'data/data_eqs_random.csv'

eq_data = pd.read_csv(data_path) 


y = eq_data.score
colnames = list(eq_data.columns)
eq_features = colnames[:-1]
X = eq_data[eq_features]

# split the data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)
eq_model = DecisionTreeRegressor(random_state=1)
eq_model.fit(train_X, train_y)
val_predictions = eq_model.predict(val_X)
print('error training all split data', round(mean_absolute_error(val_y, val_predictions),2))
print('\n')


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
maelist = []
for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %s" %(max_leaf_nodes, my_mae))
    maelist.append(round(my_mae,2))
maelist


best_tree_size = candidate_max_leaf_nodes[maelist.index(min(maelist))]
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
final_model.fit(X, y)
val_predictions = final_model.predict(X)
val_mae = mean_absolute_error(y, val_predictions)
print("Validation MAE for Decision Tree Model: {}".format(val_mae))




rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, val_predictions)
print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))






