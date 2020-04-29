import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns

import sqlite3

conn = sqlite3.connect('taxi.db')
lon_bounds = [-74.03, -73.75]
lat_bounds = [40.6, 40.88]

c = conn.cursor()

my_string = 'SELECT * FROM taxi WHERE'

for word in ['pickup_lat', 'AND dropoff_lat']:
    my_string += ' {} BETWEEN {} AND {}'.format(word, lat_bounds[0], lat_bounds[1])

for word in ['AND pickup_lon', 'AND dropoff_lon']:
    my_string += ' {} BETWEEN {} AND {}'.format(word, lon_bounds[0], lon_bounds[1])

c.execute(my_string)

results = c.fetchall()

row_res = conn.execute('select * from taxi')
names = list(map(lambda x: x[0], row_res.description))


all_taxi = pd.DataFrame(results)
all_taxi.columns = names
all_taxi.head()

def pickup_scatter(t):
    plt.scatter(t['pickup_lon'], t['pickup_lat'], s=2, alpha=0.2)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Pickup locations')

plt.figure(figsize=(8, 8))
pickup_scatter(all_taxi)

conn = sqlite3.connect('taxi.db')
lon_bounds = [-74.03, -73.75]
lat_bounds = [40.6, 40.88]
c = conn.cursor()

my_string = 'SELECT * FROM taxi WHERE'

for word in ['pickup_lat', 'AND dropoff_lat']:
    my_string += ' {} BETWEEN {} AND {}'.format(word, lat_bounds[0], lat_bounds[1])

for word in ['AND pickup_lon', 'AND dropoff_lon']:
    my_string += ' {} BETWEEN {} AND {}'.format(word, lon_bounds[0], lon_bounds[1])

my_string += ' AND passengers > 0 AND duration BETWEEN 60 AND 3600 AND distance > 0 AND (3600*distance/duration) <= 100'

c.execute(my_string)

results = c.fetchall()

row_res = conn.execute('select * from taxi')
names = list(map(lambda x: x[0], row_res.description))


clean_taxi = pd.DataFrame(results)
clean_taxi.columns = names
clean_taxi.head()

polygon = pd.read_csv('manhattan.csv')

# Recommended: First develop and test a function that takes a position
#              and returns whether it's in Manhattan.
def in_manhattan(x, y):
    """Whether a longitude-latitude (x, y) pair is in the Manhattan polygon."""

    polyX = [-74.010773, -73.999271, -73.978758, -73.971977, -73.971291, -73.973994, -73.968072, -73.941936, -73.942580,
            -73.943589, -73.939362, -73.936272, -73.932238, -73.929491, -73.928976, -73.930907, -73.934298, -73.934383,
            -73.922281, -73.908892, -73.928289, -73.947258, -73.947086, -73.955498, -74.008713, -74.013863, -74.013605,
            -74.017038, -74.020042, -74.016438]
    polyY = [40.700292, 40.707580, 40.710443, 40.721762, 40.729568, 40.733503, 40.746834, 40.775114, 40.778884, 40.781906,
            40.785351, 40.789640, 40.793149, 40.795228, 40.801141, 40.804877, 40.810496, 40.834074, 40.855371, 40.870690,
            40.878348, 40.851151, 40.844074, 40.828229, 40.754019, 40.719941, 40.718575, 40.718802, 40.704977, 40.700553]
    polyCorners = 30
    i = polyCorners - 1
    j = polyCorners - 1
    oddNodes = False

    for i in range(0, polyCorners):
        if (polyY[i] < y and polyY[j] >= y) or (polyY[j] < y and polyY[i] >= y) and (polyX[i] <= x or polyX[j] <= x):
            if polyX[i] + (y - polyY[i])/(polyY[j]-polyY[i])*(polyX[j] - polyX[i]) < x:
                if oddNodes == False:
                    oddNodes = True
                else:
                    oddNodes = False
        j = i

    return oddNodes

def in_manh(row):
     if in_manhattan(row['pickup_lon'], row['pickup_lat']):
            if in_manhattan(row['dropoff_lon'], row['dropoff_lat']):
                return True
            else:
                return False
     else:
        return False

manhattan_taxi = clean_taxi.copy()
manhattan_taxi['is_in_manh'] = manhattan_taxi.apply(in_manh, axis=1)


# Recommended: Then, apply this function to every trip to filter clean_taxi.

manhattan_taxi = manhattan_taxi.loc[manhattan_taxi.is_in_manh == True]
manhattan_taxi.drop(["is_in_manh"], axis = 1, inplace = True)

manhattan_taxi = pd.read_csv('manhattan_taxi.csv')

plt.figure(figsize=(8, 16))
pickup_scatter(manhattan_taxi)

first_var = all_taxi.shape[0]
second_var = first_var - clean_taxi.shape[0]
third_var = second_var/first_var
fourth_var = manhattan_taxi.shape[0]
mystr = "Of the original " + str(first_var) + " trips, "
mystr += str(second_var) + " anomalous trips (" + str(third_var)
mystr += "%) were removed " + "through data cleaning, and then the "
mystr += str(fourth_var) + " trips within Manhattan were selected"
mystr += " for further analysis."
print(mystr)

manhattan_taxi['date'] = pd.to_datetime(manhattan_taxi.pickup_datetime)
manhattan_taxi['date'] = manhattan_taxi['date'].dt.date
manhattan_taxi.head()

means = manhattan_taxi.groupby(["date"]).mean()['duration']
ax = means.plot(kind = 'bar')
ax.set_ylabel('average duration')

import calendar
import re

from datetime import date

atypical = [1, 2, 3, 18, 23, 24, 25, 26]
typical_dates = [date(2016, 1, n) for n in range(1, 32) if n not in atypical]
typical_dates

print('Typical dates:\n')
pat = '  [1-3]|18 | 23| 24|25 |26 '
print(re.sub(pat, '   ', calendar.month(2016, 1)))

final_taxi = manhattan_taxi[manhattan_taxi['date'].isin(typical_dates)]

import sklearn.model_selection

train, test = sklearn.model_selection.train_test_split(
    final_taxi, train_size=0.8, test_size=0.2, random_state=42)
print('Train:', train.shape, 'Test:', test.shape)

plt.figure(figsize=(15,8))
plt.xticks(rotation=90)
ax = sns.boxplot(x="date", y="duration", order=["2016-01-04", "2016-01-05","2016-01-06","2016-01-07","2016-01-08",
                                                "2016-01-09","2016-01-10","2016-01-11","2016-01-12","2016-01-13",
                                                "2016-01-14","2016-01-15","2016-01-16","2016-01-17","2016-01-19",
                                                "2016-01-20","2016-01-21","2016-01-22","2016-01-27","2016-01-28",
                                                "2016-01-29","2016-01-30","2016-01-31"], data=train)

def speed(t):
    """Return a column of speeds in miles per hour."""
    return t['distance'] / t['duration'] * 60 * 60

def augment(t):
    """Augment a dataframe t with additional columns."""
    u = t.copy()
    pickup_time = pd.to_datetime(t['pickup_datetime'])
    u.loc[:, 'hour'] = pickup_time.dt.hour
    u.loc[:, 'day'] = pickup_time.dt.weekday
    u.loc[:, 'weekend'] = (pickup_time.dt.weekday >= 5).astype(int)
    u.loc[:, 'period'] = np.digitize(pickup_time.dt.hour, [0, 6, 18])
    u.loc[:, 'speed'] = speed(t)
    return u

train = augment(train)
test = augment(test)
train.iloc[0,:] # An example row

morning_1 = train.loc[train['period'] == 1]
day_2 = train.loc[train['period'] == 2]
night_3 = train.loc[train['period'] == 3]

sns.distplot(morning_1['speed'], label="Early Morning")
sns.distplot(day_2['speed'], label="Day")
sns.distplot(night_3['speed'], label="Night")
plt.legend()
plt.show()

D = train[['pickup_lon', 'pickup_lat']]
pca_n = D.shape[0]
pca_means = np.mean(D)
X = (D - pca_means) / np.sqrt(pca_n)
u, s, vt = np.linalg.svd(X, full_matrices=False)

def add_region(t):
    """Add a region column to t based on vt above."""
    D = t[['pickup_lon', 'pickup_lat']]
    assert D.shape[0] == t.shape[0], 'You set D using the incorrect table'
    # Always use the same data transformation used to compute vt
    X = (D - pca_means) / np.sqrt(pca_n)
    first_pc = np.dot(X,vt)[:,0]
    t.loc[:,'region'] = pd.qcut(first_pc, 3, labels=[0, 1, 2])

add_region(train)
add_region(test)

plt.figure(figsize=(8, 16))
for i in [0, 1, 2]:
    pickup_scatter(train[train['region'] == i])

from sklearn.preprocessing import StandardScaler

num_vars = ['pickup_lon', 'pickup_lat', 'dropoff_lon', 'dropoff_lat', 'distance']
cat_vars = ['hour', 'day', 'region']

scaler = StandardScaler()
scaler.fit(train[num_vars])

def design_matrix(t):
    """Create a design matrix from taxi ride dataframe t."""
    scaled = t[num_vars].copy()
    scaled.iloc[:,:] = scaler.transform(scaled) # Convert to standard units
    categoricals = [pd.get_dummies(t[s], prefix=s, drop_first=True) for s in cat_vars]
    return pd.concat([scaled] + categoricals, axis=1)

# This processes the full train set, then gives us the first item
# Use this function to get a processed copy of the dataframe passed in
# for training / evaluation
design_matrix(train).iloc[0,:]

def rmse(errors):
    """Return the root mean squared error."""
    return np.sqrt(np.mean(errors ** 2))

constant_rmse = rmse(test['duration'] - np.mean(test['duration']))
constant_rmse

from sklearn.linear_model import LinearRegression

model = LinearRegression(fit_intercept=True)
only_dist = train.drop(["pickup_datetime", "dropoff_datetime", "pickup_lon", "pickup_lat",
                       "dropoff_lon", "dropoff_lat", "passengers", "duration",
                       "date", "hour", "day", "weekend",
                       "period", "speed", "region"], axis=1)
only_dur = train.drop(["pickup_datetime", "dropoff_datetime", "pickup_lon", "pickup_lat",
                       "dropoff_lon", "dropoff_lat", "passengers", "distance",
                       "date", "hour", "day", "weekend",
                       "period", "speed", "region"], axis=1)

only_dist_t = test.drop(["pickup_datetime", "dropoff_datetime", "pickup_lon", "pickup_lat",
                       "dropoff_lon", "dropoff_lat", "passengers", "duration",
                       "date", "hour", "day", "weekend",
                       "period", "speed", "region"], axis=1)

only_dur_t = test.drop(["pickup_datetime", "dropoff_datetime", "pickup_lon", "pickup_lat",
                       "dropoff_lon", "dropoff_lat", "passengers", "distance",
                       "date", "hour", "day", "weekend",
                       "period", "speed", "region"], axis=1)

model.fit(only_dist, only_dur)
y_pred = model.predict(only_dist_t)
simple_rmse = float(rmse(y_pred - only_dur_t))
simple_rmse

model = LinearRegression()
model.fit(design_matrix(train), only_dur)
y_pred = model.predict(design_matrix(test))
linear_rmse = float(rmse(y_pred - only_dur_t))
linear_rmse

model = LinearRegression()
errors = []

for v in np.unique(train['period']):
    train_v = train.loc[train['period'] == v]
    only_dur_v = train_v.drop(["pickup_datetime", "dropoff_datetime", "pickup_lon", "pickup_lat",
                       "dropoff_lon", "dropoff_lat", "passengers", "distance",
                       "date", "hour", "day", "weekend",
                       "period", "speed", "region"], axis=1)
    model.fit(design_matrix(train_v), only_dur_v)
    test_v = test.loc[test['period'] == v]
    only_dur_t_v = test_v.drop(["pickup_datetime", "dropoff_datetime", "pickup_lon", "pickup_lat",
                       "dropoff_lon", "dropoff_lat", "passengers", "distance",
                       "date", "hour", "day", "weekend",
                       "period", "speed", "region"], axis=1)
    y_pred = model.predict(design_matrix(test_v))
    errors.extend((y_pred - only_dur_t_v)['duration'].tolist())

period_rmse = rmse(np.array(errors))
period_rmse

model = LinearRegression()
only_speed = train['speed']

model.fit(design_matrix(train), only_speed)
y_pred = model.predict(design_matrix(test))
y_pred = np.divide(y_pred, 3600)
y_pred = np.divide(test['distance'], y_pred)

y_actual = test['duration']


speed_rmse = float(rmse(y_actual - y_pred))
speed_rmse

model = LinearRegression()
choices = ['period', 'region', 'weekend']

def duration_error(predictions, observations):
    """Error between duration predictions (array) and observations (data frame)"""
    return predictions - observations['duration']

def speed_error(predictions, observations):
    """Duration error between speed predictions and duration observations"""
    return predictions - observations['duration']

def tree_regression_errors(outcome='duration', error_fn=duration_error):
    """Return errors for all examples in test using a tree regression model."""
    errors = []
    for vs in train.groupby(choices).size().index:
        v_train, v_test = train, test
        for v, c in zip(vs, choices):
            v_train = v_train.loc[v_train[c] == v]
            v_test = v_test.loc[v_test[c] == v]
        v_train_labels = v_train[outcome]
        v_test_labels = v_test['duration']
        v_train = v_train.drop(outcome, axis=1)
        v_test = v_test.drop(outcome, axis=1)
        model.fit(design_matrix(v_train), v_train_labels)
        y_pred = model.predict(design_matrix(v_test))
        if outcome == 'speed':
            y_pred = np.divide(y_pred, 3600)
            y_pred = np.divide(v_test['distance'], y_pred)
        errors.extend(y_pred - v_test_labels)
    return errors

errors = tree_regression_errors()
errors_via_speed = tree_regression_errors('speed', speed_error)
tree_rmse = rmse(np.array(errors))
tree_speed_rmse = rmse(np.array(errors_via_speed))
print('Duration:', tree_rmse, '\nSpeed:', tree_speed_rmse)

models = ['constant', 'simple', 'linear', 'period', 'speed', 'tree', 'tree_speed']
pd.DataFrame.from_dict({
    'Model': models,
    'Test RMSE': [eval(m + '_rmse') for m in models]
}).set_index('Model').plot(kind='barh');

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

X_train = design_matrix(train)

import keras.backend as K

def my_rms(y_true, y_pred):
   return K.sqrt(K.mean(y_pred ** 2))

def build_model():
    model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[len(X_train.keys())]),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
  ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=[my_rms])
    return model

model = build_model()
model.summary()

y_train = train['duration']
y_train = np.asarray(y_train)
X_train = np.asarray(X_train)
history = model.fit(
  X_train, y_train,
  epochs=100, validation_split = 0.2, verbose=0)

X_test = design_matrix(test)
X_test = np.asarray(X_test)
y_test = test['duration']
from sklearn.metrics import mean_squared_error
from math import sqrt

def rmse(errors):
    """Return the root mean squared error."""
    return np.sqrt(np.mean(errors ** 2))

y_pred = model.predict(X_test).flatten()
errors = np.asarray(y_test) - y_pred
rm = rmse(errors)
print("Test Set RMSE: ", rm)

y_pred = model.predict(X_train).flatten()
errors = np.asarray(y_train) - y_pred
rm = rmse(errors)
print("Training Set RMSE: ", rm)
