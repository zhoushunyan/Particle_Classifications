import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import gc
import os
from keras.utils import to_categorical
from sklearn.metrics import classification_report
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBClassifier

pd.set_option('display.max_rows', 500)  # 显示500 行 和500 列
pd.set_option('display.max_columns', 500)


def count_column(df, column):
    tp = df.groupby(column).count().reset_index()  # 取df中的colum列为新的索引，并且统计该列出现的次数
    tp = tp[list(tp.columns)[0:2]]  # 将数据转化为列表，并且取前两列数据
    tp.columns = [column, column + '_count']  # 改变列标
    df = df.merge(tp, on=column, how='left')  # 将 count_count 添加到右边
    return df


##添加对应列的平均值到左边

def count_mean(df, base_column, count_column):
    tp = df.groupby(base_column).agg({count_column: ['mean']}).reset_index()
    tp.columns = [base_column, base_column + '_' + count_column + '_mean']
    df = df.merge(tp, on=base_column, how='left')
    return df


## 数量

def count_count(df, base_column, count_column):
    tp = df.groupby(base_column).agg({count_column: ['count']}).reset_index()
    tp.columns = [base_column, base_column + '_' + count_column + '_count']
    df = df.merge(tp, on=base_column, how='left')
    return df


##求和

def count_sum(df, base_column, count_column):
    tp = df.groupby(base_column).agg({count_column: ['sum']}).reset_index()
    tp.columns = [base_column, base_column + '_' + count_column + '_sum']
    df = df.merge(tp, on=base_column, how='left')
    return df


##
def count_std(df, base_column, count_column):
    tp = df.groupby(base_column).agg({count_column: ['sum']}).reset_index()
    tp.columns = [base_column, base_column + '_' + count_column + '_var']
    df = df.merge(tp, on=base_column, how='left')
    return df


train = pd.read_csv('jet_simple_data/simple_train_R04_jet.csv')
test = pd.read_csv('jet_simple_data/simple_test_R04_jet.csv')

# train = pd.read_csv('data/simpletrain.csv')
# test = pd.read_csv('data/simpletest.csv')

# train['jet_energy'] = train['jet_energy']/train['jet_energy'].mean()
# train['number_of_particles_in_this_jet'] = train['number_of_particles_in_this_jet']/train['number_of_particles_in_this_jet'].mean()
# train['jet_px'] = train['jet_px']/train['jet_px'].mean()
# train['jet_py'] = train['jet_py']/train['jet_py'].mean()
# train['jet_pz'] = train['jet_pz']/train['jet_pz'].mean()
# train['jet_mass'] = train['jet_mass']/train['jet_mass'].mean()

train['tan'] = train['jet_pz'] / (train['jet_px']**2 + train['jet_py']**2)**0.5
train['cos'] = ((train['jet_px']**2 + train['jet_py']**2)**0.5 / (train['jet_px']**2 + train['jet_py']**2 +train['jet_pz']**2)**0.5) *train['jet_pz']
train['sin'] = train['jet_pz'] / (train['jet_px']**2 + train['jet_py']**2 +train['jet_pz']**2)**0.5


test['tan'] = test['jet_pz'] / (test['jet_px']**2 + test['jet_py']**2)**0.5
test['cos'] = ((test['jet_px']**2 + test['jet_py']**2)**0.5 / (test['jet_px']**2 + test['jet_py']**2 +test['jet_pz']**2)**0.5) *test['jet_pz']
test['sin'] = test['jet_pz'] / (test['jet_px']**2 + test['jet_py']**2 +test['jet_pz']**2)**0.5


train = count_mean(train, 'event_id', 'sin')
train = count_sum(train, 'event_id', 'sin')
train = count_std(train, 'event_id', 'sin')

train = count_mean(train, 'event_id', 'cos')
train = count_sum(train, 'event_id', 'cos')
train = count_std(train, 'event_id', 'cos')

train = count_mean(train, 'event_id', 'tan')
train = count_sum(train, 'event_id', 'cos')
train = count_std(train, 'event_id', 'tan')


test = count_mean(test, 'event_id', 'sin')
test = count_sum(test, 'event_id', 'sin')
test = count_std(test, 'event_id', 'sin')

test = count_mean(test, 'event_id', 'cos')
test = count_sum(test, 'event_id', 'cos')
test = count_std(test, 'event_id', 'cos')

test = count_mean(test, 'event_id', 'tan')
test = count_sum(test, 'event_id', 'cos')
test = count_std(test, 'event_id', 'tan')

def energy(df):
    x = df['jet_px']
    y = df['jet_py']
    z = df['jet_pz']
    return (x ** 2 + y ** 2 + z ** 2) ** 0.5


train['energy'] = train.apply(energy, axis=1)
test['energy'] = test.apply(energy, axis=1)

# train['x_n'] = train['jet_px'] / train['energy']
# train['y_n'] = train['jet_py'] / train['energy']
# train['z_n'] = train['jet_pz'] / train['energy']
#
# test['x_n'] = test['jet_px'] / test['energy']
# test['y_n'] = test['jet_py'] / test['energy']
# test['z_n'] = test['jet_pz'] / test['energy']

# train = count_mean(train, 'event_id', 'x_n')
# train = count_sum(train, 'event_id', 'x_n')
# train = count_std(train, 'event_id', 'x_n')
#
# train = count_mean(train, 'event_id', 'y_n')
# train = count_sum(train, 'event_id', 'y_n')
# train = count_std(train, 'event_id', 'y_n')
#
# train = count_mean(train, 'event_id', 'z_n')
# train = count_sum(train, 'event_id', 'z_n')
# train = count_std(train, 'event_id', 'z_n')
#
# test = count_mean(test, 'event_id', 'x_n')
# test = count_sum(test, 'event_id', 'x_n')
# test = count_std(test, 'event_id', 'x_n')
#
# test = count_mean(test, 'event_id', 'y_n')
# test = count_sum(test, 'event_id', 'y_n')
# test = count_std(test, 'event_id', 'y_n')
#
# test = count_mean(test, 'event_id', 'z_n')
# test = count_sum(test, 'event_id', 'z_n')
# test = count_std(test, 'event_id', 'z_n')

train['abs'] = train['jet_energy'] - train['energy']
test['abs'] = test['jet_energy'] - test['energy']

# train = count_mean(train, 'event_id', 'number_of_particles_in_this_jet')
# train = count_sum(train, 'event_id', 'number_of_particles_in_this_jet')
# train = count_std(train, 'event_id', 'number_of_particles_in_this_jet')

train = count_mean(train, 'event_id', 'jet_mass')
train = count_sum(train, 'event_id', 'jet_mass')
train = count_std(train, 'event_id', 'jet_mass')

train = count_mean(train, 'event_id', 'jet_energy')
train = count_sum(train, 'event_id', 'jet_energy')
train = count_std(train, 'event_id', 'jet_energy')

train['mean_energy'] = train['jet_energy'] / train['number_of_particles_in_this_jet']
train['mean_jet_mass'] = train['jet_mass'] / train['number_of_particles_in_this_jet']
train = count_mean(train, 'event_id', 'mean_energy')
train = count_sum(train, 'event_id', 'mean_energy')
train = count_std(train, 'event_id', 'mean_energy')
train = count_mean(train, 'event_id', 'mean_jet_mass')
train = count_sum(train, 'event_id', 'mean_jet_mass')
train = count_std(train, 'event_id', 'mean_jet_mass')
train = count_mean(train, 'event_id', 'abs')
train = count_sum(train, 'event_id', 'abs')
train = count_std(train, 'event_id', 'abs')
train = count_mean(train, 'event_id', 'energy')
train = count_sum(train, 'event_id', 'energy')
train = count_std(train, 'event_id', 'energy')

# test = count_mean(test, 'event_id', 'number_of_particles_in_this_jet')
# test = count_sum(test, 'event_id', 'number_of_particles_in_this_jet')
# test = count_std(test, 'event_id', 'number_of_particles_in_this_jet')

test = count_mean(test, 'event_id', 'jet_mass')
test = count_sum(test, 'event_id', 'jet_mass')
test = count_std(test, 'event_id', 'jet_mass')

test = count_mean(test, 'event_id', 'jet_energy')
test = count_sum(test, 'event_id', 'jet_energy')
test = count_std(test, 'event_id', 'jet_energy')

test['mean_energy'] = test['jet_energy'] / test['number_of_particles_in_this_jet']
test['mean_jet_mass'] = test['jet_mass'] / test['number_of_particles_in_this_jet']
test = count_mean(test, 'event_id', 'mean_energy')
test = count_sum(test, 'event_id', 'mean_energy')
test = count_std(test, 'event_id', 'mean_energy')
test = count_mean(test, 'event_id', 'mean_jet_mass')
test = count_sum(test, 'event_id', 'mean_jet_mass')
test = count_std(test, 'event_id', 'mean_jet_mass')
test = count_mean(test, 'event_id', 'abs')
test = count_sum(test, 'event_id', 'abs')
test = count_std(test, 'event_id', 'abs')
test = count_mean(test, 'event_id', 'energy')
test = count_sum(test, 'event_id', 'energy')
test = count_std(test, 'event_id', 'energy')

train = train.drop_duplicates(subset=['event_id']).reset_index(drop=True)
train=train.sort_values(by='event_id').reset_index(drop=True)


from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

label_class_correspondence = {1: 0, 4: 1, 5: 2, 21: 3}
class_label_correspondence = {0: 1, 1: 4, 2: 5, 3: 21}


def get_class_ids(labels):
    return np.array([label_class_correspondence[alabel] for alabel in labels])


train['Class'] = get_class_ids(train.label.values)
print(set(train.Class))

features = list(set(train.columns) - {'label', 'Class', 'jet_id','event_id'})
print(features)
print(train)
training_data, validation_data = train_test_split(train, random_state=11, train_size=0.80)

len(training_data), len(validation_data)

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(learning_rate=0.05,
                                n_estimators=500,
                                min_samples_split=1200,
                                subsample=0.6,
                                random_state=13,
                                verbose=1,
                                min_samples_leaf=50,
                                min_weight_fraction_leaf=0.3,
                                max_depth=8,
                                max_features=20
                                )
gb.fit(training_data[features].values, training_data.Class.values)
#
# proba_gb = gb.predict_proba(validation_data[features].values)
#
# log_loss(validation_data.Class.values, proba_gb)

best_model = gb

submit_proba = best_model.predict_proba(test[features].values)

submit_ids = test.jet_id

from IPython.display import FileLink


def create_solution(ids, proba, filename='submission_file.csv.zip'):
    """saves predictions to file and provides a link for downloading """
    solution = pd.DataFrame({'jet_id': ids})
    for name in [1, 4, 5, 21]:
        solution[name] = proba[:, label_class_correspondence[name]]
    solution.to_csv('{}'.format(filename), index=False, float_format='%.5f', compression="gzip")
    return FileLink('{}'.format(filename))


create_solution(submit_ids, submit_proba, filename='submission2_file.csv.gz')
