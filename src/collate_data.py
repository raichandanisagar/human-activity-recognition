import datetime as dt
import os
import pickle
import numpy as np
import pandas as pd


def build_dataset(data_store='../data'):
    csv_files_arr = []

    for directory in os.listdir(data_store):
        if os.path.isfile(os.path.join(data_store, directory)):
            continue
        sensor_placement = directory

        for subdirectory in os.listdir(os.path.join(data_store, directory)):
            if subdirectory.startswith(r'.'):
                continue
            if os.path.isfile(os.path.join(data_store, directory, subdirectory)):
                continue

            activity = subdirectory

            for filename in os.listdir(os.path.join(data_store, directory, subdirectory)):
                if filename.endswith('.csv'):
                    user = filename.replace('.csv', '')
                    csv_files_arr.append(
                        [os.path.abspath(os.path.join(data_store, directory, subdirectory, filename)), sensor_placement,
                         activity, user])

    person_data = {}

    for i, row in enumerate(csv_files_arr):
        path = row[0]
        sensor_placement = row[1]
        activity_class = row[2]
        user = row[3]

        if sensor_placement != 'w':
            continue

        if user in person_data:
            activity_data = person_data[user]
        else:
            activity_data = {}

        if activity_class in activity_data:
            activity_df = activity_data[activity_class]
        else:
            activity_df = pd.DataFrame()

        _df = pd.read_csv(path)
        _df['m'] = np.sqrt(np.square(_df['x'].values) + np.square(_df['y'].values) + np.square(_df['z'].values))
        _df['user'] = user
        _df['activity_class'] = activity_class
        _df['epoch'] = _df['time'].apply(lambda x: ((dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f') -
                                                     dt.datetime.utcfromtimestamp(0)).total_seconds()) * 1000)

        try:
            _df.drop(columns=['class'], inplace=True)
        except KeyError:
            pass

        activity_df = pd.concat([activity_df, _df])
        activity_data[activity_class] = activity_df
        person_data[user] = activity_data
    return person_data


def main():
    person_data = build_dataset()
    with open('../data/wrist_dump.pkl', 'wb+') as f:
        pickle.dump(person_data, f)
