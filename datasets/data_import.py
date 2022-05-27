import math

import scipy.io
import numpy as np
from pathlib import Path
import os
import pandas as pd
import torch


def int_in_3(num: int) -> str:
    if num < 0:
        return '000'
    if num < 10:
        return f'00{num}'
    if num < 100:
        return f'0{num}'
    if num < 1000:
        return str(num)
    return 'BIG'


def extract_data(relative_path: bytes | str, name: str) -> np.ndarray:
    path = get_absolute_path(relative_path)
    return np.array(scipy.io.loadmat(path)[name])


def get_absolute_path(relative_path: bytes | str) -> bytes | str:
    return os.path.join(Path(os.path.abspath(__file__)).parent, relative_path)


def get_capgmyo_dataset() -> pd.DataFrame:
    recordings = []
    labels = []
    for test_object in range(1, 19):
        for gesture in range(1, 9):
            for recording in range(1, 11):
                data = extract_data(
                    os.path.join(
                        os.path.dirname(os.getcwd()),
                        'data',
                        'CapgMyo',
                        'raw',
                        int_in_3(test_object) + '-' +
                        int_in_3(gesture) + '-' +
                        int_in_3(recording)),
                    'data')
                labels.extend([gesture - 1 for _ in range(data.shape[0])])
                data = np.split(data.reshape((data.shape[0], 16, 8)), data.shape[0])
                recordings.extend(data)
    return pd.DataFrame({'record': [i[0] for i in recordings], 'label': labels})


def get_csl_dataset() -> pd.DataFrame:
    recordings = []
    labels = []
    for subject in range(1, 6):
        for session in range(1, 6):
            for gest in range(27):
                data = extract_data(os.path.join(
                    os.path.dirname(os.getcwd()),
                    'data',
                    'csl-hdemg',
                    f'subject{str(subject)}',
                    f'session{str(session)}',
                    f'gest{str(gest)}.mat'),
                    'gestures')
                for i in range(data.shape[0]):
                    for j in range(data[i].shape[0]):
                        trial = data[i, j]
                        # deleting edge channels
                        trial = np.delete(trial, np.s_[7:192:8], 0)
                        trial = np.reshape(trial, (24, 7, -1))
                        # trial = np.flipud(np.transpose(trial, axes=(1, 0, 2)))
                        recordings.extend([trial[:, :, k] for k in range(trial.shape[2])])
                        labels.extend([gest for _ in range(trial.shape[2])])
    return pd.DataFrame({'record': recordings, 'label': labels})


def get_ninapro_dataset() -> pd.DataFrame:
    recordings = []
    labels = []
    for subject in range(1, 28):
        for session in range(1, 4):
            data = extract_data(os.path.join(
                os.path.dirname(os.getcwd()),
                'data',
                'NinaPro',
                f'S{str(subject)}_A1_E{str(session)}.mat'),
                'emg')
            gesture = extract_data(os.path.join(
                os.path.dirname(os.getcwd()),
                'data',
                'NinaPro',
                 f'S{str(subject)}_A1_E{str(session)}.mat'),
                'stimulus')
            recordings.extend([d.reshape(10, -1) for d in data])
            labels.extend(gesture[:, 0])
    return pd.DataFrame({'record': recordings, 'label': labels})


def save_arrays(dataframe: pd.DataFrame, name: str, path: os.path) -> pd.DataFrame:
    paths = []
    base_path = os.path.join(path, name)
    for index, row in dataframe.iterrows():
        current_path = os.path.join(base_path, f'{math.floor(int(index)/200000)}')
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        path = os.path.join(current_path, f'{name}_{index}.npy')
        np.save(path, row['record'])
        paths.append(path)

    df: pd.DataFrame = pd.DataFrame({'path': paths, 'label': dataframe['label']})
    df.to_csv(os.path.join(base_path, f'{name}.csv'), index=False)
    return df


def get_capgmyo_tensor() -> tuple[torch.Tensor, torch.Tensor]:
    recordings = torch.zeros(size=(1260000, 16, 8))
    labels = torch.zeros(size=[1260000])
    previous = 0
    for test_object in range(1, 19):
        for gesture in range(1, 9):
            for recording in range(1, 11):
                data = extract_data(
                    os.path.join(
                        os.path.dirname(os.getcwd()),
                        'data',
                        'CapgMyo',
                        'raw',
                        int_in_3(test_object) + '-' +
                        int_in_3(gesture) + '-' +
                        int_in_3(recording)),
                    'data')
                future = previous + data.shape[0]
                labels[previous:future] = gesture
                recordings[previous:future] = torch.from_numpy(data.reshape((data.shape[0], 16, 8)))
                previous = future
    return recordings, labels


def get_csl_tensor() -> tuple[torch.Tensor, torch.Tensor]:
    recordings = torch.zeros(size=(44531712, 24, 7))
    labels = torch.zeros(size=[44531712])
    previous = 0
    for subject in range(1, 6):
        for session in range(1, 6):
            for gest in range(27):
                data = extract_data(os.path.join(
                    os.path.dirname(os.getcwd()),
                    'data',
                    'csl-hdemg',
                    f'subject{str(subject)}',
                    f'session{str(session)}',
                    f'gest{str(gest)}.mat'),
                    'gestures')

                for i in range(data.shape[0]):
                    for j in range(data[i].shape[0]):
                        future = previous + data[i, j].shape[1]
                        labels[previous:future] = gest
                        recordings[previous:future] = torch.from_numpy(
                            np.transpose(
                                np.reshape(
                                    np.delete(
                                        data[i, j],
                                        np.s_[7:192:8],
                                        0),
                                    (24, 7, -1)),
                                (2, 0, 1)))
                        previous = future
    return recordings, labels


def get_ninapro_tensor() -> tuple[torch.Tensor, torch.Tensor]:
    recordings = torch.zeros(size=(12556311, 10, 1))
    labels = torch.zeros(size=[12556311])
    previous = 0
    for subject in range(1, 28):
        for session in range(1, 4):
            data = extract_data(os.path.join(
                os.path.dirname(os.getcwd()),
                'data',
                'NinaPro',
                f'S{str(subject)}_A1_E{str(session)}.mat'),
                'emg')

            gesture = extract_data(os.path.join(
                os.path.dirname(os.getcwd()),
                'data',
                'NinaPro',
                f'S{str(subject)}_A1_E{str(session)}.mat'),
                'stimulus')

            future = previous + data.shape[0]
            labels[previous:future] = torch.from_numpy(gesture)[:, 0]
            recordings[previous:future] = torch.from_numpy(data.reshape((data.shape[0], 10, 1)))
            previous = future
    return recordings, labels


def main():
    print(get_ninapro_dataset())
    print(get_ninapro_dataset().shape)


if __name__ == "__main__":
    main()
