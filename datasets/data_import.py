import math
from typing import Callable

import scipy.io
import numpy as np
from pathlib import Path
import os
import pandas as pd
import requests
import patoolib

DATA_FOLDER = os.path.join('..', 'data')


def prepare_datasets() -> None:
    prepare_capgmyo()
    prepare_csl()
    prepare_ninapro()


def prepare_capgmyo() -> None:
    prepare_dataset('CapgMyo', '195rtjtCHZ1bFF9ginDeMFfhazwU-wozM', get_capgmyo_dataset)


def prepare_csl() -> None:
    prepare_dataset('csl-hdemg', '1_rpnlF1sqo1Kg5EhtCYuj1Xr6vAuv4jw', get_csl_dataset)


def prepare_ninapro() -> None:
    prepare_dataset('NinaPro', '1qXbf0SXdd8-ppIEKTmZacI_2vqrmjdzk', get_ninapro_dataset)


def prepare_dataset(dataset_name: str,  file_id: str, data_loading_function: Callable[[], pd.DataFrame]) -> None:
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    destined_folder = os.path.join(DATA_FOLDER, dataset_name)
    if not os.path.exists(destined_folder):
        os.makedirs(destined_folder)
    rar_file = os.path.join(destined_folder, dataset_name + '.rar')
    if not os.path.exists(rar_file):
        import_datasets(rar_file, file_id)
        patoolib.extract_archive(rar_file, outdir=destined_folder)
    final_folder = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'Data')
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)
    save_arrays(data_loading_function(), dataset_name, final_folder)


def import_datasets(destination: str, id: str) -> None:
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def extract_data(relative_path: bytes | str, name: str) -> np.ndarray:
    path = get_absolute_path(relative_path)
    return np.array(scipy.io.loadmat(path)[name])


def get_absolute_path(relative_path: bytes | str) -> bytes | str:
    return os.path.join(Path(os.path.abspath(__file__)).parent, relative_path)


def get_capgmyo_dataset() -> pd.DataFrame:
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
        current_path = os.path.join(base_path, f'{math.floor(int(index) / 200000)}')
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        path = os.path.join(current_path, f'{name}_{index}.npy')
        np.save(path, row['record'])
        paths.append(path)

    df: pd.DataFrame = pd.DataFrame({'path': paths, 'label': dataframe['label']})
    df.to_csv(os.path.join(base_path, f'{name}.csv'), index=False)
    return df


def main():
    print(get_ninapro_dataset())
    print(get_ninapro_dataset().shape)


if __name__ == "__main__":
    main()
