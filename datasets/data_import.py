from functools import partial
from typing import Callable

import scipy.io
import numpy as np
from pathlib import Path
import os
import pandas as pd
import patoolib
import gdown

from definitions import DATA_FOLDER


def prepare_datasets(prepare_dataset: Callable[[str, str, Callable[[], pd.DataFrame], str], None],
                     final_path: str) -> None:
    """
    Download and prepare 3 base datasets.
    Args:
        prepare_dataset: function like: prepare_frame_dataset or prepare_dataframe_dataset
        final_path: folder in which datasets will be saved.
    """
    prepare_capgmyo(prepare_dataset, final_path)
    prepare_ninapro(prepare_dataset, final_path)
    prepare_myoarmband(prepare_dataset, final_path)


def prepare_capgmyo(prepare_dataset: Callable[[str, str, Callable[[], pd.DataFrame], str], None],
                    final_path: str) -> None:
    """
    Download and prepare CapgMyo dataset.
    Args:
        prepare_dataset: function like: prepare_frame_dataset or prepare_dataframe_dataset
        final_path: folder in which datasets will be saved.
    """
    prepare_dataset('CapgMyo', '1Xjtkr-rl2m3_80BvNg1wYSXN8yNaevZl', get_capgmyo_dataset, final_path)


def prepare_ninapro(prepare_dataset: Callable[[str, str, Callable[[], pd.DataFrame], str], None],
                    final_path: str) -> None:
    """
    Download and prepare NinaPro dataset.
    Args:
        prepare_dataset: function like: prepare_frame_dataset or prepare_dataframe_dataset
        final_path: folder in which datasets will be saved.
    """
    prepare_dataset('NinaPro', '1BtNxCiGIqVWYiPtf0AxyZuTY0uz8j6we', get_ninapro_dataset, final_path)


def prepare_myoarmband(prepare_dataset: Callable[[str, str, Callable[[], pd.DataFrame], str], None],
                       final_path: str) -> None:
    """
    Download and prepare MyoArmband dataset.
    Args:
        prepare_dataset: function like: prepare_frame_dataset or prepare_dataframe_dataset
        final_path: folder in which datasets will be saved.
    """
    prepare_dataset('MyoArmband', '1dO72tvtx5HOzZZ0C56IIUdCIVO3nNGt5', get_myoarmband_dataset, final_path)


def prepare_knibm_low(prepare_dataset: Callable[[str, str, Callable[[], pd.DataFrame], str], None],
                       final_path: str) -> None:
    """
    Download and prepare KNIBM Low dataset.
    Args:
        prepare_dataset: function like: prepare_frame_dataset or prepare_dataframe_dataset
        final_path: folder in which datasets will be saved.
    """
    prepare_dataset('knibm-low', '1N0xs9oAk_DLIu4hR_Caen4h4kEiGsWNe', partial(get_knibm_dataset, "low"), final_path)


def prepare_knibm_high(prepare_dataset: Callable[[str, str, Callable[[], pd.DataFrame], str], None],
                       final_path: str) -> None:
    """
    Download and prepare KNIBM High dataset.
    Args:
        prepare_dataset: function like: prepare_frame_dataset or prepare_dataframe_dataset
        final_path: folder in which datasets will be saved.
    """
    prepare_dataset('knibm-high', '1fgyzWf9wXhiViOcoaTKi1BNNNA2kWICT', partial(get_knibm_dataset, "high"), final_path)


final_folder = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'Data')


def prepare_frame_dataset(dataset_name: str, file_id: str, data_loading_function: Callable[[], pd.DataFrame],
                          final_folder: str) -> None:
    """
    Prepare dataset in form of separate files with arrays representing samples.
    Args:
        dataset_name: Name of dataset.
        file_id: Identifier of dataset on Google Drive for download with gdown.
        data_loading_function: Function that will return loaded dataset
        final_folder: folder in which datasets will be saved.
    """
    prepare_folders(dataset_name, file_id)
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)
    save_arrays(data_loading_function(), dataset_name, final_folder)


def prepare_dataframe_dataset(dataset_name: str, file_id: str, data_loading_function: Callable[[], pd.DataFrame],
                              final_folder: str) -> None:
    """
    Prepare dataset in form of dataframe stored as .pkl file.
    Args:
        dataset_name: Name of dataset.
        file_id: Identifier of dataset on Google Drive for download with gdown.
        data_loading_function: Function that will return loaded dataset
        final_folder: folder in which datasets will be saved.
    """
    prepare_folders(dataset_name, file_id)
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)
    subfolder = os.path.join(final_folder, dataset_name)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    data_loading_function().to_pickle(os.path.join(final_folder, dataset_name, f'{dataset_name}.pkl'))


def prepare_folders(dataset_name: str, file_id: str) -> None:
    """
    Prepare folders for storing data, downloads and extracts data into them.
    Args:
        dataset_name: Name of dataset.
        file_id: Identifier of dataset on Google Drive for download with gdown.
    """
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    destined_folder = os.path.join(DATA_FOLDER, dataset_name)
    if not os.path.exists(destined_folder):
        os.makedirs(destined_folder)
    zip_file = os.path.join(destined_folder, dataset_name + '.zip')
    if not os.path.exists(zip_file):
        import_datasets(zip_file, file_id)
        patoolib.extract_archive(zip_file, outdir=destined_folder)


def import_datasets(destination: str, id: str) -> None:
    """
    Download compressed data files from Google Drive.
    Args:
        destination: Path of file to which download compressed dataset.
        id: Identifier of dataset on Google Drive for download with gdown.
    """
    gdown.download(id=id, output=destination, quiet=False)


def extract_data(relative_path: bytes or str, name: str) -> np.ndarray:
    """
    Extract data from MatLab file to an array.
    Args:
        relative_path: Relative path to .mat file
        name: Name of the column with desired data in .mat file

    Returns: Array with desired data
    """
    path = get_absolute_path(relative_path)
    return np.array(scipy.io.loadmat(path)[name])


def get_absolute_path(relative_path: bytes or str) -> bytes or str:
    """
    Get Absolute path from relative_path
    Args:
        relative_path: Relative path (in relation to current file)

    Returns: Absolute path
    """
    return os.path.join(Path(os.path.abspath(__file__)).parent, relative_path)


def get_capgmyo_dataset() -> pd.DataFrame:
    """
    Go through original files of CapgMyo dataset and convert them into form useful for computation in python.
    Returns: Dataframe with dataset.
    """
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
    series = []
    subjects = []
    for test_object in range(1, 19):
        for gesture in range(1, 9):
            for recording in range(1, 11):
                data = extract_data(
                    os.path.join(
                        os.path.dirname(os.getcwd()),
                        'data',
                        'CapgMyo',
                        int_in_3(test_object) + '-' +
                        int_in_3(gesture) + '-' +
                        int_in_3(recording)),
                    'data')
                size = data.shape[0]
                labels.extend([gesture - 1 for _ in range(size)])
                data = np.split(data.reshape((size, 16, 8)).transpose(0, 2, 1), size)
                recordings.extend(data)
                series.extend(
                    [(recording - 1) + (gesture - 1) * 10 + (test_object - 1) * 80 for _ in range(size)])
                subjects.extend([test_object for _ in range(size)])
    return pd.DataFrame({'record': [i[0] for i in recordings], 'label': labels, 'spectrograms': series, 'subject': subjects})


def get_ninapro_dataset() -> pd.DataFrame:
    """
    Go through original files of NinaPro dataset and convert them into form useful for computation in python.
    Returns: Dataframe with dataset.
    """
    recordings = []
    labels = []
    series = []
    subjects = []
    last_series = -1
    for subject in range(1, 28):
        start_gest = 0
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
            proper_gesture = [(g if g == 0 else g + start_gest) for g in gesture[:, 0]]
            start_gest = max(proper_gesture)
            labels.extend(proper_gesture)
            counter = 1 + last_series
            series.append(counter)
            previous = gesture[0, 0]
            for gest in gesture[1:, 0]:
                if gest != previous:
                    counter += 1
                previous = gest
                series.append(counter)
            last_series = counter
            subjects.extend([subject for _ in range(gesture.shape[0])])
            # series.extend([(subject-1) * 3 + (session-1) for _ in range(gesture.shape[0])])
    df = pd.DataFrame({'record': recordings, 'label': labels, 'spectrograms': series, 'subject': subjects})
    df = df.loc[df['label'] != 0]
    df['label'] = df['label'] - 1
    df.reset_index(inplace=True)
    return df


def get_myoarmband_dataset() -> pd.DataFrame:
    """
    Go through original files of MyoArmband dataset and convert them into form useful for computation in python.
    Returns: Dataframe with dataset.
    """
    def format_data_to_train(vector_to_format):
        emg_vector = []
        records = []
        for value in vector_to_format:
            emg_vector.append(value)
            if len(emg_vector) >= 8:
                records.append(np.array(emg_vector, dtype=np.float32).reshape((8, 1)))
                emg_vector = []
        return pd.DataFrame({'record': records})

    def classe_to_df(path: str, subfolder: str, start_val: int = 0):
        df = pd.DataFrame(columns=['record', 'label', 'spectrograms', 'subject'])
        for i in range(28):
            os.path.join(path, f'classe_{i}.dat')
            arr = np.fromfile(os.path.join(path, subfolder, f'classe_{i}.dat'), dtype=np.int16)
            arr = np.array(arr, dtype=np.float32)
            formatted = format_data_to_train(arr)
            formatted['label'] = i % 7
            formatted['spectrograms'] = start_val + i
            df = pd.concat([df, formatted], ignore_index=True)
        return df

    def get_dataset(path: str, subjects: list, subfolder: str, series_val: int = 0.):
        df = pd.DataFrame()
        for i, subject in enumerate(subjects):
            tmp_df = classe_to_df(os.path.join(path, subject), subfolder, series_val + i * 28)
            tmp_df['subject'] = i + series_val // 28
            df = pd.concat([df, tmp_df])
        return df

    path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'MyoArmband')
    eval_path = os.path.join(path, 'EvaluationDataset')
    pre_path = os.path.join(path, 'PreTrainingDataset')

    subjects = ['Female0', 'Female1', 'Male0', 'Male1', 'Male2', 'Male3', 'Male4', 'Male5', 'Male6', 'Male7', 'Male8',
                'Male9', 'Male10', 'Male11', 'Male12', 'Male13', 'Male14', 'Male15']
    dataset = get_dataset(eval_path, subjects, 'training0')
    dataset = pd.concat(
        [dataset, get_dataset(eval_path, subjects, 'Test0', 504)],
        ignore_index=True)
    dataset = pd.concat(
        [dataset, get_dataset(eval_path, subjects, 'Test1', 1008)],
        ignore_index=True)
    subjects2 = ['Female0', 'Female1', 'Female2', 'Female3', 'Female4', 'Female5', 'Female6', 'Female7', 'Female8',
                 'Female9', 'Male0', 'Male1', 'Male2', 'Male3', 'Male4', 'Male5', 'Male6', 'Male7', 'Male8', 'Male9',
                 'Male10', 'Male11']
    dataset2 = get_dataset(pre_path, subjects2, 'training0', 1512)
    final_dataset = pd.concat([dataset, dataset2], ignore_index=True)
    return final_dataset


def get_knibm_dataset(version: str = "low") -> pd.DataFrame:
    """
    Go through original files of given KNIBM dataset and convert them into form useful for computation in python.
    Args:
        version: name of knibm sub-dataset that should be processed
    Returns: Dataframe with dataset.
    """
    # this method transforms binary file to ndarray
    def bin_2_ndarray(input_file: str, data_bits: int = 8, channel_cnt: int = 8) -> np.ndarray:
        input_file = get_absolute_path(input_file)
        if data_bits == 12:
            byte_size = 2
        elif data_bits == 8:
            byte_size = 1
        else:
            print("data bits should be 8 or 12. Setting to 8")
            byte_size = 1

        file = open(input_file, 'rb')
        file.seek(0, 2)
        file_size = file.tell()
        file.seek(0, 0)

        loop_count = int(file_size / byte_size)
        records = []
        record = []
        for i in range(loop_count):
            val = 0
            data_bytes = file.read(byte_size)

            for j in range(len(data_bytes)):
                val = (val << 8) | data_bytes[j]

            record.append(val)
            if (i % channel_cnt) == channel_cnt - 1:
                records.append(np.array(record, dtype=float))
                record = []

        file.close()
        return np.array(records, dtype=float)

    # this method returns a series containing counts of unique records
    def records_counts(df: pd.DataFrame):
        return df["spectrograms"].value_counts()

    # this method finds minimal number of records for one recording
    def min_records(df: pd.DataFrame):
        return min(df["spectrograms"].value_counts())

    # this method cuts all recordings to 'min_records' number of records
    def cut_recordings(df: pd.DataFrame):
        rec_counts = records_counts(df).sort_index(ascending=True)
        min_rec = min_records(df)
        first_to_drop = min_rec
        indexes = []
        for spectrogram in rec_counts.index.values.tolist():
            indexes.extend(range(first_to_drop, first_to_drop + (rec_counts[spectrogram] - min_rec)))
            first_to_drop += rec_counts[spectrogram]
        df = df.drop(indexes).reset_index(drop=True)
        return df

    recordings = []
    labels = []
    series = []
    subjects = []
    for subject in range(1, 11):
        for session in range(1, 6):
            for gest in range(1, 9):
                data = bin_2_ndarray(os.path.join(
                    os.path.dirname(os.getcwd()),
                    'data',
                    f'knibm-{version}',
                    f'{str(subject)}',
                    f'{str(session)}',
                    f'{str(gest)}.bin'))
                recordings.extend([data[i] for i in range(data.shape[0])])
                labels.extend([gest - 1 for _ in range(data.shape[0])])
                series.extend([(subject - 1) * 40 + (session - 1) * 8 + gest - 1 for _ in range(data.shape[0])])
                subjects.extend([subject for _ in range(data.shape[0])])
    return cut_recordings(pd.DataFrame({'record': recordings, 'label': labels, 'spectrograms': series, 'subject': subjects}))


def save_arrays(dataframe: pd.DataFrame, name: str, path: os.path) -> pd.DataFrame:
    """
    Save all samples from dataset as np.arrays
    Args:
        dataframe: dataframe with samples to convert and save
        name: Name of dataset
        path: path to folder in which sub-folder for dataset will be created

    Returns:

    """
    paths = []
    base_path = os.path.join(path, name)
    for index, row in dataframe.iterrows():
        current_path = os.path.join(base_path, f'{index // 200000}')
        if not os.path.exists(current_path):
            os.makedirs(current_path)
        path = os.path.join(current_path, f'{name}_{index}.npy')
        np.save(path, row['record'])
        paths.append(path)

    df: pd.DataFrame = pd.DataFrame({'path': paths, 'label': dataframe['label'], 'spectrograms': dataframe['spectrograms'],
                                     'subject': dataframe['subject']})
    df.to_csv(os.path.join(base_path, f'{name}.csv'), index=False)
    return df
