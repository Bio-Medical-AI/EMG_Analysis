"""
All important constants from project, mostly paths to project folder on device, data folder and others.
"""
import os

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(ROOT_DIR, 'data')
PKL_FOLDER = os.path.join(os.path.dirname(ROOT_DIR), 'Data_pkl')
FRAMES_FOLDER = os.path.join(os.path.dirname(ROOT_DIR), 'Data')
SERIES_FOLDER = PKL_FOLDER
MODELS_FOLDER = os.path.join(ROOT_DIR, 'saved_models')

if not os.path.exists(MODELS_FOLDER):
    os.mkdir(MODELS_FOLDER)

WANDB_PROJECT_NAME = 'EMG Armband'
