import physionet as pc
import  ecg_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences

def plot_ecg(path):

    ecg_data = pc.load_challenge_data(path)
    ecg_plot.plot(ecg_data[0] / 1000, sample_rate=500, title='')
    ecg_plot.show()



plot_ecg("/kaggle/input/china-12lead-ecg-challenge-database/Training_2/Q0948.mat")
