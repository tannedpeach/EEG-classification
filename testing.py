# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import tensorflow as tf
# import tensorflow_addons as tfa
# from datetime import datetime
# import pickle
# from sklearn.preprocessing import LabelBinarizer
# from tensorflow.keras.preprocessing import image_dataset_from_directory
# from sklearn import metrics,utils
# import itertools
# import seaborn as sns
# import math
# from tensorflow_utils import *
# from tensorboard.plugins.hparams import api as hp
# from cnn_params import *
# import pandas as pd
# from cnn import *
# #from cnn import model
#
# #filename = 'finalized_model.sav'
# #loaded_model = pickle.load(open(filename, 'rb'))
# #result = loaded_model.score(X_test, Y_test)

# seizure_folds = pickle.load(open(cross_val_file, "rb"))
# #
# seizure_folds = list(seizure_folds.values())
# #
# k_validation_folds = seizure_folds[:90]  # TODO: Choose which dataset will be the test randomly
# test_fold = seizure_folds[-1]
# test_dataset = get_test_dataset(data_dir, test_fold, le)
# for example in test_dataset:
#     dataset = example[0].numpy()
#X_val, y_val = get_fold_data(data_dir, fold_data, "val", le)
#print(X_val, y_val)
#print(tf.data.Dataset.from_tensor_slices((X, y)).map(resize_eeg, num_parallel_calls=tf.data.experimental.AUTOTUNE))
#
# evaluate the model

# print("[INFO] evaluating network...")
# scores = loaded_model.evaluate(dataset, verbose=0)
#
#
# with tf.summary.create_file_writer(logs_dir + '/run').as_default():
#     tf.summary.scalar(loaded_model.metrics_names[5], scores[5], step=1)
#     tf.summary.scalar(loaded_model.metrics_names[6], scores[6], step=1)
#
#
# print(f'Score for fold {fold_no}: {loaded_model.metrics_names[0]} of {scores[0]}; {loaded_model.metrics_names[1]} of {scores[1]*100}%')
# acc_per_fold.append(scores[1] * 100)
# loss_per_fold.append(scores[0])
#
#
# print("[INFO] Predicting network and creating confusion matrix...")
#
# test_pred = np.argmax(loaded_model.predict(dataset), axis=1)
# save_confusion_matrix(test_labels, test_pred, le.classes_, logs_dir)
#
#
# # # == Provide average scores ==
# print('------------------------------------------------------------------------')
# print('Score per fold')
# for i in range(0, len(acc_per_fold)):
#     print('------------------------------------------------------------------------')
#     print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
# print('------------------------------------------------------------------------')
# print('Average scores for all folds:')
# print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
# print(f'> Loss: {np.mean(loss_per_fold)}')
# print('------------------------------------------------------------------------')

# dataset=print(dataset)
# # test_pred = np.argmax(model.predict(dataset), axis=1)

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
import pickle
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn import metrics, utils
import itertools
import seaborn as sns
import math
from tensorflow_utils import *
from tensorboard.plugins.hparams import api as hp
from cnn_params import *
from cnn import *
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

METRICS = [
      tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tfa.metrics.F1Score(name='f1_score_macro', average='macro', num_classes=7),
      tfa.metrics.F1Score(name='f1_score_weighted', average='weighted', num_classes=7)
]

le = LabelBinarizer()
le.fit(SZR_CLASSES)
# # tf.compat.v1.enable_eager_execution()
# # tf.compat.v1.disable_eager_execution()
# # tf.debugging.set_log_device_placement(False)
#
# # Cross validation inpired in https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/
#
# # gpus = tf.config.experimental.list_physical_devices('GPU')
# # if gpus:
# #         for gpu in gpus:
# #                 tf.config.experimental.set_memory_growth(gpu, True)
# # else:
# #         raise SystemError("NO GPUS")
#
#
# # config.set_visible_devices([], 'GPU')
# # print("GPUS {}", gpus)
#
#
# # physical_devices = tf.config.list_physical_devices('GPU')
# # tf.config.experimental.set_memory_growth(physical_devices[0], True)
#
#
# METRICS = [
#     tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
#     tf.keras.metrics.Precision(name='precision'),
#     tf.keras.metrics.Recall(name='recall'),
#     tf.keras.metrics.AUC(name='auc'),
#     tfa.metrics.F1Score(name='f1_score_macro', average='macro', num_classes=7),
#     tfa.metrics.F1Score(name='f1_score_weighted', average='weighted', num_classes=7)
# ]
#
#
# def get_fold_data(data_dir, fold_data, dataType, labelEncoder):
#     data = fold_data.get(dataType)
#     #print("data", data)
#     X = np.empty((len(data), EEG_WINDOWS, EEG_COLUMNS), dtype=np.float64)
#     y = list()
#
#     # print("NJ111111111, ", data)
#     print("TR, ", len(data))
#
#     for i, fname in enumerate(data):
#
#         # each file contains a named tupple
#         # 'patient_id','seizure_type', 'data'
#         seizure = pickle.load(open(os.path.join(data_dir, fname), "rb"))
#         #if (seizure.patient_id == "00004456"):
#         # print(seizure.data)
#         # print("shape:", len(seizure.data))
#         # print("fname:", fname)
#         # print("seizure:", seizure.patient_id)
#         y.append(seizure.seizure_type)
#         length = len(seizure.data)
#         if (EEG_WINDOWS > length):
#             X[i] = np.pad(seizure.data, ((0, EEG_WINDOWS - length), (0, 0)))
#         else:
#             X[i] = np.resize(seizure.data, (EEG_WINDOWS, len(seizure.data[0])))
#
#     if labelEncoder != None:
#         #if (seizure.patient_id == "00004456"):
#         y = labelEncoder.transform(y)
#     #print("X:", X)
#     #print("y:", y)
#     return X, y
#
#
# @tf.function
# def resize_eeg(data, label):
#     return tf.stack([data, data, data],
#                     axis=-1), label  # tf.py_function(lambda: tf.convert_to_tensor(np.stack([data.numpy()]*3, axis=-1), dtype=data.dtype), label, Tout=[tf.Tensor, type(label)] )
#
#
# def get_dataset(X, y):
#     return tf.data.Dataset.from_tensor_slices((X, y)).map(resize_eeg, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#
#
# def get_fold_datasets(data_dir, fold_data, le, class_probs=None, oversample=False, undersample=False):
#     print('get_fold_datasets method: ', type(fold_data))
#
#     X_train, y_train = get_fold_data(data_dir, fold_data, "train", le)
#     X_val, y_val = get_fold_data(data_dir, fold_data, "val", le)
#
#     print("DIMENSIONS: ", len(X_train), ",", len(X_train[0]))
#     train_dataset = get_dataset(X_train, y_train)
#     class_target_probs = None
#     if class_probs:
#         class_target_probs = {label_name: 0.5 for label_name in class_probs}
#         class_target_probs = tf.lookup.StaticHashTable(
#             tf.lookup.KeyValueTensorInitializer(tf.constant(list(class_target_probs.keys())),
#                                                 tf.constant(list(class_target_probs.values()),
#                                                             dtype=tf.dtypes.float64)),
#             default_value=0
#         )
#         class_probs = tf.lookup.StaticHashTable(
#             tf.lookup.KeyValueTensorInitializer(tf.constant(list(class_probs.keys())),
#                                                 tf.constant(list(class_probs.values()), dtype=tf.dtypes.float64)),
#             default_value=0
#         )
#
#     if oversample:
#         # train_dataset = train_dataset.map(, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#         train_dataset = train_dataset.flat_map(lambda x, y: tf.data.Dataset.from_tensors((x, y)).repeat(
#             oversample_classes(y, class_probs, class_target_probs)))
#     if undersample:
#         # train_dataset = train_dataset.map()
#         train_dataset = train_dataset.filter(lambda x, y: undersampling_filter(x, y, class_probs, class_target_probs))
#
#     if oversample or undersample:
#         train_dataset = train_dataset.shuffle(300).repeat()
#
#     train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(PREFETCH)
#     val_dataset = get_dataset(X_val, y_val).batch(BATCH_SIZE).prefetch(PREFETCH)
#
#     return train_dataset, val_dataset
#
#
# def get_test_dataset(data_dir, fold_data, le):
#     X_train, y_train = get_fold_data(data_dir, fold_data, "train", le)
#     X_val, y_val = get_fold_data(data_dir, fold_data, "val", le)
#
#     train_dataset = get_dataset(X_train, y_train)
#     val_dataset = get_dataset(X_val, y_val)
#     #print("val data: ", fold_data)
#     print(train_dataset.concatenate(val_dataset).batch(BATCH_SIZE).prefetch(PREFETCH))
#     #print("X_val:", X_val)
#     #print("y_val:", y_val)
#     return train_dataset.concatenate(val_dataset).batch(BATCH_SIZE).prefetch(PREFETCH)
#
#
# # data_dir ="/media/david/Extreme SSD1/Machine Learning/raw_data/fft_with_time_freq_corr/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_24" #"/home/david/Documents/Machine Learning/raw_data/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_12"
#
# # Modified by NJ
# data_dir = "C:/Users/tanis/OneDrive/Desktop/IoT Project/Code/Shared_Final/data_preparation/out_datasets/tuh/seizure_type_classification/v1.5.2/fft_with_time_freq_corr/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_24"  # "./data_preparation/out_datasets/tuh/seizure_type_classification/v1.5.2/fft_with_time_freq_corr/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_24"
# # cross_val_file = "../seizure-type-classification-tuh/data_preparation/cv_split_3_fold_patient_wise_v1.5.2.pkl"
# cross_val_file = "data_preparation/cv_split_3_fold_patient_wise_v1.5.2.pkl"  # "./data_preparation/cv_split_3_fold_patient_wise_v1.5.2.pkl"
# cross_val_file_2 = "C:/Users/tanis/OneDrive/Desktop/IoT Project/Code/Shared_Final/data_preparation/szr_49_pid_00006546_type_FNSZ.pkl"
# seizure_folds = pickle.load(open(cross_val_file_2, "rb"))
# print(seizure_folds)
# test_fold = seizure_folds[-1]
# #seizure_folds = list(seizure_folds.values())
# print(seizure_folds)
# k_validation_folds = seizure_folds[:-1]  # TODO: Choose which dataset will be the test randomly
# test_fold = seizure_folds[-1]
# # #get_dataset(X_val, y_val)
# #data = test_fold.get("train")
# #for i, fname in enumerate(data):
#     #seizure = pickle.load(open(os.path.join(data_dir, fname), "rb"))
#     #if (seizure.patient_id == "00004456"):
#         #test_dataset = get_test_dataset(data_dir, test_fold, le)
#         #print("test: ", test_dataset)
# print("test fold", test_fold)
# # print("test")

json_file = open('C:/Users/tanis/OneDrive/Desktop/IoT Project/Code/Shared_Final/model.json', 'r')
loaded_model_json = json_file. read()
json_file. close()
loaded_model = model_from_json(loaded_model_json)


def get_dataset(X, y):
    return tf.data.Dataset.from_tensor_slices((X, y)).map(resize_eeg, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_fold_data(data_dir, fold_data, dataType, labelEncoder):
    data = fold_data.get(dataType)
    X = np.empty((len(data), EEG_WINDOWS, EEG_COLUMNS), dtype=np.float64)
    y = list()

    # print("NJ111111111, ", data)
    print("TR, ", len(data))

    for i, fname in enumerate(data):
        # each file contains a named tuple
        # 'patient_id','seizure_type', 'data'
        seizure = pickle.load(open(os.path.join(data_dir, fname), "rb"))
        print("patient_id", seizure.patient_id)
        print("seizure_type", seizure.seizure_type)
        print("data", seizure.data)
        y.append(seizure.seizure_type)
        length = len(seizure.data)
        if (EEG_WINDOWS > length):
            X[i] = np.pad(seizure.data, ((0, EEG_WINDOWS - length), (0, 0)))
        else:
            X[i] = np.resize(seizure.data, (EEG_WINDOWS, len(seizure.data[0])))

    if labelEncoder != None:
        y = labelEncoder.transform(y)
    print("X,y: ", X, y)
    print("done")
    return X, y


def get_test_dataset(data_dir, fold_data, le):
    X_train, y_train = get_fold_data(data_dir, fold_data, "1", le)
    #X_val, y_val = get_fold_data(data_dir, fold_data, "val", le)

    train_dataset = get_dataset(X_train, y_train)
    #val_dataset = get_dataset(X_val, y_val)
    return train_dataset.batch(BATCH_SIZE).prefetch(PREFETCH)


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=class_names, yticklabels=class_names,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.title("Confusion matrix")
    return figure


# Use code
cross_val_file = "data_preparation/cv_split_3_fold_patient_wise_v1.5.2.pkl"
# seizure_folds = pickle.load(open(cross_val_file, "rb"))
# seizure_folds = list(seizure_folds.values())
# k_validation_folds = seizure_folds[:-1]  # TODO: Choose which dataset will be the test randomly
# fold_data = seizure_folds[-1]
# data_dir ="C:/Users/tanis/OneDrive/Desktop/IoT Project/Code/Shared_Final/data_preparation/out_datasets/tuh/seizure_type_classification/v1.5.2/fft_with_time_freq_corr/fft_seizures_wl1_ws_0.5_sf_250_fft_min_1_fft_max_24"
data_dir = "C:/Users/tanis/OneDrive/Desktop/IoT Project/Code/Shared_Final/data_preparation"
fold_data = {"1": ['szr_2891_pid_00008889_type_TNSZ.pkl']}  #, 'szr_50_pid_00006546_type_FNSZ.pkl', 'szr_51_pid_00006546_type_FNSZ.pkl', 'szr_52_pid_00006546_type_FNSZ.pkl']}
cross_val_file_2 = "C:/Users/tanis/OneDrive/Desktop/IoT Project/Code/Shared_Final/data_preparation/szr_49_pid_00006546_type_FNSZ.pkl"
# seizure_folds = pickle.load(open(cross_val_file_2, "rb"))
#X_t, y_t = get__data(data_dir, fold_data, "1", le)
#seizure_folds = pickle.load(open(cross_val_file, "rb"))

#seizure_folds = list(seizure_folds.values())

#k_validation_folds = seizure_folds[:-1]
t_dataset = get_test_dataset(data_dir, fold_data, le)
#print("dataset: ", t_dataset)
test_labels = np.concatenate([y for x, y in t_dataset], axis=0).argmax(axis=1)
currentTime = datetime.now().strftime("%Y%m%d-%H%M%S")
# acc_per_fold = []
# loss_per_fold = []
# for fold_no, fold_data in enumerate(k_validation_folds):
#     logs_dir = f"logs/fit/{currentTime}/fold_{fold_no}"
#print(t_dataset)
# print(seizure_folds)
#final = t_dataset.batch(BATCH_SIZE).prefetch(PREFETCH)
#test_fold = seizure_folds[-1]


#print("data shape", test_fold.shape())
#loaded_model = "C:/Users/tanis/OneDrive/Desktop/IoT Project/Code/Shared_Final/model.json"
loaded_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.1),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=METRICS)
#print("Shape", EEG_SHAPE + (3,))


print("[INFO] evaluating network...")

scores = loaded_model.evaluate(t_dataset, verbose=0)
SZR_CLASSES = ['TNSZ', 'SPSZ', 'ABSZ', 'TCSZ', 'CPSZ', 'GNSZ', 'FNSZ']
print(SZR_CLASSES)
print("scores: ", scores)


# with tf.summary.create_file_writer(logs_dir + '/run').as_default():
#     tf.summary.scalar(loaded_model.metrics_names[5], scores[5], step=1)
#     tf.summary.scalar(loaded_model.metrics_names[6], scores[6], step=1)

# print(
#     f'Score for fold {loaded_model.metrics_names[0]} of {scores[0]}; {loaded_model.metrics_names[1]} of {scores[1] * 100}%')
# acc_per_fold.append(scores * 100)
# loss_per_fold.append(scores[0])

print("[INFO] Predicting network and creating confusion matrix...")

test_pred = np.argmax(loaded_model.predict(t_dataset), axis=1)
#save_confusion_matrix(test_labels, test_pred, le.classes_, logs_dir)
cm = tf.math.confusion_matrix(test_labels, test_pred)
classes = le.classes_
class_names = classes
print(cm)

loaded_model.summary()

for example in t_dataset:
    print(example[0].numpy())

#         #print("test: ", test_dataset)
# # == Provide average scores ==
# print('------------------------------------------------------------------------')
# print('Score per fold')
# for i in range(0, len(acc_per_fold)):
#     print('------------------------------------------------------------------------')
#     print(f'> Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
# print('------------------------------------------------------------------------')
# print('Average scores for all folds:')
# print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
# print(f'> Loss: {np.mean(loss_per_fold)}')
