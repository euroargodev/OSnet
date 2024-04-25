#cd /home1/datahome/epauthen/All_depth
#qsub NN_train.pbs
#qstat -u epauthen

#################################
#3 layers, 256 neurons
#resources_used.walltime=00:38:40
#6 layers, 256 neurons
#resources_used.walltime=00:45:36
#6 layers, 256,512,512,512,512,256  (6Lb)
#resources_used.walltime=00:51:08
#################################

import logging
import time
from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import xarray as xr
from joblib import dump
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import max_norm

path_model = "/home/datawork-lops-bluecloud/osnet/models/all_depth500_6L"

#Load dataset and fill NaN with 0
#x_full = xr.open_dataset("/home/datawork-lops-bluecloud/osnet/data_cora_raw/CORA_alldepth_bathy100_1993-2019.nc")
x_full = xr.open_dataset("/home/datawork-lops-bluecloud/osnet/data_cora_raw/CORA_alldepth_bathy500_1993-2019.nc")


#Split in test and train
test_index = np.random.choice(x_full['N_PROF'].data, size=int(len(x_full['N_PROF'])*0.2), replace=False)
x_test = x_full.sel(N_PROF=test_index)
x = x_full.drop_sel(N_PROF=test_index)

X = np.zeros([len(x['N_PROF']),12])
X[:,0] = x['SLA'].data
X[:,1] = x['latitude'].data
X[:,2] = x['longitude'].data
X[:,3] = x['cos_day'].data
X[:,4] = x['sin_day'].data
X[:,5] = x['MDT'].data
X[:,6] = x['UGOSA'].data
X[:,7] = x['VGOSA'].data
X[:,8] = x['UGOS'].data
X[:,9] = x['VGOS'].data
X[:,10] = x['SST'].data
X[:,11] = -x['bathy'].data

scaler_input = preprocessing.StandardScaler().fit(X)
X_scaled = scaler_input.transform(X)
dump(scaler_input, f'{path_model}/scaler_input.joblib')

Tm = x.TEMP_INTERP.mean().data
Tstd = x.TEMP_INTERP.std().data
Sm = x.PSAL_INTERP.mean().data
Sstd = x.PSAL_INTERP.std().data

T_scaled = (x.TEMP_INTERP-Tm)/Tstd
S_scaled = (x.PSAL_INTERP-Sm)/Sstd

T_scaled = T_scaled.fillna(0)
S_scaled = S_scaled.fillna(0)

y_scaled = np.zeros([len(x['N_PROF']),114, 3])

y_scaled[:,:,0] = T_scaled
y_scaled[:,:,1] = S_scaled
y_scaled[:,:,2] = x.MLD_mask.data

dump(Tm, f'{path_model}/Tm.joblib')
dump(Tstd, f'{path_model}/Tstd.joblib')
dump(Sm, f'{path_model}/Sm.joblib')
dump(Sstd, f'{path_model}/Sstd.joblib')

# Model architecture
def get_model_architecture():
    max_val_weight = 2.0
    size_init_tensor = 114
    inputs = keras.Input(shape=(12,))
    mask_T = keras.Input(shape=(size_init_tensor, ))
    mask_S = keras.Input(shape=(size_init_tensor, ))
    
    x = layers.Dense(256, kernel_constraint=max_norm(max_val_weight))(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(512, kernel_constraint=max_norm(max_val_weight), activation="relu")(x)

    x = layers.Dense(512, kernel_constraint=max_norm(max_val_weight), activation="relu")(x)
    
    x = layers.Dense(512, kernel_constraint=max_norm(max_val_weight), activation="relu")(x)
    
    x = layers.Dense(512, kernel_constraint=max_norm(max_val_weight), activation="relu")(x)

    x = layers.Dense(256, kernel_constraint=max_norm(max_val_weight), activation="relu")(x)

    T = layers.Dense(size_init_tensor, kernel_constraint=max_norm(max_val_weight), activation="linear")(x)

    S = layers.Dense(size_init_tensor, kernel_constraint=max_norm(max_val_weight), activation="linear")(x)

    MLD = layers.Dense(size_init_tensor, kernel_constraint=max_norm(max_val_weight), activation="sigmoid")(x)
    MLD = layers.Reshape((size_init_tensor, 1))(MLD)

    Tmask = layers.multiply([T, mask_T])
    Tmask = layers.Reshape((size_init_tensor, 1))(Tmask)

    Smask = layers.multiply([S, mask_S])
    Smask = layers.Reshape((size_init_tensor, 1))(Smask)
    
    output = layers.concatenate([Tmask, Smask, MLD])

    model = keras.Model(inputs=[inputs, mask_T, mask_S], outputs=output, name="all_depth")
    return model

opt = tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7,
    amsgrad=False)

size_ensemble = 15
ensemble = []
for i in range(size_ensemble):
    # get NN
    model = get_model_architecture()

    # compile with loss and coefs
    model.compile(loss='mse', optimizer=opt, metrics=['mae'])
    ensemble.append(model)

checkpoint_filepath = './tmp/checkpoint'
batch_size = 32
epochs = 100

# callbacks
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=100, 
                                               mode='auto', restore_best_weights=True)
# Model training
history = []
for nb_model, model in enumerate(ensemble):
    # training loop
    start_time = time.time()
    history.append(model.fit([X_scaled, x.mask_depthT.data, x.mask_depthS.data],
                             y_scaled,
                             batch_size=batch_size, 
                             epochs=epochs,
                             callbacks=[early_stopping],
                             validation_split=0.2,
                             verbose=0))
    print(f"finished training nb {nb_model} in {time.time()-start_time}s")
    print("################################################")
    
    
for i, model in enumerate(ensemble):
    model.save(f"{path_model}/neuralnet/ensemble/model{i}")
    
