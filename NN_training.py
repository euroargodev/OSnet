#date : February 2022
#author : Loic Bachelot (loic.bachelot@gmail.com) & Etienne Pauthenet (etienne.pauthenet@gmail.com)

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


def get_args():
    """
    Extract arguments from command line

    @return: parse.parse_args(): dict of the arguments
    """
    import argparse

    parse = argparse.ArgumentParser(description="Method to train OSNet")
    parse.add_argument('input_dataset', type=str, help='path of training dataset')
    parse.add_argument('path_models', type=str, help='path to model folder to save all scalers, NN and pca')
    parse.add_argument('path_out', type=str, help='path to output directory')
    parse.add_argument('nb_models', type=int, help='number of model in ensemble')
    parse.add_argument('use_all_data', type=int, help='do a train test split')
    return parse.parse_args()


def load_dataset(path_ds, use_all_data):
    """
    load dataset and format it for the method

    @param path_ds: path to input dataset
    @type path_ds: String
    @param use_all_data: if true, the full ds will be used for the train (train+ validation). Else, it will be split between train and test
    @type use_all_data: Bool
    @return: One or 2 formatted datasets
    @rtype: xarray dataset
    """
    depth_levels = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90,
                    100, 110, 120, 133, 147, 163, 180, 199, 221, 245, 271,
                    301, 334, 371, 412, 458, 509, 565, 628, 697, 773, 857, 950, 1000]

    x_full = xr.open_dataset(path_ds)
    day = np.array(pd.DatetimeIndex(x_full['TIME'].data).dayofyear).astype(np.int32)
    x_full = x_full.assign(variables={"dayOfYear": (('N_PROF'), day)})

    # train/test split
    if not use_all_data:
        size_test = 0.2
        test_index = np.random.choice(x_full['N_PROF'].data, size=int(len(x_full['N_PROF']) * size_test), replace=False)
        x_test = x_full.sel(N_PROF=test_index)
        x_train = x_full.drop_sel(N_PROF=test_index)
        return x_train, x_test
    else:
        return x_full, 0


def format_input(x, scaler_input=None):
    """
    Scale the different variables
    Create a numpy array of the input

    @param x: input dataset
    @type x: xarray dataset
    @param scaler_input: standard scaler if already fitted on data
    @type scaler_input: Scikit learn standard scaler
    @return: Scaled numpy array with the inputs, Standard scaler used for the scaling
    @rtype: numpy ndarray, Scikit learn standard scaler
    """
    d = 1 / 365
    cos_day = np.cos(np.pi * 2 * d * x['dayOfYear'].data)
    sin_day = np.sin(np.pi * 2 * d * x['dayOfYear'].data)

    X = np.zeros([len(x['N_PROF']), 12])
    X[:, 0] = x['SLA'].data
    X[:, 1] = x['LATITUDE'].data
    X[:, 2] = x['LONGITUDE'].data
    X[:, 3] = cos_day
    X[:, 4] = sin_day
    X[:, 5] = x['MDT'].data
    X[:, 6] = x['UGOSA'].data
    X[:, 7] = x['VGOSA'].data
    X[:, 8] = x['UGOS'].data
    X[:, 9] = x['VGOS'].data
    X[:, 10] = x['SST'].data
    X[:, 11] = -x['bathy'].data

    if scaler_input is None:
        scaler_input = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler_input.transform(X)
    return X_scaled, scaler_input


def format_target(x, scal_Sm=None, scal_Sstd=None, scal_Tm=None, scal_Tstd=None, SIGm=None, SIGstd=None):
    """
    Scale the different target variables using
    Create a numpy array of the targets

    @param x: input dataset
    @type x: xarray dataset
    @param scal_Sm: mean of PSAL variable per depth
    @type scal_Sm: numpy nd array float64
    @param scal_Sstd: std of PSAL variable per depth
    @type scal_Sstd: numpy nd array float64
    @param scal_Tm: mean of TEMP variable per depth
    @type scal_Tm: numpy nd array float64
    @param scal_Tstd: std of TEMP variable per depth
    @type scal_Tstd: numpy nd array float64
    @param SIGm: mean of SIG variable per depth
    @type SIGm: numpy nd array float64
    @param SIGstd: std of SIG variable per depth
    @type SIGstd: numpy nd array float64
    @return: Scaled target, and all scalers parameters
    @rtype: numpy ndarray, numpy ndarray, numpy ndarray, numpy ndarray, numpy ndarray, numpy ndarray, numpy ndarray
    """
    if scal_Sm is None:
        scal_Sm = x.PSAL.mean().data
        scal_Sstd = x.PSAL.std().data
        scal_Tm = x.TEMP.mean().data
        scal_Tstd = x.TEMP.std().data
        SIGm = x.SIG.mean().data
        SIGstd = x.SIG.std().data

    def get_mask(mld):
        depth_levels = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 40, 45, 50, 55,
                        60, 65, 70, 75, 80, 90, 100, 110, 120, 133, 147, 163, 180, 199, 221, 245, 271,
                        301, 334, 371, 412, 458, 509, 565, 628, 697, 773, 857, 950, 1000]
        mask = np.sign(np.array(depth_levels) - mld)
        return np.where(mask >= 0, 1, 0)

    get_mask_vect = np.vectorize(get_mask, signature='()->(k)')
    MLD_mask = get_mask_vect(x.MLD.data)
    S_scaled = (x.PSAL - scal_Sm) / scal_Sstd
    T_scaled = (x.TEMP - scal_Tm) / scal_Tstd

    y_scaled = np.zeros([len(x['N_PROF']), 51, 3])

    y_scaled[:, :, 0] = S_scaled
    y_scaled[:, :, 1] = T_scaled
    y_scaled[:, :, 2] = MLD_mask

    return y_scaled, scal_Sm, scal_Sstd, scal_Tm, scal_Tstd, SIGm, SIGstd


def get_model_architecture():
    """
    Create the neural network architecture using Keras

    @return: a neural network model
    @rtype: keras.Model
    """
    max_val_weight = 2.0
    size_init_tensor = 51
    inputs = keras.Input(shape=(12,))
    x = layers.Dense(256, kernel_constraint=max_norm(max_val_weight))(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(256, kernel_constraint=max_norm(max_val_weight), activation="relu")(x)

    TS = layers.Dense(size_init_tensor * 2, kernel_constraint=max_norm(max_val_weight), activation="linear")(x)
    TS = layers.Reshape((size_init_tensor, 2))(TS)

    MLD = layers.Dense(size_init_tensor, kernel_constraint=max_norm(max_val_weight), activation="sigmoid")(x)
    MLD = layers.Reshape((size_init_tensor, 1))(MLD)

    output = layers.concatenate([TS, MLD])

    model = keras.Model(inputs=inputs, outputs=output, name="full")
    return model


def get_optimizer(lr=0.0001, beta_1=0.9, beta_2=0.999, eps=1e-7):
    """
    Get the optimizer for the training, setting the parameters of the Keras implementation of Adam
    See keras doc for more details on the parameters
    @return: optimizer for the training
    @rtype: tf.keras.optimizers
    """
    return tf.keras.optimizers.Adam(learning_rate=lr,
                                    beta_1=beta_1,
                                    beta_2=beta_2,
                                    epsilon=eps,
                                    amsgrad=False)


def get_MLD_from_mask(mask):
    """
    compute the MLD from the mask
    @param mask: mask of the MLD
    @type mask: numpy ndarray
    @return: the depth value of the MLD
    @rtype: int
    """
    depth_levels = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 40, 45, 50, 55,
                    60, 65, 70, 75, 80, 90, 100, 110, 120, 133, 147, 163, 180, 199, 221, 245, 271,
                    301, 334, 371, 412, 458, 509, 565, 628, 697, 773, 857, 950, 1000]
    mask = np.sign(0.5 - mask)
    return depth_levels[np.argmin(mask)]


def rmse_from_ae(ae):
    """
    Compute the root mean squared error on the depth axis
    @param ae: absolute error per profile and per depth
    @type ae: numpy ndarray
    @return: rmse from the absolute error
    @rtype: numpy ndarray
    """
    return np.sqrt(np.nanmean((ae ** 2), axis=0))


def get_mean_std_pred(ensemble, X_scaled, ds, scal_Sm, scal_Sstd, scal_Tm, scal_Tstd):
    """
    Predict using the trained ensemble

    @param ensemble: list of keras models
    @type ensemble: list of Keras.Model
    @param X_scaled: input array preprocessed by the format_input function
    @type X_scaled: numpy ndarray
    @param ds: dataset with true values
    @type ds: xarray dataset
    @param scal_Sm: mean of PSAL variable per depth
    @type scal_Sm: numpy nd array float64
    @param scal_Sstd: std of PSAL variable per depth
    @type scal_Sstd: numpy nd array float64
    @param scal_Tm: mean of TEMP variable per depth
    @type scal_Tm: numpy nd array float64
    @param scal_Tstd: std of TEMP variable per depth
    @type scal_Tstd: numpy nd array float64
    @return: mean predictions for TEMP and PSAL, standard deviation of predictions TEMP and PSAL, mean prediction of MLD mask and rmse for TEMP and PSAL per model
    @rtype: numpy nd_array, numpy nd_array, numpy nd_array, numpy nd_array, numpy nd_array, numpy nd_array, numpy nd_array
    """
    predS = []
    predT = []
    mld = []
    rmse_T = []
    rmse_S = []
    for model in ensemble:
        tmp_pred = model.predict(X_scaled)
        temp = tmp_pred[:, :, 1] * scal_Tstd + scal_Tm
        psal = tmp_pred[:, :, 0] * scal_Sstd + scal_Sm
        predS.append(psal)
        predT.append(temp)
        mld.append(tmp_pred[:, :, 2])
        rmse_T.append(rmse_from_ae(ds['TEMP'].data - temp))
        rmse_S.append(rmse_from_ae(ds['PSAL'].data - psal))

    return np.mean(predS, axis=0), np.std(predS, axis=0), np.mean(predT, axis=0), np.std(predT, axis=0), np.mean(mld,axis=0), rmse_T, rmse_S


def add_sig(ds):
    """
    Compute sigma0 and add it to the dataset using gsw package
    @param ds: dataset containing the predictions
    @type ds: xarray dataset
    @return: dataset with the added variable SIG_predicted (sigma0 from the prediction of TS)
    @rtype: xarray dataset
    """
    SA = gsw.SA_from_SP(ds['PSAL_predicted'], ds['PRES_INTERPOLATED'], ds['LONGITUDE'], ds['LATITUDE'])
    CT = gsw.CT_from_t(SA, ds['TEMP_predicted'], ds['PRES_INTERPOLATED'])
    sig = gsw.sigma0(SA, CT)
    ds = ds.assign(variables={"SIG_predicted": (('N_PROF', 'PRES_INTERPOLATED'), sig.data)})
    return ds


def save_models(ensemble, Sm, Sstd, Tm, Tstd, SIGm, SIGstd, pw, scaler_input, path_model):
    """
    Save all the scalers and the ensemble models

    @param ensemble: list of keras models
    @type ensemble: list of Keras.Model
    @param Sm: mean of PSAL variable per depth
    @type Sm: numpy nd array float64
    @param Sstd: std of PSAL variable per depth
    @type Sstd: numpy nd array float64
    @param Tm: mean of TEMP variable per depth
    @type Tm: numpy nd array float64
    @param Tstd: std of TEMP variable per depth
    @type Tstd: numpy nd array float64
    @param SIGm: mean of SIG variable per depth
    @type SIGm: numpy nd array float64
    @param SIGstd: std of SIG variable per depth
    @type SIGstd: numpy nd array float64
    @param pw: delta between the depth levels
    @type pw: numpy ndarray
    @param scaler_input: standard scaler if already fitted on data
    @type scaler_input: Scikit learn standard scaler
    @param path_model: path to the folder where to save the files
    @type path_model: str
    """
    dump(scaler_input, f'{path_model}/scaler_input.joblib')
    dump(Sm, f'{path_model}/Sm.joblib')
    dump(Sstd, f'{path_model}/Sstd.joblib')
    dump(Tm, f'{path_model}/Tm.joblib')
    dump(Tstd, f'{path_model}/Tstd.joblib')
    dump(SIGm, f'{path_model}/SIGm.joblib')
    dump(SIGstd, f'{path_model}/SIGstd.joblib')
    dump(pw, f'{path_model}/pw.joblib')
    for i, model in enumerate(ensemble):
        model.save(f"{path_model}/neuralnet/ensemble/model{i}")


def train_ensemble(X_scaled, y_scaled, nb_models):
    """
    Performs the training of the bootstrap ensemble:
    - creates the requested number of models
    - trains the models with the provided inputs
    @param X_scaled: Preprocessed inputs (using the format_input function)
    @type X_scaled: numpy ndarray
    @param y_scaled: Preprocessed targets (using the format_target function)
    @type y_scaled: numpy ndarray
    @param nb_models: number of models to create in the ensemble
    @type nb_models: int
    @return: ensemble of trained models
    @rtype: list of Keras.Model
    """
    size_ensemble = nb_models
    ensemble = []
    for i in range(size_ensemble):
        # get NN
        model = get_model_architecture()
        opt = get_optimizer()

        # compile with loss and coefs
        model.compile(loss='mse', optimizer=opt, metrics=['mae'])
        ensemble.append(model)

    # train  NN
    batch_size = 32
    epochs = 1000
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=100, mode='auto',
                                                   restore_best_weights=True)
    history = []
    for nb_model, model in enumerate(ensemble):
        start_time = time.time()
        # training loop
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
        history.append(model.fit(X_train, y_train,
                                 batch_size=batch_size, epochs=epochs,
                                 callbacks=[early_stopping],
                                 validation_data=(X_val, y_val),
                                 verbose=0))
        logging.info(f"finished training nb {nb_model} in {time.time() - start_time}s")
        logging.info(f"val loss min: {np.min(history[nb_model].history['val_loss']).round(6)}")
    return ensemble


def predict_dataset(ensemble, X_scaled, x, Sm, Sstd, Tm, Tstd):
    """
    Uses the trained ensemble to make predictions on temperature, salinity, MLD and sigma0

    @param ensemble: list of keras models
    @type ensemble: list of Keras.Model
    @param X_scaled: Preprocessed inputs (using the format_input function)
    @type X_scaled: numpy ndarray
    @param x: input dataset where the predicted variables will be added
    @type x: xarray dataset
    @param Sm: mean of PSAL variable per depth
    @type Sm: numpy nd array float64
    @param Sstd: std of PSAL variable per depth
    @type Sstd: numpy nd array float64
    @param Tm: mean of TEMP variable per depth
    @type Tm: numpy nd array float64
    @param Tstd: std of TEMP variable per depth
    @type Tstd: numpy nd array float64
    @return: dataset with all the predicted variables
    @rtype: xarray dataset
    """
    get_MLD_from_mask_vect = np.vectorize(get_MLD_from_mask, signature='(k)->()')
    pred_S_mean, pred_S_std, pred_T_mean, pred_T_std, mld, rmse_T, rmse_S = get_mean_std_pred(ensemble, X_scaled, x, Sm,
                                                                                              Sstd, Tm, Tstd)

    x = x.assign(variables={"PSAL_predicted": (('N_PROF', 'PRES_INTERPOLATED'), pred_S_mean.data)})
    x = x.assign(variables={"TEMP_predicted": (('N_PROF', 'PRES_INTERPOLATED'), pred_T_mean.data)})

    x = x.assign(variables={"MLD_mask": (('N_PROF', 'PRES_INTERPOLATED'), mld.data)})
    x = x.assign(variables={"MLD_pred": (('N_PROF'), get_MLD_from_mask_vect(x.MLD_mask))})

    x = x.assign(variables={"PSAL_predicted_std": (('N_PROF', 'PRES_INTERPOLATED'), pred_S_std.data)})
    x = x.assign(variables={"TEMP_predicted_std": (('N_PROF', 'PRES_INTERPOLATED'), pred_T_std.data)})

    x = add_sig(x)

    ae_S = np.fabs(x['PSAL'] - x['PSAL_predicted'])
    ae_T = np.fabs(x['TEMP'] - x['TEMP_predicted'])
    x = x.assign(variables={"ae_S": (('N_PROF', 'PRES_INTERPOLATED'), ae_S.data)})
    x = x.assign(variables={"ae_T": (('N_PROF', 'PRES_INTERPOLATED'), ae_T.data)})
    x = x.assign(variables={"rmse_T_model": (('Model', 'PRES_INTERPOLATED'), np.array(rmse_T))})
    x = x.assign(variables={"rmse_S_model": (('Model', 'PRES_INTERPOLATED'), np.array(rmse_S))})
   
    return x


def main():
    args = get_args()
    input_dataset = args.input_dataset
    nb_models = args.nb_models
    path_models = args.path_models
    path_out = args.path_out
    use_all_data = args.use_all_data == 1

    logging.basicConfig(
        format='[%(levelname)s] %(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f"training_log.log"),
            logging.StreamHandler()
        ]
    )

    arguments_str = f"nb_models: {nb_models} " \
                    f"path_models: {path_models} " \
                    f"path_out: {path_out} " \
                    f"use_all_data: {use_all_data} "

    logging.info(f"Training script launched with the following arguments:\n {arguments_str}")

    Path(path_out).mkdir(parents=True, exist_ok=True)
    Path(path_models).mkdir(parents=True, exist_ok=True)

    ds_train, ds_test = load_dataset(input_dataset, use_all_data=use_all_data)
    logging.info(f"dataset loaded")

    logging.info(f"preprocessing starting")
    # scaler, X_scaled, y_scaled, X_test_scaled, ds_test
    # Creation of pw (Deltaz)
    p = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 40, 45, 50, 55,
         60, 65, 70, 75, 80, 90, 100, 110, 120, 133, 147, 163, 180, 199, 221, 245, 271,
         301, 334, 371, 412, 458, 509, 565, 628, 697, 773, 857, 950, 1000]
    pw = np.ones(len(p))
    for n in np.arange(0, len(p) - 1):
        pw[n + 1] = p[n + 1] - p[n]

    # ------------- Train model ----------- #
    X_train, scaler_input = format_input(ds_train)
    y_train, Sm, Sstd, Tm, Tstd, SIGm, SIGstd = format_target(ds_train)
    logging.info(f"preprocessing finished")

    logging.info(f"training starting")
    ensemble = train_ensemble(X_train, y_train, nb_models)
    logging.info(f"training finished")

    logging.info(f"saving models starting")
    save_models(ensemble, Sm, Sstd, Tm, Tstd, SIGm, SIGstd, pw, scaler_input, path_models)
    logging.info(f"saving models finished and saved in {path_models}")

    logging.info(f"Predictions starting")
    if use_all_data == False:
        # ----------- Evaluate model on test andSve both datasets ------------ #
        X_test, _ = format_input(ds_test, scaler_input)
        ds_train = predict_dataset(ensemble, X_train, ds_train, Sm, Sstd, Tm, Tstd)

        ds_train.to_netcdf(f"{path_out}/train_ds.nc")

        ds_test = predict_dataset(ensemble, X_test, ds_test, Sm, Sstd, Tm, Tstd)

        ds_test.to_netcdf(f"{path_out}/test_ds.nc")


    else:
        ds_train = predict_dataset(ensemble, X_train, ds_train, Sm, Sstd, Tm, Tstd)
        ds_train.to_netcdf(f"{path_out}/full_ds.nc")
       
    logging.info(f"Predictions finished and saved in {path_out}")


if __name__ == '__main__':
    main()
