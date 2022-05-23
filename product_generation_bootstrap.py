import glob
import time
from pathlib import Path
import gsw
import numpy as np
import pandas as pd
import xarray as xr
from joblib import load
import sklearn
from tensorflow import keras


def get_args():
    """
    Extract arguments from command line

    Returns
    -------
    parse.parse_args(): dict of the arguments

    """
    import argparse

    parse = argparse.ArgumentParser(description="Ocean patterns method")
    parse.add_argument('path_ds', type=str, help='path to ds')
    parse.add_argument('path_models', type=str, help='path to model folder containing all scalers, NN and pca')
    parse.add_argument('path_out', type=str, help='path to output directory')
    return parse.parse_args()


def get_mean_std_pred(ensemble, X_scaled, scal_Sm, scal_Sstd, scal_Tm, scal_Tstd):
    """
    Predict using the trained ensemble

    @param ensemble: list of keras models
    @type ensemble: list of Keras.Model
    @param X_scaled: input array preprocessed by the format_input function
    @type X_scaled: numpy ndarray
    @param scal_Sm: mean of PSAL variable per depth
    @type scal_Sm: numpy nd array float64
    @param scal_Sstd: std of PSAL variable per depth
    @type scal_Sstd: numpy nd array float64
    @param scal_Tm: mean of TEMP variable per depth
    @type scal_Tm: numpy nd array float64
    @param scal_Tstd: std of TEMP variable per depth
    @type scal_Tstd: numpy nd array float64
    @return: mean predictions for TEMP and PSAL, standard deviation of predictions TEMP and PSAL, mean prediction of MLD mask
    @rtype: numpy nd_array, numpy nd_array, numpy nd_array, numpy nd_array, numpy nd_array
    """
    predS = []
    predT = []
    predMLD = []
    for model in ensemble:
        tmp_pred = model.predict(X_scaled)
        predS.append(tmp_pred[:, :, 0] * scal_Sstd + scal_Sm)
        predT.append(tmp_pred[:, :, 1] * scal_Tstd + scal_Tm)
        predMLD.append(tmp_pred[:, :, 2])
    return np.mean(predS, axis=0), np.std(predS, axis=0), np.mean(predT, axis=0), np.std(predT, axis=0), np.mean(
        predMLD, axis=0)


def add_sig(ds):
    """
    Compute sigma0 and add it to the dataset using gsw package
    @param ds: dataset containing the predictions
    @type ds: xarray dataset
    @return: dataset with the added variable SIG_predicted (sigma0 from the prediction of TS)
    @rtype: xarray dataset
    """
    sa_pred = gsw.SA_from_SP(ds['PSAL_predicted'], ds['DEPTH'], ds['lon'], ds['lat'])
    ct_pred = gsw.CT_from_t(sa_pred, ds['TEMP_predicted'], ds['DEPTH'])
    sig_pred = gsw.sigma0(sa_pred, ct_pred)
    ds = ds.assign(variables={"SIG_predicted": (('N_PROF', 'DEPTH'), sig_pred.data)})
    return ds


def prepare_data(path):
    """
    Load input data and change types for lighter output product
    @param path: path of input dataset
    @type path: str
    @return: dataset
    @rtype: xarray dataset
    """
    print(f'Open dataset {path}')
    x = xr.open_dataset(f"{path}")
    # optimize types
    x['mask'] = x['mask'].astype(np.bool_)
    x['BATHY'] = x['BATHY'].astype(np.float32)
    x['MDT'] = x['MDT'].astype(np.float32)
    x['SLA'] = x['SLA'].astype(np.float32)
    x['UGOS'] = x['UGOS'].astype(np.float32)
    x['UGOSA'] = x['UGOSA'].astype(np.float32)
    x['VGOS'] = x['VGOS'].astype(np.float32)
    x['VGOSA'] = x['VGOSA'].astype(np.float32)
    x['SLA_err'] = x['SLA_err'].astype(np.float32)
    # compute week of year
    day = np.array(pd.DatetimeIndex(x['time'].data).dayofyear).astype(np.int32)
    x = x.assign(variables={"dayOfYear": (('time'), day)})
    return x


def predict_month(x, ensemble, scal_Sm, scal_Sstd, scal_Tm, scal_Tstd, scaler_input, path_out, yy, mm):
    """
    Core function to predict a month:
    - format input from xarray dataset to numpy array usable by the trained NN
    - predict temperature and salinity profiles using the neural network ensemble
    - save the results in netcdf format

    @param x: input dataset
    @type x: xarray dataset
    @param ensemble:
    @type ensemble:
    @param scal_Sm: mean of PSAL variable per depth
    @type scal_Sm: numpy nd array float64
    @param scal_Sstd: std of PSAL variable per depth
    @type scal_Sstd: numpy nd array float64
    @param scal_Tm: mean of TEMP variable per depth
    @type scal_Tm: numpy nd array float64
    @param scal_Tstd: std of TEMP variable per depth
    @type scal_Tstd: numpy nd array float64
    @param scaler_input: standard scaler if already fitted on data
    @type scaler_input: Scikit learn standard scaler
    @param path_out: path where to save the datasets containing the results
    @type path_out: str
    @param yy: year to predict
    @type yy: int or str
    @param mm: month to predict (needs to include the 0 to start for month under 10, ex: '02'
    @type mm: str
    """
    stacked = x.sel(time=slice(f"{yy}-{mm}", f"{yy}-{mm}")).stack(N_PROF=('time', 'lat', 'lon'))
    stacked = stacked.dropna(dim='N_PROF', how='any')

    # ----------- create X vector --------------- #
    d = 1 / 365
    cos_day = np.cos(np.pi * 2 * d * stacked['dayOfYear'].data)
    sin_day = np.sin(np.pi * 2 * d * stacked['dayOfYear'].data)
    X = np.zeros([len(stacked['N_PROF']), 12])
    X[:, 0] = stacked['SLA'].data
    X[:, 1] = stacked['lat'].data
    X[:, 2] = stacked['lon'].data
    X[:, 3] = cos_day
    X[:, 4] = sin_day
    X[:, 5] = stacked['MDT'].data
    X[:, 6] = stacked['UGOSA'].data
    X[:, 7] = stacked['VGOSA'].data
    X[:, 8] = stacked['UGOS'].data
    X[:, 9] = stacked['VGOS'].data
    X[:, 10] = stacked['SST'].data
    X[:, 11] = -stacked['BATHY'].data

    X_scaled = scaler_input.transform(X)

    # ------------- Predict and add to dataset -------------- #
    pred_S_mean, pred_S_std, pred_T_mean, pred_T_std, mld_mask = get_mean_std_pred(ensemble, X_scaled, scal_Sm,
                                                                                   scal_Sstd, scal_Tm, scal_Tstd)

    stacked = stacked.assign(variables={"PSAL_predicted": (('N_PROF', 'DEPTH'), pred_S_mean.data)})
    stacked = stacked.assign(variables={"TEMP_predicted": (('N_PROF', 'DEPTH'), pred_T_mean.data)})
    stacked = stacked.assign(variables={"PSAL_predicted_std": (('N_PROF', 'DEPTH'), pred_S_std.data)})
    stacked = stacked.assign(variables={"TEMP_predicted_std": (('N_PROF', 'DEPTH'), pred_T_std.data)})
    stacked = stacked.assign(variables={"MLD_mask": (('N_PROF', 'DEPTH'), mld_mask.data)})

    stacked = add_sig(stacked)
    stacked = stacked.unstack('N_PROF')
    stacked = stacked.sortby('lon')
    print(f"size output file: {stacked.nbytes / 1073741824} go, saved in {path_out}/produit_{yy}{mm}.nc")
    stacked.to_netcdf(f"{path_out}/produit_{yy}{mm}.nc")


def main():
    """
    - loads all models and scalers
    - Loops on every month of inputs to call the predict_month function and saves the results
    """
    args = get_args()
    path_ds = args.path_ds
    path_models = args.path_models
    path_out = args.path_out
    Path(path_out).mkdir(parents=True, exist_ok=True)
    x = prepare_data(path_ds)
    depth_levels = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 40,
                    45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110, 120, 133, 147, 163,
                    180, 199, 221, 245, 271, 301, 334, 371, 412, 458, 509, 565, 628,
                    697, 773, 857, 950, 1000]
    x = x.assign_coords(DEPTH=depth_levels)

    scaler_input = load(f"{path_models}/scaler_input.joblib")
    scal_Sm = load(f'{path_models}/Sm.joblib')
    scal_Sstd = load(f'{path_models}/Sstd.joblib')
    scal_Tm = load(f'{path_models}/Tm.joblib')
    scal_Tstd = load(f'{path_models}/Tstd.joblib')

    models_list = glob.glob(f'{path_models}/neuralnet/ensemble/*')
    ensemble = []
    for model_path in models_list:
        ensemble.append(keras.models.load_model(model_path, compile=False))

    print(f'all models from {path_models} loaded')
    print('Computation starting')

    year_start = 1993
    year_end = 1994
    for yy in range(year_start, year_end + 1):
        time_year = time.time()
        for mm in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
            time_start = time.time()
            print(f'Starting prediction for month: {mm}-{yy}')
            
            predict_month(x=x, ensemble=ensemble, scal_Sm=scal_Sm, scal_Sstd=scal_Sstd, scal_Tm=scal_Tm,
                          scal_Tstd=scal_Tstd, scaler_input=scaler_input, path_out=path_out, yy=yy, mm=mm)

            print(f"Prediction of month {yy}-{mm} finished in {time.time() - time_start}")
        print(f'Year: {yy} done in {time.time() - time_year}')
    print('Computation finished')


if __name__ == '__main__':
    main()
