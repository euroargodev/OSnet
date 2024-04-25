#cd /home1/datahome/epauthen/All_depth
#qsub product_generation_all_depth.pbs
#qstat -u epauthen
import sklearn
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as MSE
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from joblib import load
from keras import backend as K
import xarray as xr
import numpy as np
import pandas as pd
import time
from pathlib import Path
import gsw
import glob

def get_args():
    """
    Extract arguments from command line

    Returns
    -------
    parse.parse_args(): dict of the arguments

    """
    import argparse

    parse = argparse.ArgumentParser(description="OSnet")
    parse.add_argument('path_ds', type=str, help='path to ds')
    parse.add_argument('path_models', type=str, help='path to folder containing all scalers and model')
    parse.add_argument('path_out', type=str, help='path to output directory')
    
    return parse.parse_args()


def get_mean_std_pred(ensemble, X, scal_Sm, scal_Sstd, scal_Tm, scal_Tstd):
    predS = []
    predT = []
    predK = []
    for model in ensemble:
        tmp_pred = model.predict(X)
        temp = tmp_pred[:, :, 0] * scal_Tstd + scal_Tm*X[1]
        psal = tmp_pred[:, :, 1] * scal_Sstd + scal_Sm*X[2]
        predT.append(temp)
        predS.append(psal)
        predK.append(tmp_pred[:, :, 2])
    return np.mean(predT, axis=0), np.std(predT, axis=0), np.mean(predS, axis=0), np.std(predS, axis=0), np.mean(predK,axis=0)

def add_sig(ds):
    sa_pred = gsw.SA_from_SP(ds['PSAL_predicted'], ds['depth'], ds['lon'], ds['lat'])
    ct_pred = gsw.CT_from_t(sa_pred,ds['TEMP_predicted'],ds['depth'])
    sig_pred = gsw.sigma0(sa_pred, ct_pred)
    ds = ds.assign(variables={"SIG_predicted": (('N_PROF', 'depth'), sig_pred.data)})
    return ds


def prepare_data(path):
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
    # compute day of year
    day = np.array(pd.DatetimeIndex(x['time'].data).dayofyear).astype(np.int32)
    x = x.assign(variables={"dayOfYear": (('time'), day)})
    return x


def predict_month(x, ensemble, scal_Sm, scal_Sstd, scal_Tm, scal_Tstd, scaler_input, path_out, yy, mm):
    x_month = x.sel(time=slice(f"{yy}-{mm}", f"{yy}-{mm}"))
    x_month = x_month.sel(day = slice(1,len(x_month.time)))
    x_month = x_month.assign(variables={"BATHY_mask": (('depth','time','lat','lon'), x_month.BATHY_mask.data)})
    stacked = x_month.stack(N_PROF=('time', 'lat', 'lon'))
    stacked = stacked.dropna(dim='N_PROF', how='any')

    # ----------- create X vector --------------- #
    d = 1/365
    cos_week = np.cos(np.pi * 2 * d * stacked['dayOfYear'].data)
    sin_week = np.sin(np.pi * 2 * d * stacked['dayOfYear'].data)
    X = np.zeros([len(stacked['N_PROF']), 12])
    X[:,0] = stacked['SLA'].data
    X[:,1] = stacked['lat'].data
    X[:,2] = stacked['lon'].data
    X[:,3] = cos_week
    X[:,4] = sin_week
    X[:,5] = stacked['MDT'].data
    X[:,6] = stacked['UGOSA'].data
    X[:,7] = stacked['VGOSA'].data
    X[:,8] = stacked['UGOS'].data
    X[:,9] = stacked['VGOS'].data
    X[:,10] = stacked['SST'].data
    X[:,11] = -stacked['BATHY'].data

    X_scaled = scaler_input.transform(X)

    # ------------- Predict and add to dataset -------------- #
    pred_T_mean, pred_T_std, pred_S_mean, pred_S_std, pred_K_mean = get_mean_std_pred(ensemble, [X_scaled, stacked.BATHY_mask.T.data, stacked.BATHY_mask.T.data],scal_Sm, scal_Sstd, scal_Tm, scal_Tstd)
    
    Smean_pred = np.where(pred_S_mean==0, np.nan, pred_S_mean)
    Sstd_pred = np.where(pred_S_mean==0, np.nan, pred_S_std)
    Tmean_pred = np.where(pred_S_mean==0, np.nan, pred_T_mean)
    Tstd_pred = np.where(pred_S_mean==0, np.nan, pred_T_std)
    Kmean_pred = np.where(pred_S_mean==0, np.nan, pred_K_mean)

    stacked = stacked.assign(variables={"PSAL_predicted": (('N_PROF', 'depth'), Smean_pred.data)})
    stacked = stacked.assign(variables={"TEMP_predicted": (('N_PROF', 'depth'), Tmean_pred.data)})
    stacked = stacked.assign(variables={"PSAL_predicted_std": (('N_PROF', 'depth'), Sstd_pred.data)})
    stacked = stacked.assign(variables={"TEMP_predicted_std": (('N_PROF', 'depth'), Tstd_pred.data)})
    stacked = stacked.assign(variables={"K_predicted": (('N_PROF', 'depth'), Kmean_pred.data)})

    stacked = add_sig(stacked)
    stacked = stacked.unstack('N_PROF')
    stacked = stacked.sortby('lon')
    
    #drop all other variables
    stacked = stacked.rename({"lat":"latitude","lon":"longitude","TEMP_predicted":"temp_pred","PSAL_predicted":"psal_pred",
                      "TEMP_predicted_std":"temp_std","PSAL_predicted_std":"psal_std","K_predicted":'K_pred','SIG_predicted':'SIG_pred'})
    stacked = stacked[['temp_pred', 'psal_pred','temp_std','psal_std','K_pred','SIG_pred']]
    stacked = stacked.drop('mask')
    stacked['temp_pred'] = stacked.temp_pred.astype(np.float32)
    stacked['psal_pred'] = stacked.psal_pred.astype(np.float32)
    stacked['temp_std'] = stacked.temp_std.astype(np.float32)
    stacked['psal_std'] = stacked.psal_std.astype(np.float32)
    stacked['K_pred'] = stacked.K_pred.astype(np.float32)
    stacked['SIG_pred'] = stacked.SIG_pred.astype(np.float32)

    print(f"size output file: {stacked.nbytes / 1073741824} go, saved in {path_out}/produit_{yy}{mm}.nc")
    stacked.to_netcdf(f"{path_out}/produit_{yy}{mm}.nc")


def main():
    args = get_args()
    path_ds = args.path_ds
    path_models = args.path_models
    path_out = args.path_out
    Path(path_out).mkdir(parents=True, exist_ok=True)
    x = prepare_data(path_ds)
    
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
    year_start = 2018
    year_end = 2018
    for yy in range(year_start, year_end+1):
        time_year = time.time()
        for mm in ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
            time_start = time.time()
            print(f'Starting prediction for month: {mm}-{yy}')
            predict_month(x=x, ensemble=ensemble, scal_Sm=scal_Sm, scal_Sstd=scal_Sstd, scal_Tm=scal_Tm, scal_Tstd=scal_Tstd, scaler_input=scaler_input, path_out=path_out, yy=yy, mm=mm)
            print(f"Prediction of month {yy}-{mm} finished in {time.time() - time_start}")
        print(f'Year: {yy} done in {time.time() - time_year}')
    print('Computation finished')


if __name__ == '__main__':
    main()
