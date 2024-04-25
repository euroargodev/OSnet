#Adjust product all depth - April 2022
#cd /home1/datahome/epauthen/All_depth
#qsub Adjust_product.pbs
#qstat -u epauthen
import xarray as xr
import gsw
import numpy as np
import time
from numba import float64, guvectorize
import pandas as pd
from pathlib import Path

ds_input = xr.open_dataset("/home/datawork-lops-bluecloud/osnet/data_remote_sensing/Gridded_input500_9319_maskbathy.nc")

path_models = "/home/datawork-lops-bluecloud/osnet/models/all_depth500_6L"
path_product = "/home/datawork-lops-bluecloud/osnet/product_out/all_depth500_6L"

def get_data(path, yy, mm):
    ds = xr.open_dataset(f"{path}/produit_{yy}{mm}.nc")
    print(f'Dataset opened')

    #Compute new mask K*
    b = 2
    b2 = 1
    H = 0.477     #For 500m drop and remove profiles shorter than MLD, 6 Layer 512
    mask2 = np.where(ds['K_pred'].data<H, ds['K_pred'], 1)
    ds = ds.assign(variables={"MLD_mask2": (('depth', 'time', 'latitude', 'longitude'), mask2)})
    mask3 = np.where((ds['K_pred']>H) & (ds['K_pred']<b2), b-ds['K_pred'].data, ds['MLD_mask2'].data)
    ds = ds.assign(variables={"MLD_mask3": (('depth', 'time', 'latitude', 'longitude'), mask3)})
    print(f'Computed new mask MLD_mask3')
    return ds

@guvectorize(
    "(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])",
    "(n), (n), (n), (n) -> (n), (n)"
)
def MLD_adjustment_1d(temp_in, psal_in, depth, mask, temp, psal):
    temp[:] = np.copy(temp_in)
    psal[:] = np.copy(psal_in)
    if np.isnan(temp_in).any()==False:
        bottom = 114
    else:
        bottom = len(temp_in[np.isnan(temp_in)==False])-1
    for d in range(bottom-2, -1, -1):
        # apply mask on TEMP and PSAL
        temp[d] = (temp_in[d]*mask[d] - temp_in[d+1]*mask[d]) + temp[d+1]
        psal[d] = (psal_in[d]*mask[d] - psal_in[d+1]*mask[d]) + psal[d+1]

def MLD_adjustment(ds,mask):
    temp_out, psal_out = xr.apply_ufunc(MLD_adjustment_1d,
                                    ds['temp_pred'], ds['psal_pred'], ds['depth'], mask,
                                    input_core_dims=(['depth'],['depth'],['depth'],['depth']),
                                    output_core_dims=(['depth'],['depth']),
                                    output_dtypes=[np.float64, np.float64])
    # get sig adjusted
    sa_out = gsw.SA_from_SP(psal_out, ds['depth'], ds['longitude'], ds['latitude'])
    ct_out = gsw.CT_from_t(sa_out,temp_out,ds['depth'])
    sig_out = gsw.sigma0(sa_out, ct_out)
    print(f'Adjustment applied and SIG_adj created')
    
    ds_out = ds.assign(variables={"temp_adj": (('time', 'latitude', 'longitude', 'depth'), temp_out.data),
                                  "psal_adj": (('time', 'latitude', 'longitude', 'depth'), psal_out.data),
                                  "sig_adj" : (('time', 'latitude', 'longitude', 'depth'), sig_out.data)})
    return ds_out 

year_start = 2012
year_end = 2012
for yy in range(year_start, year_end+1):
    time_year = time.time()
    for mm in "05": #["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]:
        time_start = time.time()
        print(f'Starting adjustment for month: {mm}-{yy}')

        ds = get_data(path_product, yy, mm)
        ds = MLD_adjustment(ds,mask = ds['MLD_mask3'])

        print(f"size output file: {np.around(ds.nbytes / 1073741824,2)} Go, saved in {path_product}/produit_{yy}{mm}_adj.nc")
        ds.to_netcdf(f"{path_product}/produit_{yy}{mm}_adj.nc")
        print(f"adjustment of month {yy}-{mm} finished in {np.around(time.time() - time_start,2)} secondes")

    print(f'Year: {yy} done in {time.time() - time_year}')

print('Computation finished')