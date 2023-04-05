'''
Python script to prepare forcing data sets for the ROMS model.

The input data should contain nan values for land mask.

'''



from pandas.tseries.offsets import DateOffset

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import xesmf as xe
import numpy as np
import datetime
import time

# Choose if test data with only 3 time steps (True) or normal data (False) should be used 
test = False

# Select data type 'GLORYS', 'ORA20C', 'SODA', 'ECCO'
data_type = 'ECCO'

# Path to main directory
path = '/projects/0/einf2878/ROMS'

# Import grid
grid = xr.open_dataset(f'{path}/data/grid/NorthSea4_smooth01_sponge_nudg.nc')

# Import data
if test == True:
    data = xr.open_dataset(f'{path}/data/{data_type}/forcing_input/forcing_input_test.nc')
else:
    data = xr.open_dataset(f'{path}/data/{data_type}/forcing_input/forcing_input.nc')

if test == True:
    output_dir = f'{path}/data/{data_type}/forcing_output/test'
else:
    output_dir = f'{path}/data/{data_type}/forcing_output'



start_year = data.time[0].dt.year.values
start_month = data.time[0].dt.month.values
start_day = data.time[0].dt.day.values
final_year = data.time[-1].dt.year.values
final_month = data.time[-1].dt.month.values
final_day = data.time[-1].dt.day.values




###############        PERFORM HORIZONTAL INTERPOLATION        ###############



# Create output grids
output_grid_rho = xr.Dataset(data_vars=dict(mask=(["eta_rho", "xi_rho"], grid.mask_rho.values),), 
                           coords=dict(eta_rho=(["eta_rho"], grid.eta_rho.values),
                                       xi_rho=(["xi_rho"], grid.xi_rho.values),
                                       lat=(["eta_rho", "xi_rho"], grid.lat_rho.values),
                                       lon=(["eta_rho", "xi_rho"], grid.lon_rho.values),
                                      ))

output_grid_u = xr.Dataset(data_vars=dict(mask=(["eta_u", "xi_u"], grid.mask_u.values),), 
                           coords=dict(eta_u=(["eta_u"], grid.eta_u.values),
                                       xi_u=(["xi_u"], grid.xi_u.values),
                                       lat=(["eta_u", "xi_u"], grid.lat_u.values),
                                       lon=(["eta_u", "xi_u"], grid.lon_u.values),
                                      ))


output_grid_v = xr.Dataset(data_vars=dict(mask=(["eta_v", "xi_v"], grid.mask_v.values),), 
                           coords=dict(eta_u=(["eta_v"], grid.eta_v.values),
                                       xi_u=(["xi_v"], grid.xi_v.values),
                                       lat=(["eta_v", "xi_v"], grid.lat_v.values),
                                       lon=(["eta_v", "xi_v"], grid.lon_v.values),
                                      ))





    
def horizontal_interp1D(input_data, var):
    ''' 
    Function to perform the horizontal regridding per depth level
    
    '''
    
    
    if var == 'thetao':
        
        output_grid = output_grid_rho
    
    elif var == 'uo':
        
        output_grid = output_grid_u
        
    else:
        
        output_grid = output_grid_v
    
  
    
    # Prepare the input grid
    input_grid = xr.Dataset(data_vars=dict(mask=(["lat", "lon"], np.nan_to_num(input_data[0, :, :] / input_data[0, :, :])),), 
                           coords=dict(time=(["time"], data.time.values),
                                       lat=(["lat"], data.latitude.values),
                                       lon=(["lon"], data.longitude.values),
                                      ))
    
    
    # Make arrays C_CONTIGUOUS
    input_grid = input_grid.astype(dtype = 'int64', order = 'C')
    output_grid = output_grid.astype(dtype = 'float64', order = 'C')
    
    
        
    # Build regridder
    regridder = xe.Regridder(input_grid, output_grid, "bilinear", extrap_method= 'nearest_s2d')


    # Apply to data
    output_data = regridder(input_data)
    
    
    return output_data
    
    
    
def horizontal_interp(input_data, grid):
    ''' Function to perform the horizontal interpolation from latitude, longitude coordinates to (eta_rho, xi_rho), (eta_u, xi_u) or (eta_v, xi_v). 
        The horizontal interpolation is performed as a linear interpolation.
        
        The glorys data with a resolution of 1/12° or the ora data with a resolution of 1.0° is regridded on the ROMS grid that has a resolution of 1/4°.
        
        '''
    start_time = time.time()
    
    
    # Some data preparations
    
    # Rename coordinates to 'lat' and 'lon'
    input_data = input_data.rename({'latitude':'lat', 'longitude':'lon', 'time':'time'})

    # Sort coordinates in increasing order
    input_data = input_data.sortby(['lat', 'lon'])

    # Transpose dimensions
    input_data = input_data.transpose('depth', 'time', 'lat', 'lon')
    
    

    
    # Regrid Sea level to 'rho'-coordinates
    print(f'===================== Start horizontal interpolation of Sea Level (Duration: {round(time.time() - start_time,2)} seconds) ========================================================')
    
    
    data_rho = input_data.drop(['uo', 'vo', 'thetao', 'so'])
    
    
    # Prepare the output grid
    output_grid = xr.Dataset(data_vars=dict(mask=(["eta_rho", "xi_rho"], grid.mask_rho.values),), 
                           coords=dict(eta_rho=(["eta_rho"], grid.eta_rho.values),
                                       xi_rho=(["xi_rho"], grid.xi_rho.values),
                                       lat=(["eta_rho", "xi_rho"], grid.lat_rho.values),
                                       lon=(["eta_rho", "xi_rho"], grid.lon_rho.values),
                                      ))
    
  
    # Prepare the input grid
    input_grid = xr.Dataset(data_vars=dict(mask=(["lat", "lon"], xr.where(~np.isnan(data_rho.zos.isel(time=0)), 1, 0).values),), 
                           coords=dict(time=(["time"], data_rho.time.values),
                                       lat=(["lat"], data_rho.lat.values),
                                       lon=(["lon"], data_rho.lon.values),
                                      ))


    # Make arrays C_CONTIGUOUS
    input_grid = input_grid.astype(dtype = 'int64', order = 'C')
    output_grid = output_grid.astype(dtype = 'float64', order = 'C')
    
        
    # Build regridder
    regridder = xe.Regridder(input_grid, output_grid, "bilinear", extrap_method= 'nearest_s2d')


    # Apply to data
    output_data_rho_zos = regridder(data_rho)
    





    # Regrid Salinity and Temperature to 'rho'-coordinates
    print(f'===================== Start horizontal interpolation of Salinity and Temperature (Duration: {round(time.time() - start_time,2)} seconds) =========================================')
    
    data_rho_ST = input_data.drop(['uo', 'vo', 'zos'])
    
    
    # Perform regridding parallel
    output_data_rho_ST = xr.apply_ufunc(horizontal_interp1D,
                                       data_rho_ST, 'thetao',
                                       input_core_dims=[['time', 'lat', 'lon'], []],
                                       output_core_dims = [['time', 'eta_rho', 'xi_rho']],
                                       dask = 'parallelized',
                                       output_dtypes = [data_rho_ST.thetao.dtype],
                                       vectorize = True)
    
    output_data_rho_ST = output_data_rho_ST.assign_coords(eta_rho = grid.eta_rho, xi_rho = grid.xi_rho)
    
    
    # Regrid Zonal velocities to 'u'-coordinates
    print(f'===================== Start horizontal interpolation of Zonal Velocity (Duration: {round(time.time() - start_time,2)} seconds) ====================================================')
    
    data_u = input_data.drop(['so', 'thetao', 'vo', 'zos'])
    
    
    # Perform regridding parallel
    output_data_u = xr.apply_ufunc(horizontal_interp1D,
                                       data_u, 'uo',
                                       input_core_dims=[['time', 'lat', 'lon'], []],
                                       output_core_dims = [['time', 'eta_u', 'xi_u']],
                                       dask = 'parallelized',
                                       output_dtypes = [data_u.uo.dtype],
                                       vectorize = True)
    
    
    output_data_u = output_data_u.assign_coords(eta_u = grid.eta_u, xi_v = grid.xi_u)
    
    
    
    # Regrid Meridional velocities to 'v'-coordinates
    print(f'===================== Start horizontal interpolation of Meridional Velocity (Duration: {round((time.time() - start_time),2)} seconds) =============================================')
    
    
    data_v = input_data.drop(['so', 'thetao', 'uo', 'zos'])
    
    
    
    # Perform regridding parallel
    output_data_v = xr.apply_ufunc(horizontal_interp1D,
                                       data_v, 'vo',
                                       input_core_dims=[['time', 'lat', 'lon'], []],
                                       output_core_dims = [['time', 'eta_v', 'xi_v']],
                                       dask = 'parallelized',
                                       output_dtypes = [data_v.vo.dtype],
                                       vectorize = True)
    
    
    output_data_v = output_data_v.assign_coords(eta_v = grid.eta_v, xi_v = grid.xi_v)
    
    
    # Create one final dataset
    print('===================== Finalised horizontal interpolation, creating final dataset ==============================================')
    
    
    # Create one horizontally regridded file
    output_data = output_data_rho_ST.copy()

    # Add zos, uo and vo
    output_data = output_data.assign(zos=output_data_rho_zos.zos)
    output_data = output_data.assign(uo=output_data_u.uo)
    output_data = output_data.assign(vo=output_data_v.vo)

    output_data.to_netcdf(f'{output_dir}/data_hr.nc')
    
    return output_data


data_hr = horizontal_interp(data, grid)














###############        PERFORM VERTICAL INTERPOLATION        ###############




def create_grdMODEL(data):
    ''' Function to create the dataset grdMODEL '''
    
    grdMODEL = data.copy().drop(['thetao', 'so', 'uo', 'vo'])
    
    grdMODEL['lon'] = data.lon
    grdMODEL['lat'] = data.lat
    grdMODEL['h'] = data.depth
    grdMODEL['nlevels'] = grdMODEL.h.size
    
    
    grdMODEL['fillval'] = -32767   # Change for ORA-20C
    grdMODEL['hc'] = None

    # Create grid for ESMF interpolation, probably not needed for VT

    grdMODEL['z_r'] = -grdMODEL.h

    grdMODEL['grdType'] = 'regular'
    grdMODEL['lonName'] = 'longitude'
    grdMODEL['latName'] = 'latitude'
    grdMODEL['depthName'] = 'depth'


    grdMODEL['Lp'] = len(grdMODEL.lat[1,:])
    grdMODEL['Mp'] = len(grdMODEL.lat[:,1])

    grdMODEL['L'] = grdMODEL.Lp - 1
    grdMODEL['M'] = grdMODEL.Mp - 1
    
    
    
    return grdMODEL

def create_grdROMS(grid):
    ''' Function to create the dataset grdROMS '''
    
    # Create the dataset grdROMS

    # Copy the roms grid
    grdROMS = grid.copy()

    # Drop unnecessary variables
    grdROMS = grdROMS.drop(['tracer_NudgeCoef', 'diff_factor', 'visc_factor', 'hraw', 'f', 'spherical'])



    # Add below variables to grdROMS
    grdROMS['write_clim'] = True
    grdROMS['write_bry'] = True
    grdROMS['write_init'] = True
    grdROMS['write_stations'] = False
    grdROMS['lonname'] = 'lon_rho'
    grdROMS['latname'] = 'lat_rho'
    grdROMS['inittime'] = 0                    # Set initTime to 1 if you dont want the first time-step to be the initial field (no ubar and vbar if time=0)
    grdROMS['ocean_time'] = 0
    grdROMS['NT'] = 2
    grdROMS['tracer'] = grdROMS.NT
    grdROMS['time'] = 0                      
    grdROMS['reftime'] = 0
    grdROMS['grdtype'] = 'regular'

    grdROMS['masked_h'] = grdROMS.h.where(grdROMS.h > 0, grdROMS.h, grdROMS.h.max())
    grdROMS['hmin'] = grdROMS.masked_h.min()

    grdROMS['vtransform'] = 2
    grdROMS['vstretching'] = 4

    grdROMS['nlevels'] = grdROMS.s_rho.size

    grdROMS['zeta'] = (('eta_rho', 'xi_rho'), np.zeros(grdROMS.h.shape))

    grdROMS['invpm'] = 1.0 / grdROMS.pm
    grdROMS['invpn'] = 1.0 / grdROMS.pn

    grdROMS['Lp'] = grdROMS.lat_rho[1,:].size     
    grdROMS['Mp'] = grdROMS.lat_rho[:,1].size     

    grdROMS['fillval'] = -9.99e33

    grdROMS['eta_rho_'] = grdROMS.Mp
    grdROMS['eta_u_'] = grdROMS.Mp
    grdROMS['eta_v_'] = grdROMS.Mp - 1
    grdROMS['eta_psi_'] = grdROMS.Mp - 1


    grdROMS['xi_rho_'] = grdROMS.Lp
    grdROMS['xi_u_'] = grdROMS.Lp - 1
    grdROMS['xi_v_'] = grdROMS.Lp
    grdROMS['xi_psi_'] = grdROMS.Lp - 1



    # Obtain s_rho

    c1 = 1.0
    c2 = 2.0
    p5 = 0.5

    lev = np.arange(1, int(grdROMS.nlevels) + 1, 1)
    ds = 1.0 / int(grdROMS.nlevels)


    grdROMS['s_rho_'] = - c1 + (lev - p5) * ds


    # Obtain s_w

    lev = np.arange(0, int(grdROMS.nlevels), 1)
    ds = 1.0 / (int(grdROMS.nlevels) - 1)


    grdROMS['s_w_'] = - c1 + (lev - p5) * ds




    # Obtain Cs_r

    if (grdROMS.theta_s > 0):
        Csur = (c1 - np.cosh(grdROMS.theta_s * grdROMS.s_rho)) / (np.cosh(grdROMS.theta_s) - c1)

    else:
        Csur = -grdROMS.s_rho**2

    if (grdROMS.theta_b > 0):
        Cbot = (np.exp(grdROMS.theta_b * Csur) - c1 ) / (c1 - np.exp(-grdROMS.theta_b))
        grdROMS['Cs_r'] = Cbot
    else:
        grdROMS['Cs_r'] = Csur     



    # Obtain Cs_w

    if (grdROMS.theta_s > 0):
        Csur = (c1 - np.cosh(grdROMS.theta_s * grdROMS.s_w)) / (np.cosh(grdROMS.theta_s) - c1)

    else:
        Csur = -grdROMS.s_w**2

    if (grdROMS.theta_b > 0):
        Cbot = (np.exp(grdROMS.theta_b * Csur) - c1 ) / (c1 - np.exp(-grdROMS.theta_b))
        grdROMS['Cs_w'] = Cbot
    else:
        grdROMS['Cs_w'] = Csur     




    # Obtain z_r

    z0 = (grdROMS.hc * grdROMS.s_rho + grdROMS.h * grdROMS.Cs_r) / (grdROMS.hc + grdROMS.h)
    grdROMS['z_r'] = grdROMS.zeta + (grdROMS.zeta + grdROMS.h) * z0



    # Obtain z_w

    z0 = (grdROMS.hc * grdROMS.s_w + grdROMS.h * grdROMS.Cs_w) / (grdROMS.hc + grdROMS.h)
    grdROMS['z_w'] = grdROMS.zeta + (grdROMS.zeta + grdROMS.h) * z0



    # Also ESMF grid is added but probably not needed for VT



    grdROMS['L'] = grdROMS.Lp -1
    grdROMS['M'] = grdROMS.Mp -1

    
    
    
    
    return grdROMS








def obtain_VT_data(input_data, zr, bathymetry, zs, Nroms, Ndata, fill):
    ''' Function to obtain the vertical transformed data for a certain depth layer and grid location of the ROMS grid. A time series is returned. '''
    
    outdat = np.empty(Nroms)
    
    
    for kc in range(int(Nroms)): # Loop over ROMS depth layers (30)
        
    
        # Case 1: ROMS is deeper than GLORYS. This part searches for deepest good value if ROMS depth is deeper than GLORYS. 
        # This means that if no value, or only fill_value, is available from GLORYS where ROMS is deepest, the closest value from GLORYS is found by looping upward in the water column.

        # Between GLORYS/ORA and ROMS grid, CASE 1 will never happen
        if zr[kc] < zs[Ndata - 1]:
            
            
            outdat[kc] = input_data[Ndata - 1]
            
            
            #We do not want to give the deepest depth to be nan
            
            if not np.isnan(np.sum(input_data)):  

                if np.isnan(input_data[Ndata - 1]): 
                    
                    for kT in range(int(Ndata)):

                        if not np.isnan(input_data[Ndata - 1 - kT]):

                            outdat[kc] = input_data[Ndata - 1 - kT]


        # Case 2: ROMS depth layer is shallower than GLORYS depth layer. 

        elif zr[kc] > zs[0]:   
            
            
            outdat[kc] = input_data[0]
            

        else:

            for kT in range(int(Ndata) - 1): # Do loop between surface and bottom of GLORYS depth layers, - 1 because we also check for the next GLORYS layer each step

                # Case 3: ROMS depth layer is deeper than some GLORYS depth layer, but shallower than the next GLORYS layer which is below bottom 

                if (zr[kc] <= zs[kT]) & (-(bathymetry) > zs[kT + 1]):
                    
                    
                    outdat[kc] = input_data[kT]
                    
                        
                    #We do not want to give the deepest depth a nan value
                    
                    if not np.isnan(np.sum(input_data)):  

                        if np.isnan(input_data[Ndata - 1]): 

                            print(f'Case 3: deepest depth is nan. kc={kc}')
                            for kkT in range(int(Ndata)):

                                if not np.isnan(input_data[kT - kkT]):
                                    
                                    outdat[kc] = input_data[kT - kkT]
                
                # Case 4: Special case where ROMS layers are much deeper than in reanalysis data
                
                elif (zr[kc] <= zs[kT]) & np.invert(np.isnan(input_data[kT])) & np.isnan(input_data[kT + 1])  :
                    
                    outdat[kc] = input_data[kT]

            

                # Case 5: ROMS layer in between two reanalysis data layers. This is the typical case for most layers.

                elif (zr[kc] <= zs[kT]) & (zr[kc] >= zs[kT + 1]) & (-(bathymetry) <= zs[kT + 1]):

                    rz2 = abs((zr[kc] - zs[kT + 1]) / (abs(zs[kT + 1]) - abs(zs[kT])))

                    rz1 = 1.0 - rz2
                    
                    res = (rz1 * input_data[kT+1] + rz2 * input_data[kT])
                    
                    
                    outdat[kc] = res
                    
                    #We do not want to give nan value
                    if not np.isnan(np.sum(input_data)):  
                            
                            if np.isnan(input_data[kT]) or np.isnan(input_data[kT + 1]):
                                
                                print(f'Case 5: one of the values is nan. kc={kc}')
                                for kkT in range(Ndata):
                                            
                                    if not (np.isnan(input_data[kT-kkT])) & (np.isnan(input_data[kT - kkT + 1])):
                                        
                                        res = rz1 * input_data[kT + 1 - kkT] + rz2 * input_data[kT - kkT]
                                        
                                        outdat[kc] = res
                                        
                              
    return np.asarray(outdat)
                    
                    







def vertical_transf(data, grid):
    ''' Function to perform the vertical transformation.
    '''
    start_time = time.time()
    
    
    
    # Create reanalysis data and ROMS input grids
    grdMODEL = create_grdMODEL(data)
    grdROMS = create_grdROMS(grid)

    
    data = data.drop(['lat', 'lon']).rename({'zos': 'zeta'})                      
    
    
    
    print(f'===================== Start vertical transformation of variables with rho-coordinates (Duration: {round(time.time() - start_time,2)} seconds) ===================================')
    
    
    # Obtain variables
    bathymetry = grdROMS.h
    zr = grdROMS.z_r
    zs = grdMODEL.z_r
    Nroms = grdROMS.nlevels
    Ndata = grdMODEL.nlevels
    fill = -10000                                

    dat = data.copy()

    
    # Change the name of 'depth' and 's_rho' to 'z'
    dat = dat.rename({'depth' : 'z'})
    zs = zs.rename({'depth' : 'z'})
    zr = zr.rename({'s_rho' : 'z'})

    
    # Transpose dimensions
    zr = zr.transpose('z', 'eta_rho', 'xi_rho')
    bathymetry = bathymetry.transpose('eta_rho', 'xi_rho')


    # Change the arrangememnt of zr to make sure its ordered from surface to bottom
    zr = zr.sortby('z', ascending = False)
    
    
    
    
    print(f'===================== Start vertical transformation of Temperature (Duration: {round(time.time() - start_time,2)} seconds) ======================================================')
    
    dat_t = dat.thetao.transpose('z', 'time', 'eta_rho', 'xi_rho')
    
    
    theta_dataarray = xr.apply_ufunc(obtain_VT_data,                                                         # The function that should be executed
                             dat_t, zr, bathymetry, zs, Nroms, Ndata, fill,                                  # The arguments the function needs
                             input_core_dims=[['z'], ['z'], [], ['z'], [], [], []],                     # The list of core dimensions on each input argument that should not be broadcast
                             exclude_dims=set(('z',)), 
                             output_core_dims = [['z']],
                             dask = 'parallelized',
                             output_dtypes = [dat_t.dtype],
                             vectorize = True)
    
    theta_dataarray = theta_dataarray.rename('temp').rename({'z':'s_rho'}).assign_coords(s_rho = grid.s_rho.sortby('s_rho', ascending = False))
    
    theta_dataarray.to_netcdf(f'{output_dir}/data_hr_vt_T.nc')
    
    print(f'===================== Start vertical transformation of Salinity (Duration: {round((time.time() - start_time)/60,0)} minutes) =========================================================')
    
    
    
    
    dat_s = dat.so.transpose('z', 'time', 'eta_rho', 'xi_rho')
    
    
    sali_dataarray = xr.apply_ufunc(obtain_VT_data,                                                          # The function that should be executed
                             dat_s, zr, bathymetry, zs, Nroms, Ndata, fill,                                  # The arguments the function needs
                             input_core_dims=[['z'], ['z'], [], ['z'], [], [], []],                          # The list of core dimensions on each input argument that should not be broadcast
                             exclude_dims=set(('z',)), 
                             output_core_dims = [['z']],
                             dask = 'parallelized',
                             output_dtypes = [dat_s.dtype],
                             vectorize = True)

    sali_dataarray = sali_dataarray.rename('salt').rename({'z':'s_rho'}).assign_coords(s_rho = grid.s_rho.sortby('s_rho', ascending = False))
    
    sali_dataarray.to_netcdf(f'{output_dir}/data_hr_vt_S.nc')
    
    
    
    print(f'===================== Start vertical transformation of Zonal Velocity (Duration: {round((time.time() - start_time)/60,0)} minutes) ===================================================')
    
    # Change coordinate names such that apply_ufunc works correct
    dat_u = dat.uo.rename({'xi_u' : 'xi_rho', 'eta_u' : 'eta_rho'})
    
    dat_u = dat_u.transpose('z', 'time', 'eta_rho', 'xi_rho')
    
    # Since xi_u.size = 121 and xi_rho.size = 122, drop last values
    bathymetry_u = bathymetry[:, :-1]
    zr_u = zr[:, :, :-1]
    
    zonvel_dataarray = xr.apply_ufunc(obtain_VT_data,                                                        # The function that should be executed
                             dat_u, zr_u, bathymetry_u, zs, Nroms, Ndata, fill,                              # The arguments the function needs
                             input_core_dims=[['z'], ['z'], [], ['z'], [], [], []],                          # The list of core dimensions on each input argument that should not be broadcast
                             exclude_dims=set(('z',)), 
                             output_core_dims = [['z']],
                             dask = 'parallelized',
                             output_dtypes = [dat_u.dtype],
                             vectorize = True)

    zonvel_dataarray = zonvel_dataarray.rename('u').rename({'xi_rho' : 'xi_u', 'eta_rho' : 'eta_u', 'z':'s_rho'}).assign_coords(s_rho = grid.s_rho.sortby('s_rho', ascending = False))
    
    zonvel_dataarray.to_netcdf(f'{output_dir}/data_hr_vt_U.nc')
    
    
    
    print(f'===================== Start vertical transformation of Meridional Velocity (Duration: {round((time.time() - start_time)/60,0)} minutes) ==============================================')
    
    # Change coordinate names such that apply_ufunc works correct
    dat_v = dat.vo.rename({'xi_v' : 'xi_rho', 'eta_v' : 'eta_rho'})
    
    dat_v = dat_v.transpose('z', 'time', 'eta_rho', 'xi_rho')
    
    # Since eta_v.size = 109 and eta_rho.size = 110, drop last values
    bathymetry_v = bathymetry[:-1, :]
    zr_v = zr[:, :-1, :]
    
    mervel_dataarray = xr.apply_ufunc(obtain_VT_data,                                                        # The function that should be executed
                             dat_v, zr_v, bathymetry_v, zs, Nroms, Ndata, fill,                              # The arguments the function needs
                             input_core_dims=[['z'], ['z'], [], ['z'], [], [], []],                          # The list of core dimensions on each input argument that should not be broadcast
                             exclude_dims=set(('z',)), 
                             output_core_dims = [['z']],
                             dask = 'parallelized',
                             output_dtypes = [dat_v.dtype],
                             vectorize = True)

    
    mervel_dataarray = mervel_dataarray.rename('v').rename({'xi_rho' : 'xi_v', 'eta_rho' : 'eta_v', 'z':'s_rho'}).assign_coords(s_rho = grid.s_rho.sortby('s_rho', ascending = False))
    
    mervel_dataarray.to_netcdf(f'{output_dir}/data_hr_vt_V.nc')
    
    
    result = xr.merge([theta_dataarray, sali_dataarray, zonvel_dataarray, mervel_dataarray, data.zeta]).sortby('s_rho', ascending = True)
    
    
    
    
    print(f'===================== Finished vertical transformation (Duration: {round((time.time() - start_time)/60,0)} minutes) ==================================================================')
    
    
    result.to_netcdf(f'{output_dir}/data_hr_vt.nc')
    
    
    
    return result



data_hr_vt = vertical_transf(data_hr, grid)




###############        Calculate UBAR and VBAR        ###############

    
    
    
def integrate_1D(data, z_w, Nroms):
    '''
    
    Function to perform the vertical integration for one horizontal grid point.
    '''
    
    mom = 0.0
    
    for kc in range(Nroms):
        
        mom = mom + data[kc] * abs(z_w[kc + 1] - z_w[kc])
    
    if abs(z_w[0]) > 0.0:
        
        mom = mom / abs(z_w[0])
    
    else:
        
        mom = 0.0
        
    
    return np.asarray(mom)



def vert_integrate_mom(data, grid):
    '''
    Calculate vertically integrated momentum component in eta (UBAR) and xi (VBAR) direction
    '''
    
    print(f'===================== Start integrating vertical momentum  =============================================================')
    
    
    
    # Obtain data
    dat_u = data.u
    dat_v = data.v
    
    # Create reanalysis data and ROMS input grids
    grdROMS = create_grdROMS(grid)
    
    # Obtain variables
    zr = grdROMS.z_r
    z_w = grdROMS.z_w
    Nroms = int(grdROMS.nlevels)
    
    
    # Obtain z_wu
    z_wu = 0.5*(z_w + z_w.shift(xi_rho = -1)).rename({'xi_rho' : 'xi_u', 'eta_rho' : 'eta_u', 's_w': 's_rho'})
    z_wv = 0.5*(z_w + z_w.shift(eta_rho = -1)).rename({'xi_rho' : 'xi_v', 'eta_rho' : 'eta_v', 's_w': 's_rho'})
        

    # Since xi_u.size = 121 and xi_rho.size = 122, drop last values
    z_wu = z_wu[:, :-1, :]
    z_wv = z_wv[:-1, :, :]

    print(f'===================== Start calculating UBAR ===========================================================================')
    
    
    # Calculate UBAR
    UBAR = xr.apply_ufunc(integrate_1D,
                         dat_u, z_wu, Nroms,
                         input_core_dims=[['s_rho'], ['s_rho'], []],
                         exclude_dims=set(('s_rho',)),
                         dask = 'parallelized',
                         output_dtypes = [dat_u.dtype],
                         vectorize = True)

    
    print(f'===================== Start calculating VBAR ===========================================================================')
    
    VBAR = xr.apply_ufunc(integrate_1D,
                         dat_v, z_wv, Nroms,
                         input_core_dims=[['s_rho'], ['s_rho'], []],
                         exclude_dims=set(('s_rho',)),
                         dask = 'parallelized',
                         output_dtypes = [dat_v.dtype],
                         vectorize = True)

    
    
    result = xr.merge([data, UBAR.rename('ubar'), VBAR.rename('vbar')])
    
    
    print(f'===================== Finished calculating vertical momentum ===========================================================')
    
    
    result.to_netcdf(f'{output_dir}/result.nc')
    
    
    return result
    
    
result = vert_integrate_mom(data_hr_vt, grid)


# Change time dimension to accomodate ROMS settings
ocean_time = pd.date_range(start=f'{start_year}/{start_month}/{start_day}', end=f'{final_year}/{final_month}/{final_day}', freq=DateOffset(months=1))



# Change NaN values to ROMS fillvalue
grdROMS = create_grdROMS(grid)

result['u'] = result.u.where(grdROMS.mask_u == 1, grdROMS.fillval.data)
result['v'] = result.v.where(grdROMS.mask_v == 1, grdROMS.fillval.data)

result['salt'] = result.salt.where(grdROMS.mask_rho == 1, grdROMS.fillval.data)
result['temp'] = result.temp.where(grdROMS.mask_rho == 1, grdROMS.fillval.data)
result['zeta'] = result.zeta.where(grdROMS.mask_rho == 1, grdROMS.fillval.data)

result['ubar'] = result.ubar.where(grdROMS.mask_u == 1, grdROMS.fillval.data)
result['vbar'] = result.vbar.where(grdROMS.mask_v == 1, grdROMS.fillval.data)





###############        Obtain forcing files       ###############


# Make climatology forcing file
result_clim = result
grdROMS = create_grdROMS(grid)

clim = xr.Dataset(data_vars=dict(lon_rho=(["eta_rho", "xi_rho"], grid.lon_rho.values),
                                lat_rho=(["eta_rho", "xi_rho"], grid.lat_rho.values),
                                lon_u=(["eta_u", "xi_u"], grid.lon_u.values),
                                lat_u=(["eta_u", "xi_u"], grid.lat_u.values),
                                lon_v=(["eta_v", "xi_v"], grid.lon_v.values),
                                lat_v=(["eta_v", "xi_v"], grid.lat_v.values),
                                lon_psi=(["eta_psi", "xi_psi"], grid.lon_psi.values),
                                lat_psi=(["eta_psi", "xi_psi"], grid.lat_psi.values),
                                h=(["eta_rho", "xi_rho"], grid.h.values),
                                f=(["eta_rho", "xi_rho"], grid.f.values),
                                pm=(["eta_rho", "xi_rho"], grid.pm.values),
                                pn=(["eta_rho", "xi_rho"], grid.pn.values),
                                Cs_r=(["s_rho"], grid.Cs_r.values),
                                Cs_w=(["s_w"], grid.Cs_w.values),
                                hc=([], grid.hc.values),
                                Tcline=([], grid.Tcline.values),
                                theta_s=([], grid.theta_s.values),
                                theta_b=([], grid.theta_b.values),
                                angle=(["eta_rho", "xi_rho"], grid.angle.values),
                                z_r=(["s_rho", "eta_rho", "xi_rho"], grdROMS.z_r.transpose("s_rho", "eta_rho", "xi_rho").values),
                                z_w=(["s_w", "eta_rho", "xi_rho"], grdROMS.z_w.transpose("s_w", "eta_rho", "xi_rho").values),
                                u=(["ocean_time", "s_rho", "eta_u", "xi_u"], result_clim.u.transpose("time", "s_rho", "eta_u", "xi_u").values),
                                v=(["ocean_time", "s_rho", "eta_v", "xi_v"], result_clim.v.transpose("time", "s_rho", "eta_v", "xi_v").values),
                                salt=(["ocean_time", "s_rho", "eta_rho", "xi_rho"], result_clim.salt.transpose("time", "s_rho", "eta_rho", "xi_rho").values),
                                temp=(["ocean_time", "s_rho", "eta_rho", "xi_rho"], result_clim.temp.transpose("time", "s_rho", "eta_rho", "xi_rho").values),
                                zeta=(["ocean_time", "eta_rho", "xi_rho"], result_clim.zeta.values),
                                ubar=(["ocean_time", "eta_u", "xi_u"], result_clim.ubar.transpose("time", "eta_u", "xi_u").values),
                                vbar=(["ocean_time", "eta_v", "xi_v"], result_clim.vbar.transpose("time", "eta_v", "xi_v").values),),
                  coords=dict(s_rho=(["s_rho"], grid.s_rho.values),
                              s_w=(["s_w"], grid.s_w.values),
                              ocean_time=(["ocean_time"], ocean_time),
                                  ))


# variable attributes
clim.lon_rho.attrs = {'long_name' : 'Longitude of RHO-points',
                      'units' : 'degree_east',
                      'standard_name' : 'longitude',
               '_FillValue' : grdROMS.fillval.data}
clim.lat_rho.attrs = {'long_name' : 'Latitude of RHO-points',
                      'units' : 'degree_north',
                      'standard_name' : 'latitude',
               '_FillValue' : grdROMS.fillval.data}
clim.lon_u.attrs = {'long_name' : 'Longitude of U-points',
                      'units' : 'degree_east',
                      'standard_name' : 'longitude',
               '_FillValue' : grdROMS.fillval.data}
clim.lat_u.attrs = {'long_name' : 'Latitude of U-points',
                      'units' : 'degree_north',
                      'standard_name' : 'latitude',
               '_FillValue' : grdROMS.fillval.data}
clim.lon_v.attrs = {'long_name' : 'Longitude of V-points',
                      'units' : 'degree_east',
                      'standard_name' : 'longitude',
               '_FillValue' : grdROMS.fillval.data}
clim.lat_v.attrs = {'long_name' : 'Latitude of V-points',
                      'units' : 'degree_north',
                      'standard_name' : 'latitude',
               '_FillValue' : grdROMS.fillval.data}
clim.lon_psi.attrs = {'long_name' : 'Longitude of PSI-points',
                      'units' : 'degree_east',
                      'standard_name' : 'longitude',
               '_FillValue' : grdROMS.fillval.data}
clim.lat_psi.attrs = {'long_name' : 'Latitude of PSI-points',
                      'units' : 'degree_north',
                      'standard_name' : 'latitude',
               '_FillValue' : grdROMS.fillval.data}
clim.h.attrs = {'long_name' : 'Bathymetry at RHO-points',
                      'units' : 'meter',
                      'standard_name' : 'bath, scalar',
               '_FillValue' : grdROMS.fillval.data}
clim.f.attrs = {'long_name' : 'Coriolis parameter at RHO-points',
                      'units' : 'second-1',
                      'standard_name' : 'Coriolis, scalar',
               '_FillValue' : grdROMS.fillval.data}
clim.pm.attrs = {'long_name' : 'curvilinear coordinate metric in XI',
                      'units' : 'meter-1',
                      'standard_name' : 'pm, scalar',
               '_FillValue' : grdROMS.fillval.data}
clim.pn.attrs = {'long_name' : 'curvilinear coordinate metric in ETA',
                      'units' : 'meter-1',
                      'standard_name' : 'pn, scalar',
               '_FillValue' : grdROMS.fillval.data}
clim.Cs_r.attrs = {'long_name' : 'S-coordinate stretching curves at RHO-points',
                   'valid_min' : -1.,
                   'valid_max' : 0.,
                   'field' : 's_rho, scalar',
               '_FillValue' : grdROMS.fillval.data}
clim.Cs_w.attrs = {'long_name' : 'S-coordinate stretching curves at W-points',
                   'valid_min' : -1.,
                   'valid_max' : 0.,
                   'field' : 's_w, scalar',
               '_FillValue' : grdROMS.fillval.data}
clim.hc.attrs = {'long_name' : 'S-coordinate parameter, critical depth',
                   'units' : 'meter'}
clim.Tcline.attrs = {'long_name' : 'S-coordinate surface/bottom layer depth',
                   'units' : 'meter'}
clim.theta_s.attrs = {'long_name' : 'S-coordinate surface control parameter'}
clim.theta_b.attrs = {'long_name' : 'S-coordinate bottom control parameter'}
clim.angle.attrs = {'long_name' : 'angle between xi axis and east',
                   'units' : 'radian',
               '_FillValue' : grdROMS.fillval.data}
clim.z_r.attrs = {'long_name' : 'Sigma layer to depth matrix',
                   'units' : 'meter',
               '_FillValue' : grdROMS.fillval.data}
clim.z_w.attrs = {'long_name' : 'Sigma layer to depth matrix',
                   'units' : 'meter',
               '_FillValue' : grdROMS.fillval.data}
clim.u.attrs = {'long_name' : 'u-momentum component',
                   'units' : 'meter second-1',
               'time' : 'ocean_time',
               'field' : 'u-velocity, scalar, series',
               '_FillValue' : grdROMS.fillval.data}
clim.v.attrs = {'long_name' : 'v-momentum component',
                   'units' : 'meter second-1',
               'time' : 'ocean_time',
               'field' : 'v-velocity, scalar, series',
               '_FillValue' : grdROMS.fillval.data}
clim.salt.attrs = {'long_name' : 'salinity',
               'time' : 'ocean_time',
               'field' : 'salinity, scalar, series',
               '_FillValue' : grdROMS.fillval.data}
clim.temp.attrs = {'long_name' : 'potential temperature',
                   'units' : 'Celsius',
               'time' : 'ocean_time',
               'field' : 'temperature, scalar, series',
               '_FillValue' : grdROMS.fillval.data}
clim.zeta.attrs = {'long_name' : 'sea level',
                   'units' : 'meter',
               'time' : 'ocean_time',
               'field' : 'sea level, scalar, series',
               '_FillValue' : grdROMS.fillval.data}
clim.ubar.attrs = {'long_name' : 'u-2D momentum',
                   'units' : 'meter second-1',
               'time' : 'ocean_time',
               'field' : 'u-2D velocity, scalar, series',
               '_FillValue' : grdROMS.fillval.data}
clim.vbar.attrs = {'long_name' : 'v-2D momentum',
                   'units' : 'meter second-1',
               'time' : 'ocean_time',
               'field' : 'v-2D velocity, scalar, series',
               '_FillValue' : grdROMS.fillval.data}
clim.s_rho.attrs = {'long_name' : 'S-coordinate at RHO-points',
                   'valid_min' : -1.,
                   'valid_max' : 0.,
                   'standard_name' : 'ocean_s_coordinate_g2',
                   'formula terms' : 's: s_rho C: Cs_r eta: zeta depth: h depth_c: hc',
                   'field' : 's_rho, scalar',
               '_FillValue' : grdROMS.fillval.data}
clim.s_w.attrs = {'long_name' : 'S-coordinate at W-points',
                   'valid_min' : -1.,
                   'valid_max' : 0.,
                   'standard_name' : 'ocean_s_coordinate_g2',
                   'formula terms' : 's: s_w C: Cs_w eta: zeta depth: h depth_c: hc',
                   'field' : 's_w, scalar',
               '_FillValue' : grdROMS.fillval.data}
clim.ocean_time.attrs = {'long_name' : 'seconds since 1948-01-01 00:00:00',
                        'field' : 'time, scalar, series',
               '_FillValue' : grdROMS.fillval.data}
clim.ocean_time.encoding = {'units' : 'seconds since 1948-01-01 00:00:00',
                        'calendar' : 'standard'}





# global attributes
clim.attrs = {'title' : 'Climatology forcing file (CLIM) used for forcing the ROMS model',
             'description' : 'Created for grid file: NorthSea4_smooth01_sponge_nudg.nc',
             'grd_file' : 'Gridfile: .../NorthSea4_smooth01_sponge_nudg.nc',
             'history' : f'Created {datetime.date.today().strftime("%B %d, %Y")}',
             'conventions' : 'CF-1.0'}


# Make initial forcing file
init = clim.isel(ocean_time = [0])


# Change some variables
init['Cs_rho'] = init.Cs_r

# Drop some variables
init = init.drop(['Cs_r', 'Cs_w', 'pm', 'pn', 'f'])


# global attributes
init.attrs = {'title' : 'Initial forcing file (INIT) used for forcing the ROMS model',
             'description' : 'Created for grid file: NORTH_SEA4',
             'grd_file' : 'Gridfile: .../NorthSea4_smooth01_sponge_nudg.nc',
             'history' : f'Created {datetime.date.today().strftime("%B %d, %Y")}'}




# Make boundary forcing file
bry = clim.copy()



# Assign lon_rho and lat_rho as coordinates, not as variables
bry.assign_coords(lon_rho = bry.lon_rho, lat_rho = bry.lat_rho)



# Create boundary variables
bry['temp_west'] = bry.temp.isel(xi_rho = 0)
bry['temp_east'] = bry.temp.isel(xi_rho = -1)
bry['temp_south'] = bry.temp.isel(eta_rho = 0)
bry['temp_north'] = bry.temp.isel(eta_rho = -1)

bry['salt_west'] = bry.salt.isel(xi_rho = 0)
bry['salt_east'] = bry.salt.isel(xi_rho = -1)
bry['salt_south'] = bry.salt.isel(eta_rho = 0)
bry['salt_north'] = bry.salt.isel(eta_rho = -1)

bry['zeta_west'] = bry.zeta.isel(xi_rho = 0)
bry['zeta_east'] = bry.zeta.isel(xi_rho = -1)
bry['zeta_south'] = bry.zeta.isel(eta_rho = 0)
bry['zeta_north'] = bry.zeta.isel(eta_rho = -1)

bry['u_west'] = bry.u.isel(xi_u = 0)
bry['u_east'] = bry.u.isel(xi_u = -1)
bry['u_south'] = bry.u.isel(eta_u = 0)
bry['u_north'] = bry.u.isel(eta_u = -1)

bry['v_west'] = bry.v.isel(xi_v = 0)
bry['v_east'] = bry.v.isel(xi_v = -1)
bry['v_south'] = bry.v.isel(eta_v = 0)
bry['v_north'] = bry.v.isel(eta_v = -1)

bry['ubar_west'] = bry.ubar.isel(xi_u = 0)
bry['ubar_east'] = bry.ubar.isel(xi_u = -1)
bry['ubar_south'] = bry.ubar.isel(eta_u = 0)
bry['ubar_north'] = bry.ubar.isel(eta_u = -1)

bry['vbar_west'] = bry.vbar.isel(xi_v = 0)
bry['vbar_east'] = bry.vbar.isel(xi_v = -1)
bry['vbar_south'] = bry.vbar.isel(eta_v = 0)
bry['vbar_north'] = bry.vbar.isel(eta_v = -1)




# Assign attributes to dimensions
bry.eta_u.attrs = {}
bry.xi_u.attrs = {}
bry.eta_v.attrs = {}
bry.xi_v.attrs = {}
bry.eta_psi.attrs = {}
bry.xi_psi.attrs = {}
bry.eta_rho.attrs = {}
bry.xi_rho.attrs = {}
bry.s_rho.attrs = {'long_name': 'S-coordinate at RHO-points',
 'valid_min': -1.0,
 'valid_max': 0.0,
 'standard_name': 'ocean_s_coordinate_g2',
 'formula_terms': 's: s_rho C: Cs_r eta: zeta depth: h depth_c: hc',
 'field': 's_rho, scalar'}
bry.s_w.attrs = {'long_name': 'S-coordinate at W-points',
 'valid_min': -1.0,
 'valid_max': 0.0,
 'standard_name': 'ocean_s_coordinate_g2',
 'formula_terms': 's: s_w C: Cs_w eta: zeta depth: h depth_c: hc',
 'field': 's_w, scalar'}
bry.ocean_time.attrs = {'long_name': 'seconds since 1948-01-01 00:00:00',
 'field': 'time, scalar, series'}
bry.lon_rho.attrs = {'long_name': 'Longitude of RHO-points',
 'units': 'degree_east',
 'standard_name': 'longitude'}
bry.lat_rho.attrs = {'long_name': 'Latitude of RHO-points',
 'units': 'degree_north',
 'standard_name': 'latitude'}


# Assign attributes to variables
bry.lon_u.attrs = {'long_name': 'Longitude of U-points',
 'units': 'degree_east',
 'standard_name': 'longitude'}
bry.lat_u.attrs = {'long_name': 'Latitude of U-points',
 'units': 'degree_north',
 'standard_name': 'latitude'}
bry.lon_v.attrs = {'long_name': 'Longitude of V-points',
 'units': 'degree_east',
 'standard_name': 'longitude'}
bry.lat_v.attrs = {'long_name': 'Latitude of V-points',
 'units': 'degree_north',
 'standard_name': 'latitude'}
bry.lon_psi.attrs = {'long_name': 'Longitude of PSI-points',
 'units': 'degree_east',
 'standard_name': 'longitude'}
bry.lat_psi.attrs = {'long_name': 'Latitude of PSI-points',
 'units': 'degree_north',
 'standard_name': 'latitude'}
bry.h.attrs = {'long_name': 'Bathymetry at RHO-points',
 'units': 'meter',
 'field': 'bath, scalar'}
bry.Cs_r.attrs = {'long_name': 'S-coordinate stretching curves at RHO-points',
 'valid_min': -1.0,
 'valid_max': 0.0,
 'field': 'Cs_rho, scalar'}
bry.Cs_w.attrs = {'long_name': 'S-coordinate stretching curves at W-points',
 'valid_min': -1.0,
 'valid_max': 0.0,
 'field': 'Cs_w, scalar'}
bry.hc.attrs = {'long_name': 'S-coordinate parameter, critical depth', 'units': 'meter'}
bry.z_r.attrs = {'long_name': 'Sigma layer to depth matrix', 'units': 'meter'}
bry.Tcline.attrs = {'long_name': 'S-coordinate surface/bottom layer width', 'units': 'meter'}
bry.theta_s.attrs = {'long_name': 'S-coordinate surface control parameter'}
bry.theta_b.attrs = {'long_name': 'S-coordinate bottom control parameter'}
bry.angle.attrs = {'long_name': 'angle between xi axis and east', 'units': 'radian'}
bry.temp_west.attrs = {'long_name': 'potential temperature western boundary condition',
 'units': 'Celsius',
 'field': 'temp_west, scalar, series',
 'time': 'ocean_time'}
bry.temp_east.attrs = {'long_name': 'potential temperature eastern boundary condition',
 'units': 'Celsius',
 'field': 'temp_east, scalar, series',
 'time': 'ocean_time'}
bry.temp_south.attrs = {'long_name': 'potential temperature southern boundary condition',
 'units': 'Celsius',
 'field': 'temp_south, scalar, series',
 'time': 'ocean_time'}
bry.temp_north.attrs = {'long_name': 'potential temperature northern boundary condition',
 'units': 'Celsius',
 'field': 'temp_north, scalar, series',
 'time': 'ocean_time'}
bry.salt_west.attrs = {'long_name': 'salinity western boundary condition',
 'field': 'salt_west, scalar, series',
 'time': 'ocean_time'}
bry.salt_east.attrs = {'long_name': 'salinity eastern boundary condition',
 'field': 'salt_east, scalar, series',
 'time': 'ocean_time'}
bry.salt_south.attrs = {'long_name': 'salinity southern boundary condition',
 'field': 'salt_south, scalar, series',
 'time': 'ocean_time'}
bry.salt_north.attrs = {'long_name': 'salinity northern boundary condition',
 'field': 'salt_north, scalar, series',
 'time': 'ocean_time'}
bry.zeta_west.attrs = {'long_name': 'free-surface western boundary condition',
 'units': 'meter',
 'field': 'zeta_west, scalar, series',
 'time': 'ocean_time'}
bry.zeta_east.attrs = {'long_name': 'free-surface eastern boundary condition',
 'units': 'meter',
 'field': 'zeta_east, scalar, series',
 'time': 'ocean_time'}
bry.zeta_south.attrs = {'long_name': 'free-surface southern boundary condition',
 'units': 'meter',
 'field': 'zeta_south, scalar, series',
 'time': 'ocean_time'}
bry.zeta_north.attrs = {'long_name': 'free-surface northern boundary condition',
 'units': 'meter',
 'field': 'zeta_north, scalar, series',
 'time': 'ocean_time'}
bry.u_west.attrs = {'long_name': '3D u-momentum western boundary condition',
 'units': 'meter second-1',
 'field': 'u_west, scalar, series',
 'time': 'ocean_time'}
bry.u_east.attrs = {'long_name': '3D u-momentum eastern boundary condition',
 'units': 'meter second-1',
 'field': 'u_east, scalar, series',
 'time': 'ocean_time'}
bry.u_south.attrs = {'long_name': '3D u-momentum southern boundary condition',
 'units': 'meter second-1',
 'field': 'u_south, scalar, series',
 'time': 'ocean_time'}
bry.u_north.attrs = {'long_name': '3D u-momentum northern boundary condition',
 'units': 'meter second-1',
 'field': 'u_north, scalar, series',
 'time': 'ocean_time'}
bry.v_west.attrs = {'long_name': '3D v-momentum western boundary condition',
 'units': 'meter second-1',
 'field': 'v_west, scalar, series',
 'time': 'ocean_time'}
bry.v_east.attrs = {'long_name': '3D v-momentum eastern boundary condition',
 'units': 'meter second-1',
 'field': 'v_east, scalar, series',
 'time': 'ocean_time'}
bry.v_south.attrs = {'long_name': '3D v-momentum southern boundary condition',
 'units': 'meter second-1',
 'field': 'v_south, scalar, series',
 'time': 'ocean_time'}
bry.v_north.attrs = {'long_name': '3D v-momentum northern boundary condition',
 'units': 'meter second-1',
 'field': 'v_north, scalar, series',
 'time': 'ocean_time'}
bry.ubar_west.attrs = {'long_name': '2D u-momentum western boundary condition',
 'units': 'meter second-1',
 'field': 'ubar_west, scalar, series',
 'time': 'ocean_time'}
bry.ubar_east.attrs = {'long_name': '2D u-momentum eastern boundary condition',
 'units': 'meter second-1',
 'field': 'ubar_east, scalar, series',
 'time': 'ocean_time'}
bry.ubar_south.attrs = {'long_name': '2D u-momentum southern boundary condition',
 'units': 'meter second-1',
 'field': 'ubar_south, scalar, series',
 'time': 'ocean_time'}
bry.ubar_north.attrs = {'long_name': '2D u-momentum northern boundary condition',
 'units': 'meter second-1',
 'field': 'ubar_north, scalar, series',
 'time': 'ocean_time'}
bry.vbar_west.attrs = {'long_name': '2D v-momentum western boundary condition',
 'units': 'meter second-1',
 'field': 'vbar_west, scalar, series',
 'time': 'ocean_time'}
bry.vbar_east.attrs = {'long_name': '2D v-momentum eastern boundary condition',
 'units': 'meter second-1',
 'field': 'vbar_east, scalar, series',
 'time': 'ocean_time'}
bry.vbar_south.attrs = {'long_name': '2D v-momentum southern boundary condition',
 'units': 'meter second-1',
 'field': 'vbar_south, scalar, series',
 'time': 'ocean_time'}
bry.vbar_north.attrs = {'long_name': '2D v-momentum northern boundary condition',
 'units': 'meter second-1',
 'field': 'vbar_north, scalar, series',
 'time': 'ocean_time'}


# Add Fill value attribute
bry.temp_west.attrs['_FillValue'] = grdROMS.fillval.data
bry.temp_east.attrs['_FillValue'] = grdROMS.fillval.data
bry.temp_south.attrs['_FillValue'] = grdROMS.fillval.data
bry.temp_north.attrs['_FillValue'] = grdROMS.fillval.data
bry.salt_west.attrs['_FillValue'] = grdROMS.fillval.data
bry.salt_east.attrs['_FillValue'] = grdROMS.fillval.data
bry.salt_south.attrs['_FillValue'] = grdROMS.fillval.data
bry.salt_north.attrs['_FillValue'] = grdROMS.fillval.data
bry.zeta_west.attrs['_FillValue'] = grdROMS.fillval.data
bry.zeta_east.attrs['_FillValue'] = grdROMS.fillval.data
bry.zeta_south.attrs['_FillValue'] = grdROMS.fillval.data
bry.zeta_north.attrs['_FillValue'] = grdROMS.fillval.data
bry.u_west.attrs['_FillValue'] = grdROMS.fillval.data
bry.u_east.attrs['_FillValue'] = grdROMS.fillval.data
bry.u_south.attrs['_FillValue'] = grdROMS.fillval.data
bry.u_north.attrs['_FillValue'] = grdROMS.fillval.data
bry.v_west.attrs['_FillValue'] = grdROMS.fillval.data
bry.v_east.attrs['_FillValue'] = grdROMS.fillval.data
bry.v_south.attrs['_FillValue'] = grdROMS.fillval.data
bry.v_north.attrs['_FillValue'] = grdROMS.fillval.data
bry.ubar_west.attrs['_FillValue'] = grdROMS.fillval.data
bry.ubar_east.attrs['_FillValue'] = grdROMS.fillval.data
bry.ubar_south.attrs['_FillValue'] = grdROMS.fillval.data
bry.ubar_north.attrs['_FillValue'] = grdROMS.fillval.data
bry.vbar_west.attrs['_FillValue'] = grdROMS.fillval.data
bry.vbar_east.attrs['_FillValue'] = grdROMS.fillval.data
bry.vbar_south.attrs['_FillValue'] = grdROMS.fillval.data
bry.vbar_north.attrs['_FillValue'] = grdROMS.fillval.data




# Drop some variables
bry = bry.drop(['f', 'pm', 'pn', 'z_w', 'temp', 'salt', 'zeta', 'u', 'v', 'ubar', 'vbar'])



# global attributes
bry.attrs = {'title' : 'Boundary forcing file (BRY) used for forcing the ROMS model',
             'description' : 'Created for grid file: NORTH_SEA4',
             'grd_file' : 'Gridfile: .../NorthSea4_smooth01_sponge_nudg.nc',
             'history' : f'Created {datetime.date.today().strftime("%B %d, %Y")}'}





# Save forcing files
init.to_netcdf(f'{output_dir}/NorthSea4_init_{data_type}_{start_year}{start_month}{start_day}_to_{final_year}{final_month}{final_day}.nc')
bry.to_netcdf(f'{output_dir}/NorthSea4_bry_{data_type}_{start_year}{start_month}{start_day}_to_{final_year}{final_month}{final_day}.nc')
clim.to_netcdf(f'{output_dir}/NorthSea4_clim_{data_type}_{start_year}{start_month}{start_day}_to_{final_year}{final_month}{final_day}.nc')















    
