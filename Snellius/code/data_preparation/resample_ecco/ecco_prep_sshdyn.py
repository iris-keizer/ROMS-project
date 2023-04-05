# Import packages
import xarray as xr
import ecco_v4_py as ecco
import numpy as np
from IPython.utils import io

# Declare some variables
path = '/projects/0/einf2878/ROMS/data/ECCO'
path_input_data = f'{path}/nctiles/'
variable = 'sshdyn'


# Input data
data = xr.open_mfdataset(f'{path_input_data}/{variable}/*.nc')
grid = xr.open_dataset(f'{path_input_data}/grid/ECCO-GRID.nc')

test_data = True

if test_data == True:
    data = data.isel(time=range(3))


# Modules to resample the data
def resample_to_latlon_3d(dataset, variable, resolution = 1.0, mapping_method = 'nearest_neighbor', radius_of_influence = 120000):
    '''
    Function to resample a global ECCO dataset to a latitude-longitude grid. Since the 'resample_to_latlon' function of ecco_v4_py does only work on 2D 
    datasets, a for-loop is used to make it work on 3D datasets.
    
    '''
    
    # Define the new horizontal resolution
    new_grid_delta_lat = resolution
    new_grid_delta_lon = resolution
    
    # Define latitude minimum and maximum of new grid
    new_grid_min_lat = -90+new_grid_delta_lat/2
    new_grid_max_lat = 90-new_grid_delta_lat/2
        
    # Define longitude minimum and maximum of new grid
    new_grid_min_lon = -180+new_grid_delta_lon/2
    new_grid_max_lon = 180-new_grid_delta_lon/2
    
    
    result = []
    
    for year in dataset.time:
            
        original_field_with_land_mask= np.where(grid.maskC.isel(k=0)>0, dataset[variable].sel(time=year), np.nan)
            
        with io.capture_output() as captured:
            result_year = ecco.resample_to_latlon(dataset.XC, dataset.YC, original_field_with_land_mask, new_grid_min_lat, new_grid_max_lat, 
                                         new_grid_delta_lat, new_grid_min_lon, new_grid_max_lon, new_grid_delta_lon,
                                         fill_value = np.NaN, mapping_method = mapping_method, 
                                         radius_of_influence = radius_of_influence)
            
            new_grid_lon_centers, new_grid_lat_centers, new_grid_lon_edges, new_grid_lat_edges, field_nearest_1deg = result_year
            
            result.append(field_nearest_1deg)
            
    # Create a new dataarray
    DA = xr.DataArray(result, 
                      name = 'SSHDYN', 
                      dims = ['time', 'latitude', 'longitude'], 
                      coords = {'time': dataset.time, 'latitude': new_grid_lat_centers[:,0], 'longitude': new_grid_lon_centers[0,:]}) 
    
    
    return DA



# Make function work on dataset
data_resampled = resample_to_latlon_3d(data, 'SSHDYN')


# Save data

if test_data == True:
    path_save = f'{path}/resampled_test'
else:
    path_save = f'{path}/resampled'

data_resampled.to_netcdf(f'{path_save}/sshdyn.nc')







