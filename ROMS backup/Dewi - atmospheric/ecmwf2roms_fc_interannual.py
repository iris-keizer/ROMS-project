######################################################################
# ecmwf2roms_fc_interannual: Prepare the ERA-interim data to be used in ROMS
#             - Change units
#             - Convert accumulated fields to flux
#             - Multiply by scale factor and add offset
#             (it seems that when importing variables from a dataset the 
#              offset and scale factor are automatically applied)
#             - Convert the short wave radiation from 12h accumulated 
#             to 24h averaged flux
#             - Loop on the years
######################################################################
from netCDF4 import Dataset
import os
import math
import numpy as np
import time as timemod
from datetime import datetime

# Directory from which to read and write files
InputOutputDir = "/nobackup/users/bars/ERA-interim/FromWebsite/"

for Year in range(1993,2015,1):
    name_input     = InputOutputDir + 'ERAint_fc_' + str(Year) + '_NA.nc'
    name_output    = InputOutputDir + 'ERAint_fc_' + str(Year) + '_NA_ForROMS.nc'

    fcif = Dataset( name_input, 'r')

    if os.path.isfile(name_output):
        os.remove(name_output)
    fcof = Dataset( name_output, 'w', format='NETCDF4')

    # Create dimensions in output files with same sizes as in input file
    time_rain  = fcof.createDimension('time_rain', len(fcif.dimensions['time']))
    time_swrad = fcof.createDimension('time_swrad', len(fcif.dimensions['time'])/2)
    lon  = fcof.createDimension('lon', len(fcif.dimensions['longitude']))
    lat  = fcof.createDimension('lat', len(fcif.dimensions['latitude']))

    # Create variables in output files
    times_rain  = fcof.createVariable('time_rain','i8',('time_rain',))
    times_swrad = fcof.createVariable('time_swrad','i8',('time_swrad',))
    lons  = fcof.createVariable('lon','f8',('lon',))
    lats  = fcof.createVariable('lat','f8',('lat',))
    swrad = fcof.createVariable('swrad','f8',('time_swrad','lat','lon',))
    rain  = fcof.createVariable('rain','f8',('time_rain','lat','lon',))

    # Copy or modify values from input files
    times        = fcif.variables['time']
    times2       = np.int64(times[:])*60*60           # Convert to seconds
    base_era     = datetime(1900, 1, 1)     # Change time reference
    base_target  = datetime(1948, 1, 1)
    times2[:]    = times2[:] - (base_target - base_era).total_seconds()

    times_rain[:] = times2[:] - 6*60*60          # Remove 6 hours to center the flux to the middle
                                                 # of the forcasting time (12h) instead of the beginning
    times_swrad[:] = times2[::2]                     # Read half of the values, at mid day
    lons[:]      = fcif.variables['longitude']
    lats[:]      = fcif.variables['latitude'][::-1]  # Reverse the array to make it from min to max
    # Divide by the variables bellow of sec in 12h to convert to flux
    swrad_i      = fcif.variables['ssr'][:,:,:]/(12*60*60)    # From W/m**2.s to W/m**2
    swrad[:,:,:] = (swrad_i[0::2,:,:]+swrad_i[1::2,:,:])/2
    swrad[:,:,:] = swrad[:,::-1,:]
    rain[:,:,:]  = fcif.variables['tp'][:,:,:]*1000/(12*60*60) # From m to kg/m**2/s
    rain[:,:,:]  = rain[:,::-1,:]

    # Attributes
    fcof.history          = 'Created ' + timemod.ctime(timemod.time())
    times_rain.units      = 'seconds since 1948-01-01 00:00:0.0'
    times_rain.long_name  = 'time'
    times_rain.calendar   = 'gregorian'
    times_swrad.units     = 'seconds since 1948-01-01 00:00:0.0'
    times_swrad.long_name = 'time'
    times_swrad.calendar  = 'gregorian'
    lons.units       = 'degrees_north'
    lons.long_name   = 'Longitude'
    lats.units       = 'degrees_east'
    lats.long_name   = 'Latitude'
    swrad.units      = 'W/m**2'
    swrad.long_name  = 'Surface solar radiation'
    swrad.coordinates = 'lon lat'
    swrad.time       = 'time_swrad'
    rain.units       = 'kg/m**2/s'
    rain.long_name   = 'Rain fall rate'
    rain.coordinates = 'lon lat'
    rain.time        = 'time_rain'

    fcof.close()
    fcif.close()
