######################################################################
# ecmwf2roms_an_interannual: Prepare the ERA-interim data to be used in ROMS
#             - Change units
#             - Compute relative humidity from temperature and dewpoint temperature
#             - Multiply by scale factor and add offset
#             (it seems that when importing variables from a dataset the 
#              offset and scale factor are automatically applied)
#             - Loop on years
######################################################################

from netCDF4 import Dataset
import os
import math
import numpy as np
import time as timemod
from datetime import datetime

# Directory from which to read and write files
InputOutputDir = "/nobackup/users/bars/ERA-interim/FromWebsite/"

for year in range(1993,2015,1):
    name_input     = InputOutputDir + 'ERAint_an_' + str(year) + '_NA.nc'
    name_output    = InputOutputDir + 'ERAint_an_' + str(year) + '_NA_ForROMS.nc'

    anif = Dataset( name_input, 'r')

    if os.path.isfile(name_output):
        os.remove(name_output)
    anof = Dataset( name_output, 'w', format='NETCDF4')

    # Create dimensions in output files with same sizes as in input file
    time = anof.createDimension('time', len(anif.dimensions['time']))
    lon  = anof.createDimension('lon', len(anif.dimensions['longitude']))
    lat  = anof.createDimension('lat', len(anif.dimensions['latitude']))

    # Create variables in output files
    times = anof.createVariable('time','i8',('time',))
    lons  = anof.createVariable('lon','f8',('lon',))
    lats  = anof.createVariable('lat','f8',('lat',))
    Uwind = anof.createVariable('Uwind','f8',('time','lat','lon',))
    Vwind = anof.createVariable('Vwind','f8',('time','lat','lon',))
    cloud = anof.createVariable('cloud','f8',('time','lat','lon',))
    Pair  = anof.createVariable('Pair','f8',('time','lat','lon',))
    Tair  = anof.createVariable('Tair','f8',('time','lat','lon',))
    Qair  = anof.createVariable('Qair','f8',('time','lat','lon',))

    # Copy or modify values from input files
    times[:]     = anif.variables['time'] 
    times[:]     = times[:]*60*60           # Convert to seconds
    base_era     = datetime(1900, 1, 1)     # Change time reference
    base_target  = datetime(1948, 1, 1)
    times[:]     = times[:] - (base_target - base_era).total_seconds()
    lons[:]      = anif.variables['longitude']
    lats[:]      = anif.variables['latitude'][::-1]     # Reverse the latitude array
    Uwind[:,:,:] = anif.variables['u10'][:,::-1,:]
    Vwind[:,:,:] = anif.variables['v10'][:,::-1,:]
    cloud[:,:,:] = anif.variables['tcc'][:,::-1,:]
    Pair[:,:,:]  = anif.variables['msl'][:,::-1,:]*0.01    # Convert from Pa to mbar
    t2m          = anif.variables['t2m'][:,::-1,:]
    Tair[:,:,:]  = t2m[:,:,:]-273.15                    # Concert from Kelvin to Celsius
    d2m          = anif.variables['d2m'][:,::-1,:]
    #Old formula: Qair[:,:,:]  = 100((112 - 0.1*t2m + d2m)/(112 + 0.9*t2m))**8
    ## Compute Qair from t2m and d2m using eq. 12 of Lawrence (BAMS, 2005).
    L           = 2.5*10**6  # Enthalpy of vaporization at 273.15K, (J/kg)
    Rw          = 461.5      # Gaz constant for water vapor (J/K/kg)
    Qair[:,:,:] = 100*np.exp(-L*(t2m[:,:,:]-d2m[:,:,:])/(Rw*t2m[:,:,:]*d2m[:,:,:]))

    # Attributes
    anof.history    = 'Created ' + timemod.ctime(timemod.time())
    times.units     = 'seconds since 1948-01-01 00:00:0.0'
    times.long_name = 'time'
    times.calendar  = 'gregorian'
    lons.units      = 'degrees_north'
    lons.long_name  = 'Longitude'
    lats.units      = 'degrees_east'
    lats.long_name  = 'Latitude'
    Uwind.units     = 'm/s'
    Uwind.long_name = 'Zonal wind at 10 meters'
    Uwind.coordinates = 'lon lat'
    Uwind.time      = 'time'
    Vwind.units     = 'm/s'
    Vwind.long_name = 'Meridional wind at 10 meters'
    Vwind.coordinates = 'lon lat'
    Vwind.time      = 'time'
    cloud.units     = '(0 - 1)'
    cloud.long_name = 'Cloud area fraction'
    cloud.coordinates = 'lon lat'
    cloud.time      = 'time'
    Pair.units      = 'mbar'
    Pair.long_name  = 'Mean sea level pressure'
    Pair.coordinates = 'lon lat'
    Pair.time       = 'time'
    Tair.units      = 'C'
    Tair.long_name  = '2m temperature'
    Tair.coordinates = 'lon lat'
    Tair.time       = 'time'
    Qair.units      = 'Percentage'
    Qair.long_name  = 'Surface air relative humidity'
    Qair.coordinates = 'lon lat'
    Qair.time      = 'time'

    print np.amax(Uwind)
    print np.amin(Uwind)

    anof.close()
    anif.close()
