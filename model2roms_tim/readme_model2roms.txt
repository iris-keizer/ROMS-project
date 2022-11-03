How to operate model2roms:

1) use configM2R.py to set your paths, variables, settings, etc.
2) you might need to change other .py scripts of the package as well, i.e. forcingNames.py, various scripts to change the time units (model2roms.py,grd.py, see e-mail with Trond Kristiansen)
3) set up a virtual environment (or in base) with conda and install the required packages (see trondkristiansen.com/model2roms GITHUB)
with:
source activate model2roms (if it doesn't work, activate it again)
4) then run runM2R.py
5) need at least two timesteps as input