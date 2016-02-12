#!/usr/bin/env ipython
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib
matplotlib.style.use('ggplot')

def print_progress(count, n_events):
    percent = float(count) / n_events
    hashes = '#' * int(round(percent * 20))
    spaces = ' ' * (20 - len(hashes))
    sys.stdout.write("\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()

data_name_l = '160212_161940_Ch1'
data_name_r = '160212_161940_Ch4'
n_points = 1000
window = 40e-9


data_l       = pd.read_pickle(data_name_l+'.pkl')
data_r       = pd.read_pickle(data_name_r+'.pkl')

n_events = len(data_l)/n_points

charge_subset = []
resistance = 50
dt = (data_l['time'][1] - data_l['time'][0])/1e9 # to get back to seconds
# V = IR => I = V/R; Q = int_t0^t1 I dt = int_t0^t1 V/R dt => Q = sum(V/R *Delta t)

print "time interval:", dt
print "number of events:", n_events

filterTimeRange = True
if filterTimeRange:
    data_l = data_l[(data_l.time > 0) & (data_l.time < 50)]
    data_r = data_r[(data_r.time > 0) & (data_r.time < 50)]
else:
    data_l = data_l[(np.abs(data_l.voltage)<50)]
    data_r = data_r[(np.abs(data_r.voltage)<50)]

# Group the data into events (i.e., separate triggers)
grouped_data_l = data_l.groupby(['eventID'])
grouped_data_r = data_r.groupby(['eventID'])

# Plot the time position and voltage of the max voltage in each event
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey = True)
max_voltages_l = data_l.loc[grouped_data_l.voltage.idxmin()]
max_voltages_l.plot(kind='scatter',x='time',y='voltage', title = 'Channel 1', ax = axes[0])
max_voltages_r = data_r.loc[grouped_data_r.voltage.idxmin()]
max_voltages_r.plot(kind='scatter',x='time',y='voltage', title = 'Channel 4', ax = axes[1])
fig.savefig('max_voltage_vs_time.png')

# Filter data to remove background
max_voltages_l = max_voltages_l[(max_voltages_l.voltage<-3.9)]
max_voltages_r = max_voltages_r[(max_voltages_r.voltage<-3.9)]

max_voltages_l = max_voltages_l.set_index('eventID')
max_voltages_r = max_voltages_r.set_index('eventID')

time_diff = max_voltages_l.time.subtract( max_voltages_r.time ).dropna()
print "Number of events after cuts", len(time_diff)
print "Std dev of time difference %0.3f ns" % (time_diff.std())

from iminuit import Minuit, describe
from iminuit.util import make_func_code
class LogLikelihood:
    """Need this for MINUIT. We can pass in any function
    and data we want using this class."""
    def __init__(self, f, data):
        self.f = f
        self.data = data
        f_sig = describe(f)
        # this is how you fake function
        # signature dynamically
        self.func_code = make_func_code(f_sig[1:]) # docking off independent variable
        self.func_defaults = None # this keeps np.vectorize happy
    def __call__(self, *args): # lets try to find mu and sigma
        NLL = -np.log( self.f(self.data, *args)).sum()
        return NLL

def LLwrapper(params):
    """Need this for scipy.optimize. There might be a better way of doing it."""
    NLL = LogLikelihood(gauss, s)
    return NLL(params[0], params[1])

def gauss(x, mu, sigma):
    return 1./(np.sqrt(2 * np.pi*sigma**2)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )

NLL = LogLikelihood(gauss, time_diff.values)
m = Minuit(NLL, mu = -14., sigma = 1., limit_mu=(-16., -10.), limit_sigma=(0.05, 2.), print_level=1, errordef=0.5)
result = m.migrad() # hesse called at the end of migrad
print m.values
print m.errors

max_time = -10.
min_time = -20.
nbins = 50
width = float(max_time - min_time)/nbins
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
time_diff.plot(kind='hist', bins = np.arange(min_time, max_time, width), ax = axes)
axes.set_xlabel("Time difference [ns]")
axes.set_ylabel("Entries / (%0.2f ns)" % width)
fig.savefig('time_diff.png')
