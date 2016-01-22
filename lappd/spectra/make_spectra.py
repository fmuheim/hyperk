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

#HV 2.55kV
#LED 1.0V
#J7
# Using 100ns frame window and resolution of 2500
data_name     = '151006_091754'
data_ped_name = '151006_092037'
n_points = 2500
window = 100e-9

#HV 2.55kV
#LED 1.0V
#J7
# Using 40ns frame window and resolution of 1000
data_name     = '151009_155610'
data_ped_name = '151009_161113'
n_points = 1000
window = 40e-9

#HV 2.55kV
#LED 1.0V
#J11
# Using 40ns frame window and resolution of 1000
data_name     = '151009_171100_Ch1'
data_ped_name = '151009_171617_Ch1'
#J9 turns out to be broken
# Using 40ns frame window and resolution of 1000
#data_name     = '151009_171100_Ch3'
#data_ped_name = '151009_171617_Ch3'
#J13
# Using 40ns frame window and resolution of 1000
#data_name     = '151009_171100_Ch4'
#data_ped_name = '151009_171617_Ch4'
n_points = 1000
window = 40e-9

data_name     = '160122_101920_Ch1'
data_ped_name = '160122_101920_Ch1'
n_points = 1250
window = 100e-9


data_name     = '160122_104308_Ch1'
data_ped_name = '160122_104308_Ch1'
n_points = 10000
window = 400e-9

data_name     = '160122_105757_Ch1'
data_ped_name = '160122_105757_Ch1'
n_points = 5000
window = 200e-9


data_name     = '160122_111749_Ch1'
data_ped_name = '160122_111749_Ch1'
n_points = 2500
window = 100e-9

data_name     = '160122_120123_Ch1'
data_ped_name = '160122_120304_Ch1'
n_points = 1000
window = 40e-9


data           = pd.read_pickle(data_name+'.pkl')
data_ped       = pd.read_pickle(data_ped_name+'.pkl')

n_events = len(data)/n_points
n_events_ped = len(data_ped)/n_points

charge_subset = []
resistance = 50
dt = (data['time'][1] - data['time'][0])/1e9 # to get back to seconds
# V = IR => I = V/R; Q = int_t0^t1 I dt = int_t0^t1 V/R dt => Q = sum(V/R *Delta t)

print "time interval:", dt
print "number of events:", n_events

# This part would only be used if we wanted to define some range of integration (by default we just take everything)
low_offset = 0
high_offset = 0
# Let's make some assumptions about the rise-time
rise_time = 1e-9 #seconds
rise_time_in_samples = int(rise_time/dt)
fall_time = 4e-9 #seconds
fall_time_in_samples = int(fall_time/dt)

filterTimeRange = True
if filterTimeRange:
    data = data[(np.abs(data.voltage)<90) & (data.time > 0) & (data.time < 50)]
    data_ped = data_ped[(np.abs(data_ped.voltage)<90) & (data_ped.time > 0) & (data_ped.time < 50)]
else:
    data = data[(np.abs(data.voltage)<50)]
    data_ped = data_ped[(np.abs(data_ped.voltage)<50)]

# Group the data into events (i.e., separate triggers)
grouped_data = data.groupby(['eventID'])
grouped_data_ped = data_ped.groupby(['eventID'])

# Plot the time position and voltage of the max voltage in each event
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey = True)
max_voltages = data.loc[grouped_data.voltage.idxmin()]
max_voltages.plot(kind='scatter',x='time',y='voltage', title = 'LED on', ax = axes[0])
max_voltages_ped = data_ped.loc[grouped_data_ped.voltage.idxmin()]
max_voltages_ped.plot(kind='scatter',x='time',y='voltage', title = 'LED off', ax = axes[1])
fig.savefig('max_voltage_vs_time.png')

# Convert the voltage into a collected charge and sum over all voltages in the time window
scale = -dt/resistance*1e12/1e3 #for picoColoumbs and to put voltage back in V
q     = scale*grouped_data    .voltage.sum()
q_ped = scale*grouped_data_ped.voltage.sum()

# Plot the spectrum of collected charge
loC =  1.
hiC =  6.
nBins = 250
width = float(hiC-loC)/nBins
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
q.plot(kind='hist', title = 'LED on', bins = np.arange(loC, hiC + width, width), logy = True, color='r', alpha = 0.5, ax = axes[0])
q_ped.plot(kind='hist', title = 'LED off', bins = np.arange(loC, hiC + width, width), logy = True, color='b', alpha = 0.5, ax = axes[1])

from scipy.stats import norm
mu, std = norm.fit(q_ped.values)
mu_ped  = q_ped.mean()
std_ped = q_ped.std()
print
print "Fitted mean and standard deviation of the pedestal    : %0.3f, %0.3f  " % (mu, std)
print "Calculated mean and standard deviation of the pedestal: %0.3f, %0.3f\n" % (mu_ped, std_ped)

# uncomment these lines if you want to plot the fitted Gaussian
#x = np.linspace(loC, hiC, nBins)
#pdf = n_events*norm.pdf(x, mu, std)/nBins
#plt.plot(x, pdf, 'k', linewidth = 2)

axes[0].set_xlabel("charge [pC]")
axes[0].set_ylabel("Entries / (%0.2f pC)" % width)
axes[0].set_xlim(loC, hiC)
axes[0].set_ylim(1, 1e4)
axes[1].set_xlabel("charge [pC]")
axes[1].set_ylabel("Entries / (%0.2f pC)" % width)
axes[1].set_xlim(loC, hiC)
axes[1].set_ylim(1, 1e4)
fig.savefig('charge_spectrum.png')

# Compute the mean of the collected charge above some threshold (which is defined in terms of the pedestal) to compute the gain
threshold = mu_ped + 3*std_ped
signal = q[q > threshold]
mean_signal_charge = signal.mean()
gain = mean_signal_charge*1.e-12/1.602e-19/1e7
print "Threshold set at %0.3f pC" % threshold
print "Gain of the photo-sensor is %0.2f x 10^7\n" % gain

# Now show the oscillioscope traces of a few events
interesting = []
interesting2 = []
interesting3 = []

subset1 = (q[(q > mu_ped      ) & (q < mu_ped + 0.5)]).sample(n=4).index
subset2 = (q[(q > mu_ped + 0.5) & (q < mu_ped + 1.0)]).sample(n=4).index
subset3 = (q[(q > mu_ped + 1.0) & (q < hiC         )]).sample(n=4).index

# Plot some interesting events
if len(subset1) > 0 and len(subset2) > 0 and len(subset3) > 0:
  trace, trace_ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(20, 10))
  for i in range(4):
        grouped_data.get_group(subset1[i]).plot(x='time',y='voltage'         ,ax=trace_ax[0,i], legend=False)
        grouped_data.get_group(subset1[i]).plot(x='time',y='filtered_voltage',ax=trace_ax[0,i], legend=False)
        grouped_data.get_group(subset2[i]).plot(x='time',y='voltage'         ,ax=trace_ax[1,i], legend=False)
        grouped_data.get_group(subset2[i]).plot(x='time',y='filtered_voltage',ax=trace_ax[1,i], legend=False)
        grouped_data.get_group(subset3[i]).plot(x='time',y='voltage'         ,ax=trace_ax[2,i], legend=False)
        grouped_data.get_group(subset3[i]).plot(x='time',y='filtered_voltage',ax=trace_ax[2,i], legend=False)
        '''
        # This plots the points defining the integration region
        data[bounds[interesting [k ]][0]:bounds[interesting [k ]][0]+1].plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False, style='o')
        data[bounds[interesting [k ]][1]:bounds[interesting [k ]][1]+1].plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False, style='o')
        data[bounds[interesting2[k2]][0]:bounds[interesting2[k2]][0]+1].plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False, style='o')
        data[bounds[interesting2[k2]][1]:bounds[interesting2[k2]][1]+1].plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False, style='o')
        data[bounds[interesting3[k3]][0]:bounds[interesting3[k3]][0]+1].plot(x='time',y='filtered_voltage',ax=axes[2,i], legend=False, style='o')
        data[bounds[interesting3[k3]][1]:bounds[interesting3[k3]][1]+1].plot(x='time',y='filtered_voltage',ax=axes[2,i], legend=False, style='o')
        '''
        trace_ax[0,i].set_xlabel("")
        trace_ax[1,i].set_xlabel("")
        trace_ax[2,i].set_xlabel("time [ns]")
        trace_ax[0,0].set_ylabel("voltage [mV]")
        trace_ax[1,0].set_ylabel("voltage [mV]")
        trace_ax[2,0].set_ylabel("voltage [mV]")
  trace.savefig('oscilloscope_traces.png')
