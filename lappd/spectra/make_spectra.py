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

data_name     = '160121_102056_Ch1'
data_ped_name = '160121_102056_Ch1'
n_points = 1250
window = 100e-9

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

filterTimeRange = False
if filterTimeRange:
    data = data[(np.abs(data.voltage)<50) & (data.time > 0) & (data.time < 20)]
    data_ped = data_ped[(np.abs(data_ped.voltage)<50) & (data.time > 0) & (data.time < 20)]
else:
    data = data[(np.abs(data.voltage)<50)]
    data_ped = data_ped[(np.abs(data_ped.voltage)<50)]

# Group the data into events (i.e., separate triggers)
grouped_data = data.groupby(['eventID'])
grouped_data_ped = data.groupby(['eventID'])

# Plot the time position and voltage of the max voltage in each event
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey = True)
max_voltages = data.loc[grouped_data.voltage.idxmin()]
max_voltages.plot(kind='scatter',x='time',y='voltage', ax = axes[0])
max_voltages_ped = data_ped.loc[grouped_data_ped.voltage.idxmin()]
max_voltages_ped.plot(kind='scatter',x='time',y='voltage', ax = axes[1])
fig.savefig('max_voltage_vs_time.png')

# Convert the voltage into a collected charge and sum over all voltages in the time window
scale = -dt/resistance*1e12/1e3 #for picoColoumbs and to put voltage back in V
q     = scale*grouped_data    .voltage.sum()
q_ped = scale*grouped_data_ped.voltage.sum()

print q

# Plot the spectrum of collected charge
loC =  5.
hiC =  8.
nBins = 260
width = float(hiC-loC)/nBins
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
q.plot(kind='hist', bins = np.arange(loC, hiC + width, width), logy = True, color='r', alpha = 0.5)
axes.set_xlabel("charge [pC]")
axes.set_ylabel("Entries / (%0.2f pC)" % width)
axes.set_xlim(loC, hiC)
fig.savefig('charge_spectrum.png')

# Now show the oscillioscope traces of a few events
interesting = []
interesting2 = []
interesting3 = []

# First get the event IDs of the events to look at
for eventID, group in grouped_data:
    print_progress(eventID, n_events)
    '''
    # Find the peak and define a window around it for the integration
    # Watch-out for the range going out of bounds for the event
    lo =  eventID   *n_points + low_offset
    hi = (eventID+1)*n_points - high_offset
    i = group.voltage.idxmin()
    lower_bound_id = i - 4*rise_time_in_samples
    upper_bound_id = i + 4*fall_time_in_samples
    lower_bound_id = lower_bound_id if lower_bound_id > lo else lo
    upper_bound_id = upper_bound_id if upper_bound_id < hi else hi
    '''
    q_subset = q[eventID]
    if q_subset > loC and q_subset < loC + 0.5:
        interesting.append(eventID)
    if q_subset > loC + 0.5 and q_subset < loC + 1.:
        interesting2.append(eventID)
    if q_subset > loC + 1. and q_subset < hiC:
        interesting3.append(eventID)

# Plot some interesting events
if len(interesting) > 0 and len(interesting2) > 0:
  fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(20, 10))
  from numpy.random import randint
  for i in range(4):
        k  = randint(0, len(interesting))
        k2 = randint(0, len(interesting2))
        k3 = randint(0, len(interesting3))
        grouped_data.get_group(interesting [k ]).plot(x='time',y='voltage'         ,ax=axes[0,i], legend=False)
        #grouped_data.get_group(interesting [k ]).plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False)
        grouped_data.get_group(interesting2[k2]).plot(x='time',y='voltage'         ,ax=axes[1,i], legend=False)
        #grouped_data.get_group(interesting2[k2]).plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False)
        grouped_data.get_group(interesting3[k3]).plot(x='time',y='voltage'         ,ax=axes[2,i], legend=False)
        #grouped_data.get_group(interesting3[k3]).plot(x='time',y='filtered_voltage',ax=axes[2,i], legend=False)
        '''
        # This plots the points defining the integration region
        data[bounds[interesting [k ]][0]:bounds[interesting [k ]][0]+1].plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False, style='o')
        data[bounds[interesting [k ]][1]:bounds[interesting [k ]][1]+1].plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False, style='o')
        data[bounds[interesting2[k2]][0]:bounds[interesting2[k2]][0]+1].plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False, style='o')
        data[bounds[interesting2[k2]][1]:bounds[interesting2[k2]][1]+1].plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False, style='o')
        data[bounds[interesting3[k3]][0]:bounds[interesting3[k3]][0]+1].plot(x='time',y='filtered_voltage',ax=axes[2,i], legend=False, style='o')
        data[bounds[interesting3[k3]][1]:bounds[interesting3[k3]][1]+1].plot(x='time',y='filtered_voltage',ax=axes[2,i], legend=False, style='o')
        '''
        axes[0,i].set_xlabel("")
        axes[2,i].set_xlabel("")
        axes[2,i].set_xlabel("time [ns]")
        axes[0,0].set_ylabel("voltage [mV]")
        axes[1,0].set_ylabel("voltage [mV]")
        axes[2,0].set_ylabel("voltage [mV]")
        k += 1
fig.savefig('oscilloscope_traces.png')

# Integrate the charge spectrum above some threshold to compute the gain
tot = 0.
nEventsAboveThreshold = 0
threshold = loC + 0.5 # This threshold will have to change depending on the spectrum
for Q in q:
    if Q > threshold:
        tot += Q
        nEventsAboveThreshold += 1
if nEventsAboveThreshold > 0:
    gain = tot*1.e-12/1.602e-19/nEventsAboveThreshold/1e7
    print "\nTotal charge collected above %0.2f pC threshold = %0.2f pC" % (threshold, tot)
    print "Corresponding to a gain = %0.2f x 10^7" % gain
else:
    print "No events above threshold; cannot compute gain."


