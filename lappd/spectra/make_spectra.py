import pandas as pd
import matplotlib.pyplot as plt
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
# Using 400ns frame window and resolution of 10000
#data           = pd.read_csv('151005_161059_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
#data_ped       = pd.read_csv('151005_161543_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

# Using 100ns frame window and resolution of 2500
data           = pd.read_pickle('151006_091754.pkl')
data_ped       = pd.read_pickle('151006_092037.pkl')

# Using nominal 40ns window
#data           = pd.read_csv('151005_141912_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

n_points = 2500
window = 100e-9
n_events = len(data)/n_points
n_events_ped = len(data_ped)/n_points

charge_subset = []
resistance = 50
dt = (data['time'][1] - data['time'][0])/1e9 # to get back to seconds
# V = IR => I = V/R; Q = int_t0^t1 I dt = int_t0^t1 V/R dt => Q = sum(V/R *Delta t)

print "time interval:", dt
print "number of events:", n_events

interesting = []
interesting2 = []
interesting3 = []

low_offset = 0
high_offset = 0
# Let's make some assumptions about the rise-time
rise_time = 1e-9 #seconds
rise_time_in_samples = int(rise_time/dt)
fall_time = 4e-9 #seconds
fall_time_in_samples = int(fall_time/dt)

grouped_data = data.groupby(['eventID'])

scale = -dt/resistance*1e12/1e3 #for picoColoumbs and to put voltage back in V
q     = scale*grouped_data                 .filtered_voltage.sum()
q_ped = scale*data_ped.groupby(['eventID']).filtered_voltage.sum()

bounds = []
for eventID, group in grouped_data:
    print_progress(eventID, n_events)
    # Find the peak and define a window around it for the integration
    # Watch-out for the range going out of bounds for the event
    '''
    lo =  eventID   *n_points + low_offset
    hi = (eventID+1)*n_points - high_offset
    i = group.voltage.idxmin()
    lower_bound_id = i - 4*rise_time_in_samples
    upper_bound_id = i + 4*fall_time_in_samples
    lower_bound_id = lower_bound_id if lower_bound_id > lo else lo
    upper_bound_id = upper_bound_id if upper_bound_id < hi else hi
    '''
    # If we want to always use the same window
    # Need to adjust the window for the file since the lower time bound is not the same
    lo_window = 341 if eventID < n_events/10 else 211
    hi_window = 716 if eventID < n_events/10 else 586
    lower_bound_id = eventID*n_points + lo_window
    upper_bound_id = eventID*n_points + hi_window
    q_subset  = scale*data.filtered_voltage.iloc[lower_bound_id:upper_bound_id].sum()
    charge_subset.append(q_subset)
    bounds.append([lower_bound_id, upper_bound_id])
    if q_subset > 0. and q_subset < 0.5:
        interesting.append(eventID)
    if q_subset > 1. and q_subset < 2.:
        interesting2.append(eventID)
    if q_subset > 2. and q_subset < 6.:
        interesting3.append(eventID)

# Plot some interesting events for examples
if len(interesting) > 0 and len(interesting2) > 0:
  fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(20, 10))
  from numpy.random import randint
  for i in range(4):
        k  = randint(0, len(interesting))
        k2 = randint(0, len(interesting2))
        k3 = randint(0, len(interesting3))
        grouped_data.get_group(interesting [k ]).plot(x='time',y='voltage'         ,ax=axes[0,i], legend=False)
        grouped_data.get_group(interesting [k ]).plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False)
        grouped_data.get_group(interesting2[k2]).plot(x='time',y='voltage'         ,ax=axes[1,i], legend=False)
        grouped_data.get_group(interesting2[k2]).plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False)
        grouped_data.get_group(interesting3[k3]).plot(x='time',y='voltage'         ,ax=axes[2,i], legend=False)
        grouped_data.get_group(interesting3[k3]).plot(x='time',y='filtered_voltage',ax=axes[2,i], legend=False)
        # This plots the points defining the integration region
        data[bounds[interesting [k ]][0]:bounds[interesting [k ]][0]+1].plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False, style='o')
        data[bounds[interesting [k ]][1]:bounds[interesting [k ]][1]+1].plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False, style='o')
        data[bounds[interesting2[k2]][0]:bounds[interesting2[k2]][0]+1].plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False, style='o')
        data[bounds[interesting2[k2]][1]:bounds[interesting2[k2]][1]+1].plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False, style='o')
        data[bounds[interesting3[k3]][0]:bounds[interesting3[k3]][0]+1].plot(x='time',y='filtered_voltage',ax=axes[2,i], legend=False, style='o')
        data[bounds[interesting3[k3]][1]:bounds[interesting3[k3]][1]+1].plot(x='time',y='filtered_voltage',ax=axes[2,i], legend=False, style='o')
        axes[1,i].set_xlabel("time [ns]")
        axes[0,0].set_ylabel("voltage [mV]")
        axes[1,0].set_ylabel("voltage [mV]")
        axes[2,0].set_ylabel("voltage [mV]")
        k += 1
plt.show()

# Integrate the charge spectrum above some threshold to compute the gain
tot = 0.
nEventsAboveThreshold = 0
threshold = 0.5
for Q in charge_subset:
    if Q > threshold:
        tot += Q
        nEventsAboveThreshold += 1
gain = tot*1.e-12/1.602e-19/nEventsAboveThreshold/1e7
print "Total charge collected above %0.2f pC threshold = %0.2f pC" % (threshold, tot)
print "Corresponding to a gain = %0.2f x 10^7" % gain

# Plot the charge spectrum
import numpy as np
loC = 0.
hiC =  6.
nBins = 120
width = float(hiC-loC)/nBins
print "Mean of pedestal = %0.2f pC; RMS =  %0.2f pC" % (q_ped.mean(), q_ped.std())

#ax = q.plot(kind='hist', bins = np.arange(loC, hiC + width, width), logy = True,          color='b', alpha = 0.5)
#q_ped .plot(kind='hist', bins = np.arange(loC, hiC + width, width), logy = True, ax = ax, color='y', alpha = 0.5)
spectrum_subset = pd.DataFrame(charge_subset, columns = ['voltage'])
ax = spectrum_subset.plot(kind='hist', bins = np.arange(loC, hiC + width, width), logy = True, color='r', alpha = 0.5)
ax.set_xlabel("charge [pC]")
ax.set_ylabel("Entries / (%0.2f pC)" % width)
ax.set_xlim(loC, hiC)
plt.show()

max_voltages = data.loc[grouped_data.filtered_voltage.idxmin()]
max_voltages.plot(kind='scatter',x='time',y='filtered_voltage')
plt.show()
