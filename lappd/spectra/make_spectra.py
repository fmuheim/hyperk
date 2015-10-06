import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
matplotlib.style.use('ggplot')


def filter_signal(data, nSamples = 1000, frameSize = 40e-9):
    '''Butterworth filter of signal.'''
    fs = float(nSamples)/frameSize # sample rate (1000 samples in 40ns)
    nyq = fs*0.5 # Nyquist frequency
    high = 900e6/nyq # high frequency cut-off
    b, a = signal.butter(2, high, 'low', analog = False)
    y = signal.lfilter(b, a, data)
    return y

# Conversion factors when reading in data
cfun = lambda x: float(x)/1e6
cfun2 = lambda x: float(x)*1e3
cfun3 = lambda x: float(x)/1e-9

'''
#HV 2.2kV
#LED 2.1V
data_gate      = pd.read_csv('151001_113045_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun})
data           = pd.read_csv('151001_113045_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4])
data_gate.append(pd.read_csv('151001_113709_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun}))
data     .append(pd.read_csv('151001_113709_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4]))
'''
'''
#HV 2.2kV
#LED 2.1V
data_gate      = pd.read_csv('151001_141206_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun})
data           = pd.read_csv('151001_141206_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4])
data_gate.append(pd.read_csv('151001_140919_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun}))
data     .append(pd.read_csv('151001_140919_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4]))
'''
'''
#HV 2.2kV
#LED 0V
data_gate_ped      = pd.read_csv('151001_151307_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun})
data_ped           = pd.read_csv('151001_151307_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4])
'''

'''
#HV 2.2kV
#LED 3.0V # definitely not in the single photon regime here
data_gate      = pd.read_csv('151001_151716_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun})
data           = pd.read_csv('151001_151716_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4])
'''

#HV 2.3kV
#LED 2.1V
#data_gate      = pd.read_csv('151001_152045_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun})
#data           = pd.read_csv('151001_152045_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4])

#HV 2.3kV
#LED 1.5V
#data_gate      = pd.read_csv('151001_154736_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun, 'time':cfun3})
#data           = pd.read_csv('151001_154736_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

#HV 2.3kV
#LED 1.0V # very few photons here
#data_gate      = pd.read_csv('151001_155054_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun, 'time':cfun3})
#data           = pd.read_csv('151001_155054_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

#HV 2.3kV
#LED 0.5V # essentially no photons here
#data_gate      = pd.read_csv('151001_155608_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun, 'time':cfun3})
#data           = pd.read_csv('151001_155608_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

#HV 2.3kV
#LED 0V
#data_gate_ped      = pd.read_csv('151001_152343_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun, 'time':cfun3})
#data_ped           = pd.read_csv('151001_152343_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

#HV 2.4kV
#LED 1.5V
#data_gate      = pd.read_csv('151001_163215_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun, 'time':cfun3})
#data           = pd.read_csv('151001_163215_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

#HV 2.4kV
#LED 0.5V # no light here it appears
#data_gate      = pd.read_csv('151001_165917_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun, 'time':cfun3})
#data           = pd.read_csv('151001_165917_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

#HV 2.4kV
#LED 0.0V
#data_gate_ped      = pd.read_csv('151001_170221_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun, 'time':cfun3})
#data_ped           = pd.read_csv('151001_170221_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

# Now seeing the doubling in dark current to (15 microAmps) when I am at 2.5kV.

#HV 2.5kV
#LED 1.5V
#data_gate      = pd.read_csv('151001_163547_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun})
#data           = pd.read_csv('151001_163547_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4])

#HV 2.5kV
#LED 0.5V
#data_gate      = pd.read_csv('151001_165231_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun})
#data           = pd.read_csv('151001_165231_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4])

#HV 2.5kV
#LED 0V
#data_gate_ped      = pd.read_csv('151001_163824_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun, 'time':cfun3})
#data_ped           = pd.read_csv('151001_163824_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

# I saw the doubling in dark current to (7->18 microAmps) when I went up to 2.65kV.

#HV 2.55kV
#LED 1.5V
#data_gate      = pd.read_csv('151001_173645_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun, 'time':cfun3})
#data           = pd.read_csv('151001_173645_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

#HV 2.59kV
#LED 1.5V
#J11
#data_gate      = pd.read_csv('151002_174018_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun, 'time':cfun3})
#data            = pd.read_csv('151002_174018_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
#data           = pd.read_csv('151005_102138_Ch4.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
#data_ped       = pd.read_csv('151005_103908_Ch4.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

#J7
#data           = pd.read_csv('151005_102747_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
#data_ped       = pd.read_csv('151005_104445_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

#J13
#data           = pd.read_csv('151005_103115_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
#data_ped       = pd.read_csv('151005_103545_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

#HV 2.55kV
#LED 1.0V
#J7
# Using 400ns frame window and resolution of 10000
#data           = pd.read_csv('151005_161059_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
#data_ped       = pd.read_csv('151005_161543_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

# Using 100ns frame window and resolution of 2500
data                  = pd.read_csv('151006_091754.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
data = data.append(     pd.read_csv('151006_094857.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3}), ignore_index = True)
data = data.append(     pd.read_csv('151006_095026.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3}), ignore_index = True)
data = data.append(     pd.read_csv('151006_095159.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3}), ignore_index = True)
data = data.append(     pd.read_csv('151006_095321.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3}), ignore_index = True)
data = data.append(     pd.read_csv('151006_104103.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3}), ignore_index = True)
data = data.append(     pd.read_csv('151006_104235.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3}), ignore_index = True)
data = data.append(     pd.read_csv('151006_104431.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3}), ignore_index = True)
data = data.append(     pd.read_csv('151006_104607.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3}), ignore_index = True)
data = data.append(     pd.read_csv('151006_104730.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3}), ignore_index = True)
data_ped       = pd.read_csv('151006_092037.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

# Using nominal 40ns window
#data           = pd.read_csv('151005_141912_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})


'''
HV 2.2kV
LED 2.1V
different resolution used
data_gate      = pd.read_csv('151001_141511_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun})
data           = pd.read_csv('151001_141511_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4])
'''

n_points = 2500
window = 100e-9
n_events = len(data)/n_points
n_events_ped = len(data_ped)/n_points

weights = []
charge = []
charge_subset = []
charge_ped = []
resistance = 50
dt = (data['time'][1] - data['time'][0])/1e9 # to get back to seconds
# V = IR => I = V/R; Q = int_t0^t1 I dt = int_t0^t1 V/R dt => Q = sum(V/R *Delta t)

print "time interval:", dt
print "number of events:", n_events

interesting = []
interesting2 = []

low_offset = 0
high_offset = 0


filtered_voltages = []
filtered_voltages_ped = []

# First filter the voltages
for i in range(n_events):
    d = filter_signal(data['voltage'][i*n_points+low_offset:(i+1)*n_points-high_offset].as_matrix(), n_points, window)
    filtered_voltages.extend(d)
for i in range(n_events_ped):
    d = filter_signal(data_ped['voltage'][i*n_points+low_offset:(i+1)*n_points-high_offset].as_matrix(), n_points, window)
    filtered_voltages_ped.extend(d)
    weights.append(5)

# Add a new column to the DataFrame
data['filtered_voltage'] = pd.Series(filtered_voltages, index = data.index)
data_ped['filtered_voltage'] = pd.Series(filtered_voltages_ped, index = data_ped.index)

'''
# Find the 10% and 90% points for the rise and fall times
lower_bounds = []
upper_bounds = []
for i in range(n_events):
    peak_voltage_id = data[i*n_points+low_offset:(i+1)*n_points-high_offset].filtered_voltage.idxmin()
    peak_voltage    = data[i*n_points+low_offset:(i+1)*n_points-high_offset].filtered_voltage[peak_voltage_id]
    peak_time       = data[i*n_points+low_offset:(i+1)*n_points-high_offset].time[peak_voltage_id]
    #print i, peak_time, peak_voltage,data[i*n_points+low_offset:peak_voltage_id][data.voltage >= 0.1*peak_voltage]
    peak_time_10    = data[i*n_points+low_offset:peak_voltage_id][data.filtered_voltage >= 0.1*peak_voltage].time[-1:].iloc[0]
    peak_time_90    = data[i*n_points+low_offset:peak_voltage_id][data.filtered_voltage >= 0.9*peak_voltage].time[-1:].iloc[0]
    rise_time = peak_time_90 - peak_time_10
    lower_bound = peak_time - 4*rise_time
    upper_bound = peak_time + 4*rise_time
    lower_bounds.append(lower_bound)
    upper_bounds.append(upper_bound)
print lower_bounds
'''


# Let's make some assumptions about the rise-time
rise_time = 1e-9 #seconds
rise_time_in_samples = int(rise_time/dt)
fall_time = 5e-9 #seconds
fall_time_in_samples = int(fall_time/dt)
lower_bounds = []
upper_bounds = []

# Now compute the sum for each event to get the total charge collected
for i in range(n_events):
    scale = -dt/resistance*1e12/1e3 #for picoColoumbs and to put voltage back in V
    #q     = scale*data    [i*n_points+low_offset:(i+1)*n_points-high_offset][data.time>lower_bounds[i]][data.time<upper_bounds[i]].filtered_voltage.sum()
    lo = i    *n_points + low_offset
    hi = (i+1)*n_points - high_offset
    q  = scale*data[lo:hi].filtered_voltage.sum()
    charge.append(q)
    # Find the peak and define a window around it for the integration
    # Watch-out for the range going out of bounds for the event
    peak_voltage_id = data[lo:hi].filtered_voltage.idxmin()
    lower_bound_id = peak_voltage_id - 4*rise_time_in_samples
    upper_bound_id = peak_voltage_id + 4*fall_time_in_samples
    lower_bound_id = lower_bound_id if lower_bound_id > lo else lo
    upper_bound_id = upper_bound_id if upper_bound_id < hi else (hi - 1)
    q_subset  = scale*data[lower_bound_id:upper_bound_id].filtered_voltage.sum()
    charge_subset.append(q_subset)
    lower_bounds.append(lower_bound_id)
    upper_bounds.append(upper_bound_id)
    if q > 0. and q < 0.5:
        interesting.append(i)
    if q > 1. and q < 6.:
        interesting2.append(i)

# Add column that will tell us where the upper and lower bounds are for integration so we can plot them
data['lower'] = pd.Series(False, index = data.index)
data['upper'] = pd.Series(False, index = data.index)
for i in lower_bounds: data.loc[i, 'lower'] = True
for i in upper_bounds: data.loc[i, 'upper'] = True

for i in range(n_events_ped):
    scale = -dt/resistance*1e12/1e3 #for picoColoumbs and to put voltage back in V
    lo = i    *n_points + low_offset
    hi = (i+1)*n_points - high_offset
    q_ped = scale*data_ped[lo:hi].filtered_voltage.sum()
    charge_ped.append(q_ped)

# Plot some interesting events for examples
if len(interesting) > 0 and len(interesting2) > 0:
  fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(14, 10))
  from numpy.random import randint
  for i in range(4):
        k  = randint(0, len(interesting))
        k2 = randint(0, len(interesting2))
        data[interesting [k] *n_points+low_offset:(interesting [k] +1)*n_points-high_offset].plot(x='time',y='voltage'         ,ax=axes[0,i], legend=False)
        data[interesting [k] *n_points+low_offset:(interesting [k] +1)*n_points-high_offset].plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False)
        data[interesting2[k2]*n_points+low_offset:(interesting2[k2]+1)*n_points-high_offset].plot(x='time',y='voltage'         ,ax=axes[1,i], legend=False)
        data[interesting2[k2]*n_points+low_offset:(interesting2[k2]+1)*n_points-high_offset].plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False)
        # This plots the points defining the integration region
        data[interesting [k] *n_points+low_offset:(interesting [k] +1)*n_points-high_offset][data.lower == True].plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False, style='o')
        data[interesting [k] *n_points+low_offset:(interesting [k] +1)*n_points-high_offset][data.upper == True].plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False, style='o')
        data[interesting2[k2]*n_points+low_offset:(interesting2[k2]+1)*n_points-high_offset][data.lower == True].plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False, style='o')
        data[interesting2[k2]*n_points+low_offset:(interesting2[k2]+1)*n_points-high_offset][data.upper == True].plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False, style='o')
        axes[1,i].set_xlabel("time [ns]")
        axes[0,0].set_ylabel("voltage [mV]")
        axes[1,0].set_ylabel("voltage [mV]")
        k += 1

# Integrate the charge spectrum above some threshold to compute the gain
tot = 0.
nEventsAboveThreshold = 0
threshold = 1.
for q in charge:
    if q > threshold:
        tot += q
        nEventsAboveThreshold += 1
gain = tot*1.e-12/1.602e-19/nEventsAboveThreshold/1e7
print "Total charge collected above %0.2f pC threshold = %0.2f pC" % (threshold, tot)
print "Corresponding to a gain = %0.2f x 10^7" % gain

# Plot the charge spectrum
from scipy.stats import norm
import numpy as np
loC = 0.
hiC =  6.
nBins = 120
width = float(hiC-loC)/nBins
mu, std = norm.fit(charge_ped)
x = np.linspace(loC, hiC, nBins)
p = norm.pdf(x, mu, std)
print "Mean of pedestal = %0.2f pC; RMS =  %0.2f pC" % (np.mean(charge_ped), np.std(charge_ped))
print "Mean of pedestal (from fit) = %0.2f pC; Width (from fit) =  %0.2f pC" % (mu, std)

spectrum     = pd.DataFrame(charge, columns = ['voltage'])
spectrum_subset = pd.DataFrame(charge_subset, columns = ['voltage'])
spectrum_ped = pd.DataFrame(charge_ped, columns = ['voltage'])
ax = spectrum.plot(kind='hist', bins = np.arange(loC, hiC + width, width), logy = True)
spectrum_ped .plot(kind='hist', bins = np.arange(loC, hiC + width, width), logy = True, ax = ax, color='y', alpha = 0.5)
spectrum_subset.plot(kind='hist', bins = np.arange(loC, hiC + width, width), logy = True, ax = ax, color='b', alpha = 0.5)
ax.set_xlabel("charge [pC]")
ax.set_ylabel("Entries / (%0.2f pC)" % width)
ax.set_xlim(loC, hiC)
plt.show()


