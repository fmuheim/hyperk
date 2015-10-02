import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
matplotlib.style.use('ggplot')


def filter_signal(data):
    '''Butterworth filter of signal.'''
    fs = 1000/40e-9 # sample rate (1000 samples in 40ns)
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
data_gate      = pd.read_csv('151001_163215_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun, 'time':cfun3})
data           = pd.read_csv('151001_163215_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

#HV 2.4kV
#LED 0.5V # no light here it appears
#data_gate      = pd.read_csv('151001_165917_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun, 'time':cfun3})
#data           = pd.read_csv('151001_165917_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

#HV 2.4kV
#LED 0.0V
data_gate_ped      = pd.read_csv('151001_170221_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun, 'time':cfun3})
data_ped           = pd.read_csv('151001_170221_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

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
#data_gate_ped      = pd.read_csv('151001_163824_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun})
#data_ped           = pd.read_csv('151001_163824_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4])

# I saw the doubling in dark current to (7->18 microAmps) when I went up to 2.65kV.

'''
HV 2.2kV
LED 2.1V
different resolution used
data_gate      = pd.read_csv('151001_141511_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun})
data           = pd.read_csv('151001_141511_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4])
'''

n_points = 1000
n_events = len(data)/n_points

charge = []
charge_fil = []
charge_ped = []
resistance = 50
dt = (data['time'][1] - data['time'][0])/1e9 # to get back to seconds
# V = IR => I = V/R; Q = int_t0^t1 I dt = int_t0^t1 V/R dt => Q = sum(V/R *Delta t)

print "time interval:", dt

interesting = []
interesting2 = []

low_offset = 0
high_offset = 0

filtered_voltages = []

# First filter the voltages
for i in range(n_events):
    d = filter_signal(data['voltage'][i*n_points+low_offset:(i+1)*n_points-high_offset].as_matrix())
    filtered_voltages.extend(d)

# Add a new column to the DataFrame
data['filtered_voltage'] = pd.Series(filtered_voltages, index = data.index)

# Now compute the sum
for i in range(n_events):
    scale = -dt/resistance*1e12/1e3 #for picoColoumbs and to put voltage back in V
    q     = scale*data    [i*n_points+low_offset:(i+1)*n_points-high_offset].sum()['voltage']
    q_fil = scale*data    [i*n_points+low_offset:(i+1)*n_points-high_offset].sum()['filtered_voltage']
    q_ped = scale*data_ped[i*n_points+low_offset:(i+1)*n_points-high_offset].sum()['voltage']
    charge    .append(q)
    charge_fil.append(q_fil)
    charge_ped.append(q_ped)
    if q > 1.:
        interesting.append(i)
    if q > 0.5 and q < 1.:
        interesting2.append(i)

fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(14, 10))
from numpy.random import randint
for i in range(4):
        k = randint(0, len(interesting))
        k2 = randint(0, len(interesting2))
        data[interesting[k]*1000+low_offset:(interesting[k]+1)*1000-high_offset].plot(x='time',y='voltage',ax=axes[0,i], legend=False)
        data[interesting2[k2]*1000+low_offset:(interesting2[k2]+1)*1000-high_offset].plot(x='time',y='voltage',ax=axes[1,i], legend=False)
        data[interesting[k]*1000+low_offset:(interesting[k]+1)*1000-high_offset].plot(x='time',y='filtered_voltage',ax=axes[0,i], legend=False)
        data[interesting2[k2]*1000+low_offset:(interesting2[k2]+1)*1000-high_offset].plot(x='time',y='filtered_voltage',ax=axes[1,i], legend=False)
        axes[1,i].set_xlabel("time [ns]")
        axes[0,0].set_ylabel("voltage [mV]")
        axes[1,0].set_ylabel("voltage [mV]")
        k += 1

import numpy as np
spectrum     = pd.DataFrame(charge, columns = ['voltage'])
spectrum_fil = pd.DataFrame(charge_fil, columns = ['voltage'])
spectrum_ped = pd.DataFrame(charge_ped, columns = ['voltage'])
ax = spectrum.plot(kind='hist', bins = np.arange(0., 6. + 0.06, 0.06), logy = True)
#spectrum_fil .plot(kind='hist', bins = np.arange(0., 6. + 0.06, 0.06), logy = True, ax = ax, color='g', alpha = 0.5)
spectrum_ped .plot(kind='hist', bins = np.arange(0., 6. + 0.06, 0.06), logy = True, ax = ax, color='y', alpha = 0.5)
ax.set_xlabel("charge [pC]")
ax.set_ylabel("Entries / (0.06 pC)")
plt.show()


