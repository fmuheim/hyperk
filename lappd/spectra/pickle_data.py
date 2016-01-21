#!/usr/bin/env ipython
import pandas as pd
from scipy import signal

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
'''
data_ped       = pd.read_csv('151006_092037.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
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
'''
'''
nPoints = 2500
window = 100e-9
# Using nominal 40ns window
#data           = pd.read_csv('151005_141912_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})

#HV 2.55kV
#LED 1.0V
#J7
# Using 20ns frame window and resolution of 1000
data_name     = '151009_155610'
data_ped_name = '151009_161113'
data                  = pd.read_csv(data_name   +'.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
data = data.append(     pd.read_csv('151009_163419.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3}), ignore_index = True)
data = data.append(     pd.read_csv('151009_163542.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3}), ignore_index = True)
data = data.append(     pd.read_csv('151009_163739.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3}), ignore_index = True)
data_ped              = pd.read_csv(data_ped_name+'.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
nPoints = 1000
window = 40e-9

#HV 2.55kV
#LED 1.0V
#J11
data_name     = '151009_171100_Ch1'
data_ped_name = '151009_171617_Ch1'
#J9
data_name     = '151009_171100_Ch3'
data_ped_name = '151009_171617_Ch3'
#J13
#data_name     = '151009_171100_Ch4'
#data_ped_name = '151009_171617_Ch4'
data                  = pd.read_csv(data_name    +'.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
data_ped              = pd.read_csv(data_ped_name+'.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
nPoints = 1000
window = 40e-9
'''

data_name     = '160121_102056_Ch1'
data_ped_name = '160121_102056_Ch1'
data                  = pd.read_csv(data_name    +'.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
data_ped              = pd.read_csv(data_ped_name+'.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun2, 'time':cfun3})
nPoints = 1250
window = 100e-9


print "number of entries", len(data)

'''
HV 2.2kV
LED 2.1V
different resolution used
data_gate      = pd.read_csv('151001_141511_Ch2.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4],converters={'voltage':cfun})
data           = pd.read_csv('151001_141511_Ch3.csv',names=['nan1','nan2','nan3','time','voltage'],usecols=[3, 4])
'''

nEvents = len(data)/nPoints
nEvents_ped = len(data_ped)/nPoints

eventID = []
for i in range(nEvents):
    for j in range(nPoints):
        eventID.append(i)
data['eventID'] = pd.Series( eventID, index = data.index)

eventID = []
for i in range(nEvents_ped):
    for j in range(nPoints):
        eventID.append(i)
data_ped['eventID'] = pd.Series( eventID, index = data_ped.index)

filtered_voltages = []
filtered_voltages_ped = []

# Filter the voltages
for i in range(nEvents):
    d = filter_signal(data['voltage'][i*nPoints:(i+1)*nPoints].as_matrix(), nPoints, window)
    filtered_voltages.extend(d)
for i in range(nEvents_ped):
    d = filter_signal(data_ped['voltage'][i*nPoints:(i+1)*nPoints].as_matrix(), nPoints, window)
    filtered_voltages_ped.extend(d)

# Add a new column to the DataFrame
data['filtered_voltage'] = pd.Series(filtered_voltages, index = data.index)
data_ped['filtered_voltage'] = pd.Series(filtered_voltages_ped, index = data_ped.index)

data.to_pickle(data_name+'.pkl')
data_ped.to_pickle(data_ped_name+'.pkl')

