#!/usr/bin/env ipython
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.fftpack
from matplotlib import rcParams, style
style.use('seaborn-muted')
rcParams.update({'font.size': 12})
rcParams['figure.figsize'] = 12, 12
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16

def print_progress(count, n_events):
    percent = float(count) / n_events
    hashes = '#' * int(round(percent * 20))
    spaces = ' ' * (20 - len(hashes))
    sys.stdout.write("\rPercent: [{0}] {1}%".format(hashes + spaces, int(round(percent * 100))))
    sys.stdout.flush()

# Take filename from terminal input.
data_name     = sys.argv[1]
data_ped_name = sys.argv[2]
# Match with pickle_data        #TODO: make this automatic
n_samples = 402
frameSize = 10.05e-8  # Not actually used in the analysis currently, but can be useful to know

data           = pd.read_pickle(data_name + '.pkl')
data_ped       = pd.read_pickle(data_ped_name + '.pkl')

charge_subset = []
resistance = 50
dt = (data['time'][1] - data['time'][0])/1e9 # to get back to seconds
# V = IR => I = V/R; Q = int_t0^t1 I dt = int_t0^t1 V/R dt => Q = sum(V/R *Delta t)

n_events = len(data)/n_samples

print data_name, data_ped_name
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

# Define range of integration
filterTimeRange = True
lower_time = 56.
upper_time = 77.
lower_time = 1.
upper_time = 100.
lower_time = 45.
upper_time = 65.
#lower_time = 50.
#upper_time = 65.
if filterTimeRange:
    data = data[(np.abs(data.filtered_voltage)<100) & (data.time > lower_time) & (data.time < upper_time)]
    data_ped = data_ped[(np.abs(data_ped.filtered_voltage)<100) & (data_ped.time > lower_time) & (data_ped.time < upper_time)]
else:
    data = data[(np.abs(data.filtered_voltage)<100)]
    data_ped = data_ped[(np.abs(data_ped.filtered_voltage)<100)]

# Shift baseline to zero
mean_ped_voltage = data_ped.filtered_voltage.mean()
print "Offet all voltages by the average baseline voltage of the pedestal:", mean_ped_voltage
data_ped.voltage          = data_ped.voltage - mean_ped_voltage
data_ped.filtered_voltage = data_ped.filtered_voltage - mean_ped_voltage
data.voltage              = data    .voltage - mean_ped_voltage
data.filtered_voltage     = data    .filtered_voltage - mean_ped_voltage
# Apply Time Over Threshold Filter
voltage_threshold = -2.
min_TOT = data[(data.filtered_voltage < voltage_threshold)].groupby(['eventID']).time.min()
max_TOT = data[(data.filtered_voltage < voltage_threshold)].groupby(['eventID']).time.max()
diff = (max_TOT - min_TOT) > 0.7 # 700ps
diff = diff[diff] # only select the events where the above condition is true
good_data = data[data.eventID.isin(diff.index)]
print "Number of good pulses below threshold (-2 mV) is:", len(diff), len(good_data), len(data)
bad_data = data[-(data.eventID.isin(diff.index))]

# Restrict the good data to within the time above the threshold. Only integrate in this window
good_data = good_data[(good_data.time > min_TOT[good_data.eventID]) & (good_data.time < max_TOT[good_data.eventID])]

# Group the data into events (i.e., separate triggers)
# Only keep events without spikes where there is some DAQ error due to large signals cut by the scope
spikes      = data     [(data     .voltage > 10)].groupby(['eventID']).eventID.min()
spikes_ped  = data_ped [(data_ped .voltage > 10)].groupby(['eventID']).eventID.min()
spikes_good = good_data[(good_data.voltage > 10)].groupby(['eventID']).eventID.min()
data_spikes_removed      = data     [~data     .eventID.isin(spikes.index)]
data_ped_spikes_removed  = data_ped [~data_ped .eventID.isin(spikes_ped.index)]
good_data_spikes_removed = good_data[~good_data.eventID.isin(spikes_good.index)]
print len(data_spikes_removed.groupby(['eventID']))

#data_spikes_removed = data
#data_ped_spikes_removed = data_ped
#good_data_spikes_removed = good_data

grouped_data      = data_spikes_removed     .groupby(['eventID'])
grouped_data_ped  = data_ped_spikes_removed .groupby(['eventID'])
grouped_data_good = good_data_spikes_removed.groupby(['eventID'])

print "Number of events after removing large voltage spikes", len(grouped_data)

# Plot the time position and voltage of the max voltage in each event
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey = True)
max_voltages     = data    .loc[grouped_data    .filtered_voltage.idxmin()]
max_voltages_ped = data_ped.loc[grouped_data_ped.filtered_voltage.idxmin()]
max_voltages    .plot(kind='scatter',x='time',y='filtered_voltage', title = '',  ax = axes[0])
max_voltages_ped.plot(kind='scatter',x='time',y='filtered_voltage', title = '', ax = axes[1])
axes[0].set_xlabel("time [ns]", fontsize = 20)
axes[1].set_xlabel("time [ns]", fontsize = 20)
axes[0].set_ylabel("filtered voltage [mV]", fontsize = 20)
axes[0].set_ylabel("")
fig.savefig('plots/' + sys.argv[1] + '_' + sys.argv[2] + '_max_voltage_vs_time.png')
print "number of max voltages", len(max_voltages)

mv_subset0 = max_voltages[(max_voltages.filtered_voltage < -4.)].set_index('eventID')

nrows = 4
ncols = 4
trace, trace_ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(10, 10))
count = 0
for i in range(nrows):
    for j in range(ncols):
        try:
            grouped_data.get_group(mv_subset0.index[count]).plot(x='time',y='filtered_voltage',ax=trace_ax[i,j], legend=False)
        except:
            print "No more traces found for C1", count
        trace_ax[i,j].set_xlabel("time [ns]")
        trace_ax[i,j].set_ylabel("voltage [mV]")
        count += 1
trace.savefig('plots/' + sys.argv[1] + '_' + sys.argv[2] + '_oscilloscope_traces_max.png')


# Plot the amplitude of each signal in a histogram
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey = True, sharex = True)
max_voltages['minus_voltage'] = -1*max_voltages['voltage']
max_voltages_ped['minus_voltage'] = -1*max_voltages_ped['voltage']

max_voltages['minus_voltage']    .hist(histtype='step', bins = 60, color='r', ax = axes[0])
max_voltages_ped['minus_voltage'].hist(histtype='step', bins = 60, color='r', ax = axes[1])
axes[0].set_yscale('log')
axes[1].set_yscale('log')
axes[0].set_xlim(-10, 100)
axes[1].set_xlim(-10, 100)
axes[0].set_ylim(ymin=0.1)
axes[1].set_ylim(ymin=0.1)
axes[0].set_xlabel("max signal amplitude [mV]", fontsize = 20)
axes[0].set_ylabel("Entries", fontsize = 20)
axes[1].set_xlabel("max signal amplitude [mV]", fontsize = 20)
axes[1].set_ylabel("Entries", fontsize = 20)
fig.savefig('plots/' + sys.argv[1] + '_' + sys.argv[2] + '_signal_amplitude_spectrum.png')

print len(grouped_data)

# Convert the voltage into a collected charge and sum over all voltages in the time window
scale  = -dt/resistance*1e12/1e3 #for picoColoumbs and to put voltage back in V
q      = scale*grouped_data     .filtered_voltage.sum()
q_ped  = scale*grouped_data_ped .filtered_voltage.sum()
q_good = scale*grouped_data_good.filtered_voltage.sum()

# Fit a normal distribution to the pedestal
from scipy.stats import norm
mu, std = norm.fit(q_ped.values)
mu_ped  = q_ped.mean()
std_ped = q_ped.std()
threshold = 5*std_ped
print "Fitted mean and standard deviation of the pedestal    : %0.3f, %0.3f  " % (mu, std)
print "Calculated mean and standard deviation of the pedestal: %0.3f, %0.3f\n" % (mu_ped, std_ped)

# Plot the spectrum of collected charge
loC = -1
hiC =  6.
nBins = 140
width = float(hiC-loC)/nBins
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
axes[0].set_yscale('log')
axes[1].set_yscale('log')
q     .hist(histtype='step', bins = np.arange(loC, hiC + width, width), color='r', ax = axes[0])
#q_good.hist(histtype='step', bins = np.arange(loC, hiC + width, width), color='g', ax = axes[0])
q_ped .hist(histtype='step', bins = np.arange(loC, hiC + width, width), color='b', ax = axes[1])
axes[0].plot((threshold, threshold), (0, 1e5), 'k-')
axes[1].plot((threshold, threshold), (0, 1e5), 'k-')


from ROOT import TH1F, TFile
from root_numpy import fill_hist
q_file = TFile('plots/' + sys.argv[1] + '_' + sys.argv[2] + '_charge_spectrum.root', "recreate")
q_hist1 = TH1F('hist1', 'title', nBins, loC, hiC)
q_hist2 = TH1F('hist2', 'title', 55, -10, 100)
fill_hist(q_hist1, q.as_matrix())
fill_hist(q_hist2, max_voltages['minus_voltage'].as_matrix())
q_hist1.Write();
q_hist2.Write();
q_file.Close();


max_q = q.max()

# uncomment these lines if you want to plot the fitted Gaussian
#x = np.linspace(loC, hiC, nBins)
#pdf = n_events*norm.pdf(x, mu, std)/nBins
#plt.plot(x, pdf, 'k', linewidth = 2)

axes[0].set_xlabel("charge [pC]", fontsize = 20)
axes[0].set_ylabel("Entries / (%0.2f pC)" % width, fontsize = 20)
axes[0].set_xlim(loC, hiC)
axes[0].set_ylim(0.1, 1e4)
axes[1].set_xlabel("charge [pC]", fontsize = 20)
#axes[1].set_ylabel("Entries / (%0.2f pC)" % width, fontsize = 20)
axes[1].set_xlim(loC, hiC)
axes[1].set_ylim(0.1, 1e4)
fig.savefig('plots/' + sys.argv[1] + '_' + sys.argv[2] + '_charge_spectrum.png')

# Now show the oscillioscope traces of a few events
subset1 = (q[(q > 0.0)       & (q < max_q/10.)]).sample(n=4).index
subset2 = (q[(q > max_q/10.) & (q < max_q/5. )]).sample(n=4).index
subset3 = (q[(q > max_q/5.)  & (q < max_q    )]).sample(n=4).index


if len(subset3) > 0:
    trace, trace_ax = plt.subplots(nrows=1, ncols=1, figsize=(7,7))
    grouped_data.get_group(subset3[0]).plot(x='time',y='voltage'         ,c='r', ax=trace_ax, legend=False)
    grouped_data.get_group(subset3[0]).plot(x='time',y='filtered_voltage',c='b',ax=trace_ax, legend=False)
    trace_ax.set_xlabel("time [ns]", fontsize = 20)
    trace_ax.set_ylabel("voltage [mV]", fontsize = 20)
    trace.savefig('plots/' + sys.argv[1] + '_' + sys.argv[2] + '_oscilloscope_traces_single.png')

    N = len(grouped_data.get_group(subset3[0]).voltage)
    # Make the FFT of the signal
    yf = scipy.fftpack.fft(grouped_data.get_group(subset3[-1]).voltage)
    yf_filtered = scipy.fftpack.fft(grouped_data.get_group(subset3[-1]).filtered_voltage)
    yf_noise = scipy.fftpack.fft(grouped_data.get_group(subset1[0]).voltage)
    yf_noise_filtered = scipy.fftpack.fft(grouped_data.get_group(subset1[0]).filtered_voltage)
    xf = np.linspace(0.0, 1.0/(2.0*dt), N/2)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (7,7))
    ax.plot(xf/1e9, 2.0/N * np.abs(yf[:N/2]), c='r', linewidth = 2, label='signal raw')
    ax.plot(xf/1e9, 2.0/N * np.abs(yf_filtered[:N/2]), c='b', linestyle='-', linewidth=2, label='signal filtered')
    ax.plot(xf/1e9, 2.0/N * np.abs(yf_noise[:N/2]), c='g', linestyle='-',linewidth=2, label='noise')
    ax.legend(loc='upper right')
    ax.set_xlabel("Frequency [GHz]", fontsize = 20)
    ax.set_ylabel("Magnitude", fontsize = 20)
    fig.savefig('plots/' + sys.argv[1] + '_' + sys.argv[2] + '_fft.png')

if len(subset1) > 0 and len(subset2) > 0 and len(subset3) > 0:
    trace, trace_ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(20, 10))
    for i in range(4):
        grouped_data.get_group(subset1[i]).plot(x='time',y='voltage'         ,ax=trace_ax[0,i], legend=False)
        grouped_data.get_group(subset1[i]).plot(x='time',y='filtered_voltage',ax=trace_ax[0,i], legend=False)
        grouped_data.get_group(subset2[i]).plot(x='time',y='voltage'         ,ax=trace_ax[1,i], legend=False)
        grouped_data.get_group(subset2[i]).plot(x='time',y='filtered_voltage',ax=trace_ax[1,i], legend=False)
        grouped_data.get_group(subset3[i]).plot(x='time',y='voltage'         ,ax=trace_ax[2,i], legend=False)
        grouped_data.get_group(subset3[i]).plot(x='time',y='filtered_voltage',ax=trace_ax[2,i], legend=False)
        ''' TODO: gives error, not sure why - look in to this.
        trace_ax[0,i].axvline(min_TOT[subset1[i]])
        trace_ax[0,i].axvline(max_TOT[subset1[i]])
        trace_ax[1,i].axvline(min_TOT[subset2[i]])
        trace_ax[1,i].axvline(max_TOT[subset2[i]])
        trace_ax[2,i].axvline(min_TOT[subset3[i]])
        trace_ax[2,i].axvline(max_TOT[subset3[i]])
        '''
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
        trace_ax[2,i].set_xlabel("time [ns]", fontsize = 20)
        trace_ax[0,0].set_ylabel("voltage [mV]", fontsize = 20)
        trace_ax[1,0].set_ylabel("voltage [mV]", fontsize = 20)
        trace_ax[2,0].set_ylabel("voltage [mV]", fontsize = 20)

trace.savefig('plots/' + sys.argv[1] + '_' + sys.argv[2] + '_oscilloscope_traces.png')

# Compute the mean of the collected charge above some threshold (which is defined in terms of the pedestal) to compute the gain
from math import sqrt
dark_count = q_ped[q_ped > threshold]
time_interval_secs = len(q_ped)*(upper_time - lower_time)/1e9 # to get into seconds
dark_rate = len(dark_count)/float(time_interval_secs)
dark_rate_err = np.sqrt(len(dark_count))/float(time_interval_secs)
frac_dark = len(dark_count)/float(len(q_ped))
err_dark = np.sqrt(frac_dark*(1-frac_dark)/len(q_ped))
print "Fraction of dark events above threshold is %0.3f \\pm %0.3f \t %d \t %d \t %0.3f \t %f" % (frac_dark, err_dark, len(dark_count), len(q_ped), threshold, time_interval_secs)
print "Dark rate in signal window %f \\pm %f Hz" % (dark_rate, dark_rate_err)

signal = q[q > threshold]
mean_signal_charge = signal.mean()
mean_signal_charge_err = signal.mean()/sqrt(len(signal))
gain0     = mean_signal_charge*1.e-12/1.602e-19/1e7
gain0_err = mean_signal_charge*1.e-12/1.602e-19/1e7/sqrt(len(signal)) # i.e. error on the mean
gain      = q_good.mean()*1.e-12/1.602e-19/1e7
gain_err  = gain/sqrt(len(q_good))
frac = len(signal)/float(len(q))
err = np.sqrt(frac*(1-frac)/len(q))
print "Fraction of events above threshold is %0.3f \\pm %0.3f \t %d \t %d \t %0.3f" % (frac, err, len(signal), len(q), threshold)
print "Using time-over-threshold:             gain of the photo-sensor is (%0.3f +- %0.3f) x 10^7" % (gain, gain_err)
print "Using 3-sigma threshold from pedestal: gain of the photo-sensor is (%0.3f +- %0.3f) x 10^7\n" % (gain0, gain0_err)
print "Using 3-sigma threshold from pedestal: mean of the events above threshold is (%0.3f +- %0.3f) x 10^7\n" % (mean_signal_charge, mean_signal_charge_err)

