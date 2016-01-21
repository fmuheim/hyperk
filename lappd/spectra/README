Greig Cowan, 2015

This code can be used to process the spectra obtained from a Tektronic oscilloscope
operating in FastFrame mode. The code produces the following plots

   * 2D distribution of maximum voltages recorded for each event versus the time in the event
   * Integrated charge spectrum
   * Some example oscilloscope traces (useful for debugging)

The code operates in a two stage procedure.

   1. First process the .csv files into python .pkl files by modifing pickle_data.py to
   include the path to your .csv file. Then run
      $ ./pickle_data.py

   2. To make the plots modify make_spectra.py to put the name of the pickle file and then run
      $ ./make_spectra.py

Requirements

   * The code requires ipython, numpy, pandas and matplotlib. All of these should be available
   with any modern python distribution (i.e., Anaconda).

ToDo

   * Add command line arguments to allow paths to be specified at run-time.

Example plots

![My image](plots/max_voltage_vs_time.png?raw=true "Title")
![My image](plots/charge_spectrum.png?raw=true "Title")
![My image](plots/oscilloscope_traces.png?raw=true "Title")
