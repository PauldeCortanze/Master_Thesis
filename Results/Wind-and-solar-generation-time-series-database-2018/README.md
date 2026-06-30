IEA Wind, Task 25 time series database
======================================

Database contains realised and forecast values for wind and solar photovoltaic 
generation and electrical load. 
All CF values are in p.u. and load is share of maximum load during year. 
All times are in UTC.

Realised values include any curtailment of generation except for Germany for which
also total ‘possible’ generation data was readily available. 

Database version: 2018.0 \
Author: Erkka Rinne <erkka.rinne@vtt.fi>


## Data sources

See file *summary.xlsx* for a summary of used data sources and variable status.
For a complete list of sources, please refer to the descriptor file *datapackage.json* 
under the "sources" key.


## Data cleaning

Data processing methodology is published at https://github.com/vttresearch/windtask25-timeseries.

Missing values in the raw time series were interpolated up to four consecutive time steps.

An algortihm to remove sudden peaks and drops from original data was developed.
See functions `remove_drops()` and `remove_peaks()` in module [`src.data_cleaning`][data_cleaning].
The main functionality is based on [`scipy.signal.find_peaks`][find_peaks].
The idea was to detect single time step peaks (or drops) or ‘plateaus’ of two time steps.
The threshold value for a peak was set to the smallest absolute change where the change
was greater than one standard deviation away from the mean change. 
Plateau prominence was set to the same threshold.
Peaking (dropping) values were removed and the gaps interpolated from the edges.
Highest mean-normalised RMSE of the filtered time series for onshore wind generation was 
less than 10% (Great Britain), see more analysis of the data cleaning process in the Notebook 
[*02b-analyze-cleaning.ipynb*][analyze-cleaning].

Some even larger errors were removed manually. For example, GB load data had a 6-hour sudded drop of
relative load from above 0.7 to under 0.1. This period was cleared and interpolated linearyl from the edges.
See notebook [04-analyze-data.ipynb][analyze-data] for details.



[data_cleaning]: https://github.com/vttresearch/windtask25-timeseries/blob/main/src/data_cleaning.py
[find_peaks]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
[analyze-cleaning]: https://github.com/vttresearch/windtask25-timeseries/blob/main/notebooks/02b-analyze-cleaning.ipynb
[analyze-data]: https://github.com/vttresearch/windtask25-timeseries/blob/main/notebooks/04-analyze-data.ipynb
