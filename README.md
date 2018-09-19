# HFOD

## HFO Detector

Fast Cython implementation of scalp HFO detector first stage using rolling RMS values as described in:

Bernardo, Danilo, et al. "Visual and semi-automatic non-invasive detection of interictal fast ripples: A potential biomarker of epilepsy in children with tuberous sclerosis complex." Clinical Neurophysiology 129.7 (2018): 1458-1466.

based on:

von Ellenrieder, Nicolás, et al. "Automatic detection of fast oscillations (40–200 Hz) in scalp EEG recordings." Clinical Neurophysiology 123.4 (2012): 670-680.

## Installation / Setup
1. Run python setup_getBaseline.py to setup Cython file
2. To run use function procData(settings_dict, curr_batch, subject, params, overwrite=False, remove_blocky_artifact=False). See Notes below for clarification.
3. This detector requires at least one bandpassed data signal to run on which should be located in the bandpassed_data_path folder

## Notes
1. settings_dict should have following format, where data_batch is name of the experiment: 
		settings_dict = {
			  "analyzed_data_path": "data_batch/anal_data",
			  "hkl_data_path": "data_batch/hkl_data",
			  "plot_data_path": "data_batch/plot_data",
			  "bandpassed_data_path": "data_batch/bp_data",
			  "bb_data_path": "data_batch/bb_data",
			  "hp1_data_path": "data_batch/hp1_data",
			  "raw_data_path": "data_batch/raw_data",
			  "feat_data_path": "data_batch/feat_data",
			  "saved_stats_path": "data_batch/savedStats",
			  "trace_path": "data_batch/traces",
			  "luigi_mode": True
			}
 2. curr_batch is the name of the current experiment
 3. subject is the name of the current experiment subject
 4. params should have format [pc, bc, flc], where pc is the proportionality constant, bc is baseline constant bc as described in von Ellenrieder et al, and flc is fir length constant which is proportionality constant of FIR filter effective duration.
 
