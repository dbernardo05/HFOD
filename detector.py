#!/usr/bin/env python2.7
""" 
HFO DETECTOR - for scalp and iEEG

"""
import cPickle as pkl
import os
import os.path
import sys
import time

import h5py
import hickle as hkl
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

import getBaseline_Cy
from tasks import contiguous_regions, gen_params_string, get_dirPack, \
	get_files_paths, getLowcut, histo_2, \
	load_hHFOs, window_rms


def findHFOs(chN,chSig,sigRMS,t_hHFOs,t_hHFO_mrmsv,b,lowcut,sRate,Nb,winSize,lenReq, bc, pc, good_epochs):  #chN:current chan

	sigRMS[winSize-1:] = window_rms(chSig, winSize)
	firstBaseline = np.mean(sigRMS[:Nb])

	# t0 = time.time()
	len_rmsVals = len(sigRMS)
	lenBaseline = len_rmsVals-Nb

	baseline = np.empty(lenBaseline, dtype = 'float64')
	b[Nb:] = bc*getBaseline_Cy.getBaseline_Cy(sigRMS,baseline,Nb,pc,sRate,firstBaseline,len_rmsVals, lenBaseline) # [Nb:]

	hSig = sigRMS > b

	hSig = np.logical_and(hSig, good_epochs)
	hSig[-Nb*2:] = 0 # remove last 60 sec
	hSig[:Nb*2] = 0 # remove first 60 sec
	HFOs_chunk = contiguous_regions(hSig, lenReq) # returns array

	cHFOs_mRMSv_chunk = np.empty(HFOs_chunk.shape[0]).astype('float64')
	searchBuff = 50
	for idx,HFO in enumerate(HFOs_chunk):
		cHFOs_mRMSv_chunk[idx] = np.amax(sigRMS[HFO[0]-searchBuff :HFO[0] + lenReq + searchBuff+200]) # 50 = broad-BP_30-800 adjustment. 200 = narrow-BP e.g. 490-500

	# Find cHFOs_mRMSv for hHFOs that occur within this channel
	for idx,HFO in enumerate(t_hHFOs):
		curMRMSval = np.amax(sigRMS[HFO[0]-searchBuff:HFO[0] +lenReq + searchBuff + 200])
		if curMRMSval > t_hHFO_mrmsv[idx]:
			t_hHFO_mrmsv[idx] = curMRMSval
			t_hHFOs[idx,2] = lowcut # freqband of HFO; maybe calculate this in future spectrally

	return hSig, HFOs_chunk, cHFOs_mRMSv_chunk, t_hHFOs, t_hHFO_mrmsv, b, sigRMS

def procData(settings_dict, curr_batch, subject, params, overwrite, remove_blocky_artifact=False):
	sRate, pc, bc, flc, mode_ext = params
	params_str= gen_params_string(params)
	anal_dir,bb_dir,bp_dir,feat_dir,hkl_dir,hp1_dir,plot_dir,savedStats_dir,trace_dir,raw_dir = get_dirPack(curr_batch, subject, params, settings_dict)

	print '\tParameters:', params, ' | Overwrite:', overwrite, ' | Band mode:', mode_ext

	if mode_ext == 'nb':
		bp_dir += '/nb'
	elif mode_ext == 'hfob':
		bp_dir += '/hfob'

	for_cluster_hEOIs_hkl_file = anal_dir + '/' + params_str + '/' + subject + params_str + '_hHFOs_for_cluster.hkl' # Includes selected max narrowbands (not spectrally located)
	cHFO_hSig_file = hkl_dir + '/'+ subject + params_str + '_b-all' +'_cHFO_hSig.hkl'
	debug_Sigs_file = hkl_dir + '/'+ subject + params_str + '_b-all' +'_debug_Sigs.hkl'

	hHFO_rmsFile = hkl_dir + '/' + subject + params_str + '_b-' + str(250) +'_hHFOrms.hkl'
	#hEOIs_hkl_file = hkl_dir +'/'+subject + '_sr-'+ str(int(sRate)) + '_b-' + str(250) + params + '_hHFOs_ann.hkl'
	bp_proc_data_dir = hkl_dir + '/bp_proc/'

	if not os.path.exists(bp_proc_data_dir):
		os.makedirs(bp_proc_data_dir)

	if (not overwrite) and (os.path.isfile(hHFO_rmsFile) & os.path.isfile(cHFO_hSig_file)):
		print '\t\tAlready ran procData! Skipping ahead!'
		return

	bp_files = get_files_paths(bp_dir) # bandpassed files
	num_bp_files = int(len(bp_files)) # number of bp_files

	HOF_buffer = np.empty(shape=(1000000), dtype='uint32')
	HOFl_buffer = np.empty(shape=(1000000), dtype='uint16')
	HOFc_buffer = np.empty(shape=(1000000), dtype='uint8') # channel
	HFOrmsMax_buffer = np.empty(shape=(1000000), dtype='float64')

	# Prepare hHFOs
	hHFOs = np.array([ h[:4] for h in load_hHFOs(anal_dir, subject)], dtype='uint32')
	hHFO_mrmsv = np.zeros(hHFOs.shape[0], dtype='float64')

	print '\tNumber hHFOs :', hHFOs.shape

	print '#'*10, bp_dir
	print '#'*10, bp_files

	data = h5py.File(bp_files[0])["data"] # or "EEG" (old way)
	sRate = 2000
	baseline_len = 10

	# add zeros for finding good epochs
	epochLen = 5 * sRate
	nWins = np.ceil(data.shape[1] / (epochLen + 0.0))
	rem = nWins * epochLen - data.shape[1]
	data = np.hstack((data, np.zeros((int(data.shape[0]), int(rem)))))  # adds zeros so data can be partitioned equally
	dShape = data.shape
	data_len = dShape[1]

	if remove_blocky_artifact:
		ref_good_epochs = build_goody_store(subject,params,int(sRate))
		bp_good_epochs = np.ones(data.shape,dtype='bool')
		print 'Preparing BP chan:'

		for bp_chan,bp_sig in enumerate(bp_good_epochs):
			print '   Bipolar Chan:', bp_chan,
			ref_chans = get_ref_from_bp_ch_key(bp_chan) # Converts ch_a to chN (also changes from matlab format; subtracts one)
			for ref_chan in ref_chans:
				bp_good_epochs[bp_chan,:] = np.logical_and(bp_good_epochs[bp_chan,:],ref_good_epochs[ref_chan,:])
				percent_artifacts = 1 - np.sum(ref_good_epochs[ref_chan,:])/(data_len+0.0)
				print '\nCh-ref:', ref_chan, " | ", "%0.3f" % percent_artifacts, '% blocky signal'

		print 'Finished. Stats for BP good epoch mask:'
		for ch_n, ch in enumerate(bp_good_epochs):
			percent_artifacts = 1 - np.sum(ch)/(data_len+0.0)
			print 'Ch-BP:', ch_n, " | ", "%0.3f" % percent_artifacts, '% blocky signal'

	else:
		# No blocky artifact, just use bipolar montage, and use all epochs
		ref_good_epochs = np.ones(dShape,dtype='bool')
		bp_good_epochs = np.ones(dShape,dtype='bool')

	hSig = np.zeros((dShape[0],dShape[1],num_bp_files),dtype=bool)
	bSig = np.zeros((dShape[0],dShape[1],num_bp_files))
	rmsSig = np.zeros((dShape[0],dShape[1],num_bp_files))

	for r, bpFile in enumerate(bp_files):
		lowcut = getLowcut(bpFile,bp_dir,subject)

		cHFO_hklFile = bp_proc_data_dir + subject + params_str + '_b-' + str(lowcut) + '_HFOidx.hkl'
		cHFO_rmsFile = bp_proc_data_dir + subject + params_str + '_b-' + str(lowcut) +'_HFOrms.hkl'

		if not overwrite and (os.path.isfile(cHFO_hklFile) & os.path.isfile(cHFO_rmsFile) & os.path.isfile(hHFO_rmsFile)):
			#above used to include   os.path.isfile(hEOIs_hkl_file); this was removed b/c not in code below
			print 'Already processed files for:', subject, bpFile
			continue # skip if bpFile already processed

		# dataStruct = loadmat(bpFile) # for .mat v5 file
		dataStruct = h5py.File(bpFile)
		data = np.array(dataStruct["data"])
		firLen = dataStruct.attrs['len']

		z=0 #index for HOF_buffer

		lenReq = int((sRate/(lowcut+5))*4 + firLen*flc)

		# Add zero buffer to end of file for epoch feature
		epochLen = 5 * sRate
		nWins = np.ceil(data.shape[1] / (epochLen + 0.0))
		rem = nWins * epochLen - data.shape[1]
		data = np.hstack((data,np.zeros((int(data.shape[0]), int(rem)))))  # adds zeros so data can be partitioned equally

		print 'Processing {0} | Band: {1} | firLen: {2:.2f} | Length req: {3:.2f}'.format(bpFile.rsplit('/', 1)[-1], lowcut, firLen, lenReq)
		t0 = time.time()
		# Init to speed up
		sigRMS = np.zeros((int(data.shape[1])))
		b = np.zeros(sigRMS.shape)
		Nb = int(baseline_len*sRate)
		ptsPerCycle = int(sRate / (lowcut + 10))  # 10 is 1st passband + 10, b/c bandwidth is 20 Hz
		winSize=ptsPerCycle*4

		for c,col in enumerate(data):
			#print 'c shape', col.shape
			#print '-Ch', c, '...'
			#hSig[c,:,r],rmsVals[c,:,r] = findHFOs(col,hSig[c,:,r],lowcut,rmsVals[c,:,r]) # for debugging
			#hSig[c,:,r],cHFOs = findHFOs(c,col,hSig[c,:,r], hsN, lowcut,HOF_lilbuffer)

			# Passing hHFOs. Only pass hHFOs that have same channel c
			chIdx = hHFOs[:, 3] == c  # c ranges from 0 - 17
			hSig[c,:,r], cHFOs, rmsMaxVals, t_hHFOs, t_hHFO_mrmsv, bSig[c,:,r], rmsSig[c,:,r] = findHFOs(c, col, sigRMS, hHFOs[chIdx,:],
																				hHFO_mrmsv[chIdx], b, lowcut, sRate,
																				Nb, winSize, lenReq,
																				bc, pc, bp_good_epochs[c])  #current channel HFOs

			hHFOs[chIdx, :] = t_hHFOs
			hHFO_mrmsv[chIdx] = t_hHFO_mrmsv

			zz=z+len(cHFOs[:,0])
			# print 'z',z,'addZ',len(cHFOs),'zz',zz # debugging
			HOF_buffer[z:zz]=cHFOs[:,0]
			HOFl_buffer[z:zz]=cHFOs[:,1]
			#print 'Last HFO:', cHFOs[-1:,:]
			HOFc_buffer[z:zz]=c
			HFOrmsMax_buffer[z:zz]=rmsMaxVals
			z=zz

		print 'Total RMS spike events found:', z, '|',
		t1 = time.time()
		print 'Search time calc:', t1 - t0

		# WHEN RAW DATA USED, THIS CORRECTS cEOI LOCATIONS TO ABSOLUTE RAW DATA EOI
		grpDelay = 512 + 51  # 395 = Order of 790 / 2. 50 is additional delay from pre-bandpass processing

		HOF_idx = np.zeros((len(HOF_buffer[:zz]),4),dtype='uint32')
		HOF_idx[:, 0] = HOF_buffer[:zz] - grpDelay ## + np.floor_divide(HOFl_buffer[:zz],2) (center of HFO)
		HOF_idx[:, 1] = HOFl_buffer[:zz] # length of HFO
		HOF_idx[:, 2] = lowcut # freqband of HFO
		HOF_idx[:, 3] = HOFc_buffer[:zz] # channel of HFO
		#HOF_idx[:,4] = sRate # sRate of HFO
		HFOrms = HFOrmsMax_buffer[:zz] # remove end

		hkl.dump(HOF_idx.astype('uint32'), cHFO_hklFile, mode='w')
		hkl.dump(HFOrms.astype('float64'), cHFO_rmsFile, mode='w')

	if not overwrite and os.path.isfile(cHFO_hSig_file):
		print 'Already saved files for:', subject, cHFO_hSig_file
	else:
		hkl.dump(hSig.astype('uint8'), cHFO_hSig_file, mode='w')

	with open(for_cluster_hEOIs_hkl_file, mode='w') as f:
		hkl.dump(hHFOs.astype('uint32'), f)  # commented out 2015.09.23 b/c didn't seem necessary w/ new hEOIs paradigm

	if np.isnan(np.sum(hHFO_mrmsv)):  # make sure this isn't zero
		hHFO_mrmsv = np.zeros_like(hHFO_mrmsv)

	if not overwrite and os.path.isfile(hHFO_rmsFile):
		print 'Already saved files for:', subject, hHFO_rmsFile
	else:
		hkl.dump(hHFO_mrmsv, hHFO_rmsFile, mode='w')

	if not overwrite and os.path.isfile(debug_Sigs_file):
		print 'Already saved files for:', subject, debug_Sigs_file
	else:
		hkl.dump(np.array([rmsSig, bSig]), debug_Sigs_file, mode='w')
