#!usr/bin/env python
# -------------------------------------------------------------
# eeglib.py - EEG data analysis library 
# (c) 2012 // Cris Lanting (cris@ihr.mrc.ac.uk // c.lanting@gmail.com)
# -------------------------------------------------------------

import numpy as np
import sys, string
import matplotlib.pyplot as plt
import shutil as shu
import os, sys, glob
import os.path as op
import time, sys, math

	
def eeg_baselinecorrect(array,tim,window):
	"""Baselinecorrection of EEG data

	Args:
		array : contains EEG data; can be either a single channel (ntpts), multiple channels (ntpts x nchan), or
				data from multiple  subjects (nsub x nchan x ntpts)
		tim : timing information, array of timepoints in the eeg channel
		window : time-window to perform the baseline correction on
	Returns:
		arr_bl : baseline-corrected EEG data 
		
	"""

	rmin,rmax = window[0],window[1]	
	idx = np.squeeze(np.logical_and(tim>=rmin,tim<=rmax))
	arr_bl = np.zeros(array.shape)
	
	if len(array.shape) == 1:
		ntpts = len(array)
		arr_bl = array - np.mean(array[idx])
	elif len(array.shape) == 2:
		[nchan,ntpts]=array.shape
		for i in range(nchan):
			arr_bl[i,:]=array[i,:]-np.mean(array[i,idx])
	elif len(array.shape) == 3:
		[nsub,nchan,ntpts]=array.shape
		for i in range(nsub):
			for j in range(nchan):
				arr_bl[i,j,:]=array[i,j,:]-np.mean(array[i,j,idx])
	else:
		print 'Error: only works with 1,2 or 3 dimensions'

	return arr_bl
	
def esp(tim,array):
	gfp = eeg_rms(array)
	tmin = np.min(tim)
	tmax = np.max(tim)
	figure3 = plt.figure(3,figsize=[7,4])
	ax1 = plt.subplot(111,autoscale_on=False, xlim=[tmin,tmax], ylim=[-4,4])
	plt.plot(tim,array.transpose(),color='grey',lw=1,label='all channels')
	plt.plot(tim,gfp,color='black',lw=3,label='GFP')
	plt.plot(tim,array[0,:],color='red',lw=3, label='Cz channel')
	ax1.set_xlabel('time (ms)')
	ax1.set_ylabel(r'Amplitude ($\mu$V)')
	plt.legend(loc='best')
		

def eeg_bootptp(bootsample,tim,window):
	""" Determine peak-to-peak values after bootstrapping EEG data

	Args:
		bootsample : contains bootstrapped EEG data (timepoints [ntpts] x number of bootstrap samples [nboot])
		tim : timing information, array of timepoints in the eeg channel
		window : time-window to in which eeg_ptp lookts for the peaks for each of the bootstrap-samples
	Returns:
		ptpvals : peak-to-peak values (of the P2 - N1 deflection) for each of the bootstrap samples
		
	Dependence:
		function eeg_ptp
		
	"""
	[ntpts,nboot] = bootsample.shape
	ptpvals = np.zeros((nboot))
	for i in range(0,nboot):
		ptpval, minval, t_min, maxval, t_max = eeg_ptp(bootsample[:,i],tim,win)
		ptpvals[i]=ptpval
		cnt = 100.0*i/nboot
		sys.stdout.write("progress: \r%d%%" %cnt)
	return ptpvals


def eeg_bootstrp(array,nboot):
	""" Bootstrap EEG data

	Args:
		array : contains original EEG data (number of subject [nsub] x number of channels [nchan] x number of timepoints [ntpts])
			note: EEG data can be either be 3D (nsub x nchan x ntpts) or 2D (nsub x ntpts), representing (averaged) source waveforms. 
		nboot : number of bootstrap samples (e.g, 1000)

	Returns:
		boots : bootstrapped EEG data; gives an average EEG response for each of the bootstrap samples  (number of timepoints [ntpts] x number of bootstrap samples [nboot])
		sample : original bootstrap sample indices used to create the averages (number of subject [nsub] x number of bootstrap samples [nboot]; special case is the first sample, 
		which represents the actual data
		
	"""	
	if len(array.shape) == 3:
		print "Bootstrap on 3D data set - taking Cz data"
		#get info from original sample and bootstrap Cz data
		[nsub,nchan,ntpts] = array.shape
		#bootstrap indices	
		sample = np.random.random_integers(0,nsub-1,[nsub,nboot])
		sample[:,0] = range(nsub)
		cz = array[:,0,:] 	#assuming cz channel is first channel in file
		boots = np.zeros((ntpts,nboot))
		for i in range(0,ntpts):
			cnt = 100.0*i/ntpts
			sys.stdout.write("progress: \r%d%%" %cnt)
			cz_sample = cz[:,i]
			for j in range(0,nboot):
				boots[i,j] = np.mean(cz_sample[sample[:,j]])
	elif len(array.shape) == 2:
		print "Bootstrap on 2D data set - assuming source waveform data\n"
		[nsub,ntpts] = array.shape
		sample = np.random.random_integers(0,nsub-1,[nsub,nboot])
		sample[:,0] = range(nsub)
		boots = np.zeros((ntpts,nboot))
		for i in range(0,ntpts):
			cnt = 100.0*i/ntpts
			sys.stdout.write("\r%d%%" %cnt)
			tmp = array[:,i]
			for j in range(0,nboot):
				boots[i,j] = np.mean(tmp[sample[:,j]])
				
		#boots = np.sort(boots,axis=1)
		#sample = np.sort(sample,axis=1)
		print('\n')
	elif len(array.shape) == 1:	
		print "Error - can't perform bootstrap procedure on 1d EEG data"
	return boots,sample
	
def eeg_bootstrapCI(array,alpha):
	"""  Given a bootstrap sample of EEG data (n-timepoint [ntpts] x m-bootstrapsamples [nboot]) eeg_bootstrapCI returns the [alpha/2 (1-alpha/2)] confidence interval
		
	Args:
		array :  contains bootstrapped EEG data (ntpts x nboot)
		alpha : confidence level (e.g. 0.05 or 0.01)
		
	Returns:
		array_low : lower limit of the confidence interval for each of the time-point (ntpts)
		array_high : upper limit of the confidence interval for each of the time-point (ntpts)
	
	"""	
	
	if len(array.shape) == 3:
		print "Only works on 2D bootstrapped data (ntpts x nboot)"
		array_low = []
		array_high = []
	else:
		ntpts, nboot = array.shape
		#sort along last (bootstrap) dimension
		array_srt = np.sort(array,axis=1)
		array_low = array_srt[:,np.round(nboot*alpha/2)-1]
		array_high = array_srt[:,np.round(nboot*(1-alpha/2))-1]
		return array_low,array_high

def eeg_butterfly(array,tim):
	""" Butterfly plot of EEG data

	Args:
		array : EEG data (number of channels [nchan] x timepoints [ntpts])
		tim : timing information, array of timepoints in the eeg channel
		
	Returns:
		butterfly plot in current figure		
		
	"""
	lineprops = dict(linewidth=1, color='grey', linestyle='-')
	linerms = dict(linewidth=3, color='black', linestyle='-')
	linecz  = dict(linewidth=3, color='red', linestyle='-')
	lineax = dict(linewidth=1, color='black', linestyle='--')        
	[nchan,ntpts] = array.shape
	array_rms = eeg_rms(array)
	for i in range(0,nchan):
		plt.plot(tim,array[i,:],**lineprops)
	
	rms = plt.plot(tim,array_rms,**linerms)
	cz = plt.plot(tim,array[0,:],**linecz)
	plt.axhline(**lineax)
	plt.axvline(**lineax)


def eeg_derivative(array):
	""" Determine temporal derivative of EEG data (1-dimensional )

	Args:
		array : EEG data (timepoints [ntpts])
		
	Returns:
		derivative: derivative of EEG data
		
	"""
	if len(array.shape) == 1:
		derivative = []	
		for i in xrange(1,len(array)):
			derivative.append(array[i]-array[i-1])
	else:
		print "Not yet implemented"
		derivative = []
	return derivative

def eeg_fdr(p_array,q,plot='false'):
	"""  Benjamini & Hochberg's FDR control algorithm for N multiple comparisons of significane tests on EEG data
		
		The FDR is defined as the proportion of false positives (N_fp) among the total number of positives (N_p).
		See e.g. http://en.wikipedia.org/wiki/False_discovery_rate, or:
		Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. 
		Journal of the Royal Statistical Society Series B, 57, 289\u2013300.

	Args:
		p_array :  array of p-values (p_i) from any significance test (e.g. eeg_permute or t-test), consisting of N p-values. 
		q : FDR threshold (e.g. 0.05); implies that this fraction of all significant tests will result in a false positive 
		plot = 'true' : optional keyword to plot, if not used, data will not be plotted 
		
	Returns:
		p_thresh_fdr : FDR adjusted threshold for single tests
		
		
	"""	
	N = len(p_array)                             
	p_array_srtind = np.argsort(p_array)
	p_array_srt = p_array[p_array_srtind]
	p_bound=np.zeros((N))
	for i in range(N):
		p_bound[i] = i * q/N
	
	idx = p_array_srt < p_bound
	
	if np.sum(idx == 'true')  > 0:
		p_thresh_fdr = np.max(p_array_srt[idx])
		i_max_fdr = p_array_srt[idx].argmax()			
		if plot == 'true':
			fig = plt.figure(19,figsize=[5,5])
			ax = fig.add_subplot(111, autoscale_on=False, xlim=[0,120], ylim=[0,10])
			plt.plot(N*p_array_srt,'bo-',lw=2)
			plt.plot(N*p_bound,'k--',lw=2)
			plt.plot([10,20],[9,9],'bo-',lw=2)
			plt.plot([10,20],[8,8],'k--',lw=2)
			plt.text(25,9, r'$p_{i} \cdot N$', fontsize = 13, va='center')
			plt.text(25,8, r'$q \cdot i$', fontsize = 13,va='center')
			plt.text(1.2 * i_max_fdr,0.5 * p_thresh_fdr*N,'FDR (q = ' + str('%.2f' %q) + ') = ' + str('%.4f' %p_thresh_fdr),va = 'center')
			plt.fill_between([0,i_max_fdr],[p_thresh_fdr*N,p_thresh_fdr*N],0,color='k',alpha=0.3,lw=0)
			plt.xlabel('i',fontsize = 13)
			plt.ylabel(r'$N_{FP}$',fontsize = 13)	
	else:
		print "None of the p-value exceeds the FDR threshold (there is no p-value (p_i) for which p_i < i*q/N) "
		p_thresh_fdr = []
	
	return p_thresh_fdr
	


def eeg_findnearest(x,X):
	"""  Find the value and its index in array 'x' that is nearest to the scalar value 'X'.
		
	Args:
		x : array
		X : scalar value
			
	Returns:
		val : value of the closest value to X
		idx : index of the element in x that is nearest in value to X
		
	"""	
	#x array or vector and X a scalar
	absdif = np.abs(x-X)
	val = np.min(absdif)
	idx = absdif.argmin()
	return val,idx

def eeg_loaddata(filedir,filemask):
	"""  Loads a list of AVR files of a certain type, filemask (e.g., '*_condition1.avr'), in the directory, filedir (e.g., '~/eeg_data/results/'), into a variable 
		
	Args:
		filedir : directory (e.g., '~/eeg_data/results/' )
		filemask: filemask (e.g., '*_condition1.avr')
		
	Returns:
		data : array of data (number of files x number of channels x number of timepoints)
		tim : timing information, array of timepoints in the eeg channel
		nchan: number of EEG data channels in each of the files
		files: list of all files (according to filemask) in the directory (filedir)
		
	"""	
	files = glob.glob1(filedir,filemask)
	print "loading %d files" %len(files)
	eeg,tim,nchan,ntpts = eeg_readavr(op.join(filedir,files[0])) #just to initialize the next line
	data = np.zeros((len(files),eeg.shape[0],eeg.shape[1]))
	for i in range(len(files)):
		eeg,tim,nchan,ntpts = eeg_readavr(op.join(filedir,files[i]))
		data[i,:,0:ntpts]=eeg[:,0:ntpts]
		cnt = 100.0*i/len(files)	
		sys.stdout.write("progress: \r%d%%" %cnt)

        return data,tim,nchan,files

def eeg_smooth(array,window,window_len):
	"""  Smoothing of EEG data based on the convolution of a smoothing window with the original signal
		see e.g., http://scipy.org/Cookbook/SignalSmooth
		
	Args:
		array :  contains EEG data; can be either a single channel (ntpts) or multiple channels (ntpts x nchan)
		window : the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
			flat window will produce a moving average smoothing.
		window_len: dimension of the smoothing window; should be odd integer
		
	Returns:
		array_smooth : smoothed EEG data
		
	"""	
	array_smooth = np.zeros(array.shape)
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser']:
		raise ValueError, "Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'kaiser'"
		
	if window == 'flat':
		w = np.ones(window_len)
	elif window == 'kaiser':
		w = eval('np.'+window+'(window_len,4)')		
	else:
		w = eval('np.'+window+'(window_len)')		
		
	
	if len(array.shape) == 1:
		ntpts = len(array)
		array_smooth = np.convolve(array, w/w.sum(), mode='same')
	
	elif len(array.shape) == 2:
		[nchan,ntpts] = array.shape
		for i in range(0,nchan):
			array_smooth[i,:] = np.convolve(array[i,:], w/w.sum(), mode='same')
	
	elif len(array.shape) > 2:
		print 'Error: only works with 1 or 2 dimensions'
		
	return array_smooth

def eeg_peaks(array,tim,window,plot='false'):
	"""  Determine the P1, N1 and P2 peak values of the evoked responses in a time-window
		
	Args:
		array :  contains single channel or source-waveform data. EEG data can be either a single channel (ntpts) or multiple channels (ntpts x nchan)
		tim : timing information, array of timepoints in the eeg channel
		window : time-window in which to look for the peaks (e.g., [0,125,250] )
		
		
	Returns:
		p1 : P1 peak value
		tp1 : latency of the P1 peak in ms 
		n1 : N11 peak value
		tn1 : latency of the N1 peak in ms 
		p2 : P2 peak value
		tp2 : latency of the P2 peak in ms 
		(optional) : plot of  the waveform and the P1,N1,P2 values 
		
	"""	
	r0,r1,r2 = window[0],window[1],window[2]
	idx_1 = np.squeeze(np.logical_and(tim>=r0,tim<=r1))
	idx_2 = np.squeeze(np.logical_and(tim>=r1,tim<=r2))        
	p1 = np.max(array[idx_1])
	n1 = np.min(array[idx_1])
	p2 = np.max(array[idx_2])
	tp1 = tim[idx_1][array[idx_1].argmax()]
	tn1 = tim[idx_1][array[idx_1].argmin()]
	tp2 = tim[idx_2][array[idx_2].argmax()]
	if plot == 'true':          
		lineax = dict(linewidth=1, color='black', linestyle=':')        
		fig = plt.figure(19,figsize=[7,5])
		ax = fig.add_subplot(111, autoscale_on=False, xlim=[window[0]-200,window[-1]+200], ylim=[1.3*np.min(array[idx_1]),1.3*np.max(array[idx_2])])
		plt.plot(tim,array,'k-',lw=3)
		plt.plot(tp1,p1,'ro')
		plt.plot(tn1,n1,'go')
		plt.plot(tp2,p2,'bo')
		ax.axvline(float(r0),**lineax)
		#ax.axvline(float(r1),**lineax)	
		ax.axvline(float(r2),**lineax)
		ax.axhline(**lineax)
		plt.text(tp1-220,1.2*p1,'P1 = %.2f nAm at %.0f ms'  %(p1,tp1))
		plt.text(tn1-40,1.2*n1,'N1 = %.2f nAm at %.0f ms'  %(n1,tn1))
		plt.text(tn1+40,1.1*p2,'P2 = %.2f nAm at %.0f ms'  %(p2,tp2))
		plt.xlabel('time (ms)',fontsize = 13)
		plt.ylabel('Amplitude',fontsize = 13)


	return [p1,n1,p2,tp1,tn1,tp2]
	
def eeg_peaks_gfp(array,tim,onset,plot='false'):
	"""  Determine the P1, N1 and P2 peak values of the global field power of the evoked responses in a time-window
		
	Args:
		array :  contains single channel or source-waveform data. EEG data can be either a single channel (ntpts) or multiple channels (ntpts x nchan)
		tim : timing information, array of timepoints in the eeg channel
		window : time-window in which to look for the peaks (e.g., [0,125,250] )
		
		
	Returns:
		p1 : P1 peak value
		tp1 : latency of the P1 peak in ms 
		n1 : N11 peak value
		tn1 : latency of the N1 peak in ms 
		p2 : P2 peak value
		tp2 : latency of the P2 peak in ms 
		(optional) : plot of  the waveform and the P1,N1,P2 values 
		
	"""	
	r0 = onset + 25
	r1 = onset + 75
	r2 = onset + 150
	r3 = onset + 300	
	idx_p1 = np.squeeze(np.logical_and(tim>=r0,tim<=r1))
	idx_n1 = np.squeeze(np.logical_and(tim>=r1,tim<=r2)) 
	idx_p2 = np.squeeze(np.logical_and(tim>=r2,tim<=r3))  
	p1 = np.max(array[idx_p1])
	n1 = np.max(array[idx_n1])
	p2 = np.max(array[idx_p2])
	tp1 = tim[idx_p1][array[idx_p1].argmax()]
	tn1 = tim[idx_n1][array[idx_n1].argmax()]
	tp2 = tim[idx_p2][array[idx_p2].argmax()]
	if plot == 'true':          
		lineax = dict(linewidth=1, color='black', linestyle=':')        
		fig = plt.figure(19,figsize=[7,5])
		ax = fig.add_subplot(111, autoscale_on=False, xlim=[r0-100,r3+150], ylim=[0,1.25*np.max([p1,n1,p2])])
		plt.plot(tim,array,'k-',lw=3)
		plt.plot(tp1,p1,'ro')
		plt.plot(tn1,n1,'go')
		plt.plot(tp2,p2,'bo')
		ax.axvline(float(r0),**lineax)
		ax.axvline(float(r1),**lineax)	
		ax.axvline(float(r2),**lineax)
		ax.axvline(float(r3),**lineax)
		ax.axhline(**lineax)
		plt.text(tp1-220,1.2*p1,'P1 = %.2f nAm at %.0f ms'  %(p1,tp1))
		plt.text(tn1-40,1.2*n1,'N1 = %.2f nAm at %.0f ms'  %(n1,tn1))
		plt.text(tn1+40,1.1*p2,'P2 = %.2f nAm at %.0f ms'  %(p2,tp2))
		plt.xlabel('time (ms)',fontsize = 13)
		plt.ylabel('Amplitude',fontsize = 13)
	return [p1,n1,p2,tp1,tn1,tp2]
	
def eeg_det_ptp(array,win_p1,win_n1,win_p2,tim):
	"""  Single-subject Peak-to-peak measurement based on time-windows around the P1 (win_p1),N1 (win_n1) and P2 (win_p2) deflection
		The peak-to-peak value is determined as: PTP = (P1+P2)/2 - N1
		
	Args:
		array:  contains EEG data of multiple subjects (nsub x ntpts)
		win_p1 : window around (average) P1 deflection
		win_n1 : window around (average) N1 deflection
		win_p2 : window around (average) P2 deflection
		tim : timing information, array of timepoints in the eeg channel
		
	Returns:
		ptp_val: peak-to-peak values for each of the subjects in 'array'
		
	"""	
	nsub,npts = array.shape
	idx_p1 =  np.squeeze(np.logical_and(tim>=win_p1[0],tim<=win_p1[1]))
	idx_n1 =  np.squeeze(np.logical_and(tim>=win_n1[0],tim<=win_n1[1]))
	idx_p2 =  np.squeeze(np.logical_and(tim>=win_p2[0],tim<=win_p2[1]))
	ptp_val = np.zeros(nsub)
	for i in range(nsub):
		tmp = array[i,:]
		p1 = np.max(tmp[idx_p1])
		n1 = np.min(tmp[idx_n1])
		p2 = np.max(tmp[idx_p2])
		ptp_val[i] = (p1+p2)/2 - n1
		
	return ptp_val

	
def eeg_det_ptp_avg(array,win_p1,win_n1,win_p2,tim):
	"""  Single-subject Peak-to-peak measurement based on the average within the time-windows around the P1 (win_p1),N1 (win_n1) and P2 (win_p2) deflection
		The peak-to-peak value is determined as: PTP = (P1+P2)/2 - N1
		
	Args:
		array:  contains EEG data of multiple subjects (nsub x ntpts)
		win_p1 : window around (average) P1 deflection
		win_n1 : window around (average) N1 deflection
		win_p2 : window around (average) P2 deflection
		tim : timing information, array of timepoints in the eeg channel
		
	Returns:
		ptp_val: peak-to-peak values for each of the subjects in 'array'
		
	"""	
	nsub,npts = array.shape
	idx_p1 =  np.squeeze(np.logical_and(tim>=win_p1[0],tim<=win_p1[1]))
	idx_n1 =  np.squeeze(np.logical_and(tim>=win_n1[0],tim<=win_n1[1]))
	idx_p2 =  np.squeeze(np.logical_and(tim>=win_p2[0],tim<=win_p2[1]))
	
	ptp_val = np.zeros(nsub)
	for i in range(nsub):	
		p1 = np.mean(array[i,idx_p1],axis=0)
		n1 = np.mean(array[i,idx_n1],axis=0)	
		p2 = np.mean(array[i,idx_p2],axis=0)
		ptp_val[i] = (p1+p2)/2 - n1
		
	return ptp_val

def eeg_diss(tim,array1,array2,t_index='all'):
	""" Calculate the global dissimilarity between two topographies (number of channels x number of time-points) at a all time-point,
	as explained in Murray et al., Topographic ERP Analysis: A step-by-step tutorial review,Brain Topogr (2008) 20:249-264
	
	Args:    
		tim:  timing information, array of timepoints in the EEG data channels
		array1 :  contains EEG data of multiple channels (nchan x ntpts) for condition 1
		array2 :  contains EEG data of multiple channels (nchan x ntpts) for condition 2
		t_index : either 'all' or a vector indicating the starting point and end point over which to average the dissimilarity index
		
		
	Returns:
		diss: global dissimilarity index at time-point or time-window t_diss 
		
	"""	
	# first, create scaled array (i.e. for each time-point, divide the value by its instantaneous rms value to get unitary strength)
	v1 = array1/eeg_rms(array1)
	v2 = array2/eeg_rms(array2)
	# second, calculate the square root of the mean of the squared differences between the potentials measured at each electrode (see p.255)
	if t_index == 'all':		
		diss = np.sqrt(np.mean((v1-v2)**2,axis=0))
	else:
		idx = np.logical_and(tim>=t_index[0],tim<=t_index[1])
		t1 = np.mean(v1[:,idx],axis=1)
		t2 = np.mean(v2[:,idx],axis=1)		
		diss = np.sqrt(np.mean((t1-t2)**2,axis=0))
		
	return diss	

def eeg_diss_t(array1,array2):
	""" Calculate the global dissimilarity between two topographies (number of channels x number of time-points) at a all time-point,
	as explained in Murray et al., Topographic ERP Analysis: A step-by-step tutorial review,Brain Topogr (2008) 20:249-264
	
	Args:    
		tim:  timing information, array of timepoints in the EEG data channels
		array1 :  contains EEG data of multiple channels (nchan x ntpts) for condition 1
		array2 :  contains EEG data of multiple channels (nchan x ntpts) for condition 2
		t_index : either 'all' or a vector indicating the starting point and end point over which to average the dissimilarity index
		
		
	Returns:
		diss: global dissimilarity index at time-point or time-window t_diss 
		
	"""	
	# first, create scaled array (i.e. for each time-point, divide the value by its instantaneous rms value to get unitary strength)
	v1 = array1/eeg_rms(array1)
	v2 = array2/eeg_rms(array2)	
	diss = np.sqrt(np.mean((v1-v2)**2,axis=0))
	return diss	

def eeg_load_all_data(freq):	
	resdir = '/Users/crislanting/Projects/EEG/data/FrequencyTuning/results'
	#resdir = '/mnt/Homedrive/projects/EEG/data/FrequencyTuning/results'

	f = str(freq) + 'Hz'
	x_range=[-100,400]
	y_range=[-30,30]
	y_rangerms = [0,1.5]
	
	#general information
	frequency = {
		'500Hz' : 500,
		'1000Hz' : 1000,
		'2000Hz' : 2000,
		'4000Hz' : 4000
		}
	files ={
		'500Hz' : ['*_s1.swf','*_s2.swf','*_s3.swf','*_s4.swf','*_s5.swf'],
		'1000Hz' : ['*_s6.swf','*_s7.swf','*_s8.swf','*_s9.swf','*_s10.swf'],
		'2000Hz' : ['*_s11.swf','*_s12.swf','*_s13.swf','*_s14.swf','*_s15.swf'],
		'4000Hz' : ['*_s16.swf','*_s17.swf','*_s18.swf','*_s19.swf','*_s20.swf']
		}		
	files_avr ={
		'500Hz' : ['*_s1.avr','*_s2.avr','*_s3.avr','*_s4.avr','*_s5.avr'],
		'1000Hz' : ['*_s6.avr','*_s7.avr','*_s8.avr','*_s9.avr','*_s10.avr'],
		'2000Hz' : ['*_s11.avr','*_s12.avr','*_s13.avr','*_s14.avr','*_s15.avr'],
		'4000Hz' : ['*_s16.avr','*_s17.avr','*_s18.avr','*_s19.avr','*_s20.avr']
		}
		
	swfdir = op.join(resdir,'swf/', str(frequency[f])+'Hz/SWF')
	avrdir = op.join(resdir,'swf/', str(frequency[f])+'Hz/Ind')

	tmp = glob.glob1(swfdir,files[f][0])
	tmpavr = glob.glob1(avrdir,files_avr[f][0])
	[eeg,tim,ntpts] = eeg_readswf(op.join(swfdir,tmp[0])) #initialise some values
	[avr,tim,nchan,ntpts] = eeg_readavr(op.join(avrdir,tmpavr[0]))

	t = tim-2000
	t = t[0:ntpts]
		
	# SWF data
	data = np.zeros((len(files[f]),len(tmp),ntpts))
	# Original (AVR) data 
	data_rms = np.zeros((len(files[f]),len(tmp),ntpts))
		
	#Separate sources in left and right hemisphere	
	#data_left = np.zeros((len(files[f]),len(tmp),ntpts))
	#data_right = np.zeros((len(files[f]),len(tmp),ntpts))
	
	for i in range(len(files[f])):
		filelist = glob.glob1(swfdir,files[f][i])
		filelistavr = glob.glob1(avrdir,files_avr[f][i])	
		#sort 
		filelist.sort()
		filelistavr.sort()	
		for j in range(len(filelist)):
			[eeg,tim,ntpts] = eeg_readswf(op.join(swfdir,filelist[j]))
			data[i,j,0:ntpts-1]=np.mean(eeg[:,0:ntpts-1],axis=0)
			[avr,tim,nchan,ntpts] = eeg_readavr(op.join(avrdir,filelistavr[j]))
			data_rms[i,j,0:ntpts-1] = eeg_rms(avr[:,0:ntpts-1])
			#data_left[i,j,0:ntpts-1]=eeg[0,0:ntpts-1]
			#data_right[i,j,0:ntpts-1]=eeg[1,0:ntpts-1]
		cnt = 100.0*i/len(files[f])
		sys.stdout.write("\r")
		sys.stdout.write("progress reading files: \r%2d%%" %cnt)
		sys.stdout.flush()
	
	return t,data,data_rms	
	

def eeg_twosample_ttest(array1,array2):
	"""  Two-sample t-test comparing the values of two EEG data-sets 
		
	Args:
		array1 :  contains EEG data of multiple subjects (nsub x ntpts)
		array1 :  contains EEG data of the same multiple subjects (nsub x ntpts) but from a different condition		
		
	Returns:
		t : t-values, one for each of the timepoints
		p : p-values, also one for each of the timepoints
		
	Dependence:
		scipy.stats.ttest_rel
		
	"""	
	from scipy.stats import ttest_rel
	s1 = array1.shape
	p = np.zeros(s1[1])
	t = np.zeros(s1[1])
	for i in range(s1[1]):
		tval,pval = ttest_rel(array1[:,i],array2[:,i])
		p[i]=pval
		t[i]=tval
		
	return t,p

def eeg_permute(array1, array2, nperm):          
	""" Comparison of the EEG data of two different condition using permutation testing
		
	Args:
		array1 :  contains EEG data of multiple subjects (nsub x ntpts)
		array1 :  contains EEG data of the same multiple subjects (nsub x ntpts) but from a different condition		
		
	Returns:
		actdif : the actual difference between the average values of the two arrays [ntpts]
		p : significance value (exact),  one for each of the timepoints, that the two array differ
	
	"""	
	s1 = array1.shape                
	# create random permutations of indices (k samples x nperm permuations)
	ss = np.concatenate((np.zeros(s1[0]),np.ones(s1[0]))) 
	perm = np.zeros([2*s1[0],nperm])
	perm[:,0] = ss    #actual group indices: zeros belong to array1 and ones to array2     
	for i in range(1,nperm):
		perm[:,i]=np.random.permutation(ss)
	p = np.zeros(s1[1])
	actdif = np.zeros(s1[1])
	for i in range(s1[1]):
		tmp = np.concatenate((array1[:,i],array2[:,i]))
		act_difference = np.mean(tmp[perm[:,0]==1]) - np.mean(tmp[perm[:,0]==0])
		actdif[i] = act_difference
		diff=np.zeros([1,nperm-1])
		for j in range(1,nperm):
			diff[0,j-1] = np.mean(tmp[perm[:,j]==1])-np.mean(tmp[perm[:,j]==0])

		p[i] = np.sum(np.abs(diff) > np.abs(act_difference))/(nperm-1.0)
		cnt = 100.0*i/s1[1]
		sys.stdout.write("progress: \r%d%%" %cnt)
	
	return actdif,p

def eeg_ptp(array,tim,window):
        # maybe eeg_peaks is more concise and includes P1
	rmin,rmax = window[0],window[1]	
	idx = np.squeeze(np.logical_and(tim>=rmin,tim<=rmax))
	arr = array[idx]
	tim2 = tim[idx]
	idxmin = arr.argmin(axis=0)
	idxmax = arr.argmax(axis=0)	
	minval = arr[idxmin]
	t_min = tim2[idxmin]	
	maxval = arr[idxmax]
	t_max = tim2[idxmax]	
	ptpval = maxval-minval
	return [ptpval, minval, t_min, maxval, t_max]


def eeg_readavr(file):
	""" Read EEG data (AVR files) and return the data
		
	Args:
		file :  filename of the datafile 
		
	Returns:
		eeg : EEG data (shape of data array: number of channels [nchan] x number of timepoints [ntpts])
		tim:  timing information, array of timepoints in the eeg channel
		nchan : number of data-channels in the data
		ntpts : number of timepoints
	
	"""	
	f=open(file,'r')	
	firstline = f.readline() # ntpts TSB info etc
	str = string.split(firstline)
	ntpts = int(str[1])
	nchan = int(str[11])
	tsb = float(str[3])
	di = float(str[5])	
	tim = np.arange(tsb,ntpts*di+tsb,di)
	secondline = f.readline()
	chnam = string.split(secondline)
	eeg = np.zeros([nchan,ntpts])		
	for i in range(0,nchan):
		testline = f.readline()
		testline = testline.strip().split()		
		eeg[i,:]=np.array(map(float,testline))
		
	f.close()
	return eeg,tim,nchan,ntpts
	
def eeg_readelp(file):
	""" Read EEG ELP file and return the data
	
	BESA-'.elp' spherical coordinates:
    The elevation angle (phi) is measured from the vertical axis. Positive 
    rotation is toward right ear. Next, perform azimuthal/horizontal rotation 
    (theta): 0 is toward right ear; 90 is toward nose, -90 toward occiput. 
    Angles are in degrees.
    
    Azimuth Theta
    Elevation Phi
    Radius R
    
    Args:
		file :  filename of the ELP file 
		
	Returns:
		
	
	"""	
	
	f = open(file,'r')
	phi = np.zeros(33)
	theta = np.zeros(33)
	r = np.ones(33)
	chan = []
	for i in range(33):
		line=f.readline()
		str = string.split(line)
		#chan[i]=str[1]
		phi[i]=float(str[2])
		theta[i]=float(str[3])
		chan.append(str[1])
	
	f.close()
	# theta and phi are in degrees -> convert to radians
	x,y,z = eeg_sph2cart(theta*pi/180,phi*pi/180,r)
	return chan,x,y,z,theta,phi
	
def eeg_sph2cart(az,el,r):
	# transform spherical to Cartesian coordinates
	# azimuth (theta), elevation (phi), radius (r)
	# all angles in radians
	z = r * np.sin(el)
	rc = r * np.cos(el)
	x = rc * np.cos(az)
	y = rc * np.sin(az)
	return x,y,z
	
	
	

def eeg_readswf(file):
	""" Reads BESA EEG source waveform files (SWF files) and returns the data
		Assumed two sources, one for the left and one for the right auditory cortex  (TE1.0, see Morosan et al. 2001) 
		
	Args:
		file :  filename of the datafile 
		
	Returns:
		eeg : EEG data (shape of data array: number of channels [nchan] x number of timepoints [ntpts])
		tim:  timing information, array of timepoints in the eeg channel
		ntpts : number of timepoints
		
	"""		
	f=open(file,'r')	
	firstline = f.readline() # ntpts TSB info etc
	str = string.split(firstline)
	ntpts = int(str[1])	
	tsb = float(str[3])
	di = float(str[5])	
	tim = np.arange(tsb,ntpts*di+tsb,di)	
	line = f.readline()	
	str = string.split(line)
	eeg0 = np.array(map(float,str[1:]))
	line = f.readline()	
	str = string.split(line)
	eeg1 = np.array(map(float,str[1:]))
	eeg = np.zeros([2,ntpts])
	eeg[0,:]=eeg0
	eeg[1,:]=eeg1
	return [eeg,tim,ntpts]

def eeg_rms(array, axis=0):
	""" Calculate RMS value of EEG dataset (over all channels)
	This is the same as calculating the global field power (GFP) as explained in Murray et al., Topographic ERP Analysis: A step-by-step tutorial review,
	Brain Topogr (2008) 20:249-264
		
	Args:    
		array :  contains EEG data of multiple channels (nchan x ntpts) from e.g. an AVR file
		
	Returns:
		rms value for each of the timepoints in the EEG data (ntpts)
		
	"""		
	return np.sqrt(np.mean(array ** 2,axis))
	


def eeg_writeavr(array,tsb,di,file):
	""" Writes an AVR file 	
		
	Args:
		array : EEG data containing the data of 32 EEG Channels (see variable chnam for naming convention)
		tsb: first timepoint (e.g., -100 ms)
		di: time between succesive samples (e.g. 4 ms)
		file :  filename of the datafile 
		
	Returns:
		data file ('file')
		
	"""		
        import shutil as shu
        f=open(file,'w')
        firstline = 'Npts= %i TSB= %i DI= %7.5f SB= %7.5f SC= %i NChan= %i\n' %(array.shape[1],tsb,di,1,200,array.shape[0])   
        chnam = 'Cz FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T7 T8 P7 P8 Fz Pz FC1 FC2 CP1 CP2 FC5 FC6 CP5 CP6 FT9 FT10 TP9 TP10 PO9 PO10\n'
        f.write(firstline)
        f.write(chnam)
        for i in range(array.shape[0]):
                tmp = array[i,:]
                f.write(('%7.5f ' * len(tmp)) %tuple(tmp))
                f.write('\n')
                
        f.close()
        #may want to change this on different machines...
        src = '/Users/crislanting/Projects/EEG/data/33.elp'
        dest = file[:-4] + '.elp'
        shu.copyfile(src,dest)
              
	
# various stats functions 
#======================

def stats_holmBonf(p,plot='false'):
	"""  Holm-Bonferroni method to control the familywise error rate at a (global) level of p = 0.05
		http://en.wikipedia.org/wiki/Holm-Bonferroni_method
		
	Args:
		p : array of p-values
		plot = 'true' to get plots of sorted p-values and the critical p-values according to the Holm-Bonferroni method
		
	Returns:
		h : array of booleans (False: reject the null hypothesis; True: accept null hypothesis
		corrected_p : (sorted) critical p-values [alpha/k, alpha/(k-1), alpha/(k-2), ..., alpha/2, alpha] 
		p_holm : critical p-value of the first element for which you cannot reject the jull hypothesis anymore
				
	"""	
	alpha = 0.05
	k = len(p)
	idx = [i[0] for i in sorted(enumerate(p), key=lambda x:x[1])]
	corrected_p = alpha/np.arange(k+1,1,-1)
	
	h = p <= corrected_p
	if np.sum(h) > 0 :
		idx_first_zero = np.where(h == 0)[0][0] #get first index where you cannot reject the null-hypothesis
		h[idx_zero:] = 0
	
		if plot == 'true':    
			plt.plot(p,'ko-')
			plt.plot(c,'r-')
			x=np.arange(len(p))
			plt.fill_between(x,0,p,where=h,facecolor='red')
	
		h = h[idx]
		p_holm = c[idx_first_zero]
	else:
		print "None of the p-value exceed the Holm-Bonferroni corrected threshold"
		p_holm = nan
		
	return h,corrected_p,p_holm
	
#various functions (maybe obsolete)
#======================

def my_errorbar(x,y,yerr,w):
	for i in range(len(x)):
		plt.plot([x[i],x[i]],[y[i]-yerr[i],y[i]+yerr[i]],lw=1,c='black',ls='-')
		plt.plot([x[i]-w/2,x[i]+w/2],[y[i]-yerr[i], y[i]-yerr[i]],lw=1,c='black',ls='-')
		plt.plot([x[i]-w/2,x[i]+w/2],[y[i]+yerr[i], y[i]+yerr[i]],lw=1,c='black',ls='-')
	
def roex_fit(p,g):
	return (1+p*g)*np.exp(-p*g)
	
def residuals(y,p,g):
	err = y-roex_fit(p,g)
	return err

def gauss(p, x):
	A, mu, sigma = p
	return A*np.exp(-(x-mu)**2/(2*sigma**2))


