import argparse
import pickle
#from scipy import fftpack
from scipy import fftpack, ndimage 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import scipy.io
from PIL import Image
from past.builtins import xrange
import itertools
import multiprocessing
import pdb
import sys
import imutils
from tqdm import tqdm
import os
import errno
import math


base=50
scaling_factor=10000*500
increment=25
threshold=50
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def cal_dist(params):
	a=np.asarray(params[0])
	b=np.asarray(params[1])
	#ForkedPdb().set_trace()
	#print(type(a))
	dist=np.linalg.norm(a-b)
	#print(threshold)
	if abs(dist) < threshold :
		return True
	else:
		return False
	 
def find_if_close(cnt1,cnt2,num_core):
	row1,row2 = cnt1.shape[0],cnt2.shape[0]
	#import pdb;pdb.set_trace()
	paramlist=list(itertools.product(cnt1,cnt2))
	print('Starting distance calculation for two contours')
	max_iter=len(paramlist)
	index=0
	dist=False
	#import pdb;pdb.set_trace() 
	pool = multiprocessing.Pool(num_core)
	res=pool.map(cal_dist,paramlist)
	pool.close()
	return any(res)
	#ForkedPdb().set_trace()



def combine_contours(al,dest_name,num_core):
	
	for id in tqdm(al):
		#import pdb;pdb.set_trace()
		global threshold
		second=al[id][0]
		scaled=(second.shape[0]*second.shape[1])/scaling_factor
		threshold=math.ceil(scaled)*increment+base	 
		im=np.array(second*255,dtype='uint8')
		import pdb;pdb.set_trace()
		morph = im.copy()

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
		morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
		
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
		gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

		image_channels = gradient_image



		_, image_channels = cv2.threshold(~image_channels, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
		contours_p, heirarchy = cv2.findContours(~image_channels, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnts = sorted(contours_p, key = cv2.contourArea, reverse = True)
		contours=cnts[0:25]
		LENGTH = len(contours)
		status = np.zeros((LENGTH,1))
		for i,cnt1 in enumerate(contours):
			x = i    
			if i != LENGTH-1:
				for j,cnt2 in enumerate(contours[i+1:]):
					x = x+1
					dist = find_if_close(cnt1,cnt2,num_core)
					print('This Pair is Done')				
					if dist == True:
						val = min(status[i],status[x])
						status[x] = status[i] = val
					else:
						if status[x]==status[i]:
							status[x] = i+1
				print('End of calculation for contour {} of total {} contours'.format(i+1,len(contours)))
		print('All distances are measured; its time for drawing combined regions')
		#import pdb;pdb.set_trace()
		unified = []
		maximum = int(status.max())+1
		for i in xrange(maximum):
			pos = np.where(status==i)[0]
			if pos.size != 0:
				cont = np.vstack(contours[i] for i in pos)
				hull = cv2.convexHull(cont)
				unified.append(hull)
		print('Combining regions is done')
		meta=dest_name+'/'+id+'.pickle'
		with open(meta,'wb')as fp:pickle.dump([unified,contours],fp,protocol=pickle.HIGHEST_PROTOCOL)
		#import pdb;pdb.set_trace()
		print('This image is done')	
	

if __name__ == '__main__':

	parser=argparse.ArgumentParser()
	parser.add_argument('-l','--location',nargs='+',required=True,help='Path to the pickle file which contains ManTranet output')
	parser.add_argument('-d','--destination',nargs='+',required=True,help='Name of the folder to store all the filtered masks')
	parser.add_argument('-n','--num_cores',type=int,required=False,default=80,help='Number of cores to use for multiprocessing, deafault value is set to 80')
	args=parser.parse_args()
	path=args.location[0]+'.pickle'
	folder_name=args.destination[0]
	num_cores=args.num_cores
	with open(path,'rb')as fp:
		al=pickle.load(fp) 
	try:
		os.mkdir(folder_name)
	except OSError as exc:
		if exc.errno != errno.EEXIST:
			raise
		pass
	
	combine_contours(al,folder_name,num_cores)

import pdb;pdb.set_trace()


