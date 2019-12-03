import os
import sys
import numpy as np 
import cv2
import requests
from datetime import datetime 
from PIL import Image
from io import BytesIO
from matplotlib import pyplot
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import pickle
import math
import glob
import argparse




manTraNet_root = './'
manTraNet_srcDir = os.path.join( manTraNet_root, 'src' )
sys.path.insert( 0, manTraNet_srcDir )
manTraNet_modelDir = os.path.join( manTraNet_root, 'pretrained_weights' )
manTraNet_dataDir = os.path.join( manTraNet_root, 'data' )
sample_file = os.path.join( manTraNet_dataDir, 'samplePairs.csv' )
import modelCore
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'




def get_a_random_pair() :
    idx = np.random.randint(0,L)
    return ( os.path.join( manTraNet_dataDir, this ) for this in sample_pairs[idx] )

manTraNet = modelCore.load_pretrain_model_by_index( 4, manTraNet_modelDir )
def read_rgb_image( image_file ) :
    rgb = cv2.imread( image_file, 1 )[...,::-1]
    return rgb

def decode_an_image_array( rgb, manTraNet ) :
    x = np.expand_dims( rgb.astype('float32')/255.*2-1, axis=0 )
    t0 = datetime.now()
    y = manTraNet.predict(x)[0,...,0]
    t1 = datetime.now()
    return y, t1-t0

def decode_an_image_file( image_file, manTraNet ) :
    rgb = read_rgb_image( image_file )
    mask, ptime = decode_an_image_array( rgb, manTraNet )
    return rgb, mask, ptime.total_seconds()
def get_image_from_url(url, xrange=None, yrange=None) :
    #response = requests.get(url)
    #img = Image.open(BytesIO(response.content))
    img = Image.open(url)
    #import pdb;pdb.set_trace()
    img=img
    img = np.array(img)
    size=img.shape
    known=400
    dsize=(known,int(known*(size[1]/size[0])))

    #img=img.resize(dsize)
    if img.shape[-1] > 3 :
        img = img[...,:3]
    ori = np.array(img)
    if xrange is not None :
        img = img[:,xrange[0]:xrange[1]]
    if yrange is not None :
        img = img[yrange[0]:yrange[1]]
    #big_img=np.pad(img, ((0,dsize[0]-size[0]%dsize[0]),(0,dsize[1]-size[1]%dsize[1]),(0,0)), 'constant', constant_values=(0))
    big_img=img
    #mask, ptime =  decode_an_image_array( img, manTraNet )
    #row_order=int(big_img.shape[0]/dsize[0])
    row_order=math.ceil(big_img.shape[0]/dsize[0])
    #col_order=int(big_img.shape[1]/dsize[1])
    col_order=math.ceil(big_img.shape[1]/dsize[1])
    row_indices=0
    big_res=np.zeros((big_img.shape[0],big_img.shape[1]))
    ov_fact=0.9
    #for i in range(row_order):
    while row_order>0:
        col_indices=0
        #for k in range(col_order):
        while col_order>0:
            #try:
            block=big_img[row_indices:dsize[0]+row_indices,col_indices:dsize[1]+col_indices]
                #import pdb;pdb.set_trace()
            mask, ptime =  decode_an_image_array( block, manTraNet )
            big_res[row_indices:dsize[0]+row_indices,col_indices:dsize[1]+col_indices]=mask
            col_indices=int(ov_fact*dsize[1])+col_indices
            if col_indices>=size[1]:
                 break

        
        row_indices=int(ov_fact*dsize[0])+row_indices
        if row_indices>=size[0]:
             break
             
    #import pdb;pdb.set_trace()
            
    ptime = ptime.total_seconds()
    # show results


  #  if xrange is None and yrange is None :
  #      pyplot.figure( figsize=(15,5) )
  #      #pyplot.title('Original Image')
  #      pyplot.subplot(131)
  #      pyplot.imshow( big_img )
  #      pyplot.title('ManTra-Net Input')
  #      pyplot.subplot(132)
  #      #pyplot.imshow( mask, cmap='gray' )
  #      pyplot.imshow( big_res, cmap='gray' )
  #      pyplot.title('Predicted Mask (ManTra-Net Output)')
  #      pyplot.subplot(133)
  #      #pyplot.imshow( np.round(np.expand_dims(mask,axis=-1) * img).astype('uint8'), cmap='jet' )
  #      pyplot.imshow( np.round(np.expand_dims(big_res,axis=-1) * big_img).astype('uint8'), cmap='jet' )
  #      pyplot.title('Highlighted Forged Regions')
  #      #pyplot.suptitle('Decoded {} of size {} for {:.2f} seconds'.format( url, rgb.shape, ptime ) )
  #      pyplot.show()
    return big_res,big_img



all_res={}



if __name__ == '__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-l','--location',nargs='+',required=True,help='Path to the folder of the images to be analysed, expecting only .jpg images in that folder')
	parser.add_argument('-d','--destination',nargs='+',required=True,help='Name of the file to store')
	args=parser.parse_args()
	path=args.location
	dest=args.destination
	fullpath=path[0]+'/*.jpg'
	dest_name=dest[0]+'.pickle'
	for index,filename in enumerate(glob.glob(fullpath)):
		key=filename.split('/')[-1].split('.')[0]
		#import pdb;pdb.set_trace()
		#file_name='test_images/'+str(name)+'.jpg'
		mask,img=get_image_from_url(filename)
		all_res[key]=[mask,img]
		print('Image no {} is done'.format(str(index+1)))
with open(dest_name,'wb')as fp:pickle.dump(all_res,fp,protocol=pickle.HIGHEST_PROTOCOL)
import pdb;pdb.set_trace()
