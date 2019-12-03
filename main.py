import os
import sys
manTraNet_root = '/home/iftekhar/project2ece278/ManTraNet-master/'
manTraNet_srcDir = os.path.join( manTraNet_root, 'src' )
sys.path.insert( 0, manTraNet_srcDir )
manTraNet_modelDir = os.path.join( manTraNet_root, 'pretrained_weights' )
manTraNet_dataDir = os.path.join( manTraNet_root, 'data' )
sample_file = os.path.join( manTraNet_dataDir, 'samplePairs.csv' )
import numpy as np 
import cv2
import requests
from datetime import datetime 
import modelCore
from PIL import Image
from io import BytesIO
from matplotlib import pyplot
#from get_model_size import check_size
import tensorflow as tf
import numpy as np  
from keras.backend.tensorflow_backend import set_session
import pickle
import math
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3 
set_session(tf.Session(config=config))

with open( sample_file ) as IN :
    sample_pairs = [line.strip().split(',') for line in IN.readlines() ]
L = len(sample_pairs)
print("INFO: in total, load", L, "samples")

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
    if xrange is None and yrange is None :
        pyplot.figure( figsize=(15,5) )
        pyplot.title('Original Image')
        pyplot.subplot(131)
        pyplot.imshow( big_img )
        pyplot.title('Forged Image (ManTra-Net Input)')
        pyplot.subplot(132)
        #pyplot.imshow( mask, cmap='gray' )
        pyplot.imshow( big_res, cmap='gray' )
        pyplot.title('Predicted Mask (ManTra-Net Output)')
        pyplot.subplot(133)
        #pyplot.imshow( np.round(np.expand_dims(mask,axis=-1) * img).astype('uint8'), cmap='jet' )
        pyplot.imshow( np.round(np.expand_dims(big_res,axis=-1) * big_img).astype('uint8'), cmap='jet' )
        pyplot.title('Highlighted Forged Regions')
        #pyplot.suptitle('Decoded {} of size {} for {:.2f} seconds'.format( url, rgb.shape, ptime ) )
        pyplot.show()
    return big_res[0:size[0],0:size[1]],big_img
all_res={}
for name in range(1,11):
    file_name='test_images/'+str(name)+'.jpg'

    #list = os.listdir(dir) # dir is your directory path
    #number_files = len(list)
    mask,img=get_image_from_url(file_name)
    all_res[name]=[mask,img]
    print('{} image is done'.format(str(name)))
#with open('res.pickle','wb')as fp:pickle.dump(all_res,fp,protocol=pickle.HIGHEST_PROTOCOL)
import pdb;pdb.set_trace()
