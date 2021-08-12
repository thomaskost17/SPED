'''
  File: sparse_representation.py
 
  Author: Thomas Kost
  
  Date: 08 August 2021
  
  @breif represent images in a sparse manner using SVD and perform classification (use mnist)
 '''
import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from PIL import Image

def downsample_img(img:np.array, n :int, m:int )->np.array:
  return np.array(Image.fromarray(img).resize(size=(n,m), resample=Image.LANCZOS))

def corrupt_img(img:np.array, p:float, style:str = "erasure")->np.array:
  if(style == "erasure"):
    return img*(np.random.rand(img.shape[0])>p)
  elif(style == "static"):
    max_error = int(p*255/2)
    img = img + np.random.randint(-max_error,max_error, size=img.shape())
    return np.clip(img,0,255)
  elif(style == "mixed"):
    pass
  else:
    raise ValueError("unrecognized type of noise")

def classify(theta:np.array, y:np.array, epsilon:float, debug:bool =False)->np.array:
  # setup LP
  s = cp.Variable(theta.shape[1])
  objective = cp.Minimize(cp.norm(s,1))
  constraints = [cp.norm(y-theta@s,2) <= epsilon]
  prob = cp.Problem(objective, constraints)
  prob.solve(solver=cp.ECOS, max_iters=200, verbose=debug)
  return s.value


if __name__ == "__main__":

  # Import datasets
  train_dat = pd.read_csv('../fixtures/mnist_train.csv')
  test_dat  = pd.read_csv('../fixtures/mnist_test.csv')
  train_dat = train_dat.sort_values(by='label') 
  
  # Convert from pd dataframe to numpy array
  train_dat = train_dat.to_numpy() #overly tailored basis
  test_dat = test_dat.to_numpy()

  # seperate labels and data
  training_data_only = train_dat[:,1:]
  training_labels_only = train_dat[:,0]
  testing_data_only = test_dat[:,1:]
  testing_labels_only = test_dat[:,0]

  # Downsample
  dim = 28
  ds_factor = 2
  ds_training_data_only = np.zeros((training_data_only.shape[0],int(pow(dim,2)/pow(ds_factor,2))))
  for i,img in enumerate(training_data_only):
    ds_training_data_only[i,:] = downsample_img(np.reshape(img.astype(np.uint8),(dim,dim)),int(dim/2),int(dim/2)).flatten()
    
  index = np.random.randint(0,len(testing_labels_only))

  # Pick Image
  test_img = testing_data_only[index,:]
  # Corrupt Image
  p = 0.5
  corrupt_test_img = corrupt_img(test_img,p)
  print(testing_labels_only[index])

  # Show Image
  fig, ax = plt.subplots(2,3)
  ax[0,0].imshow(np.reshape(test_img,(dim,dim)),cmap='gray', vmin=0, vmax=255)
  ax[0,1].imshow(np.reshape(corrupt_test_img,(dim,dim)),cmap='gray', vmin=0, vmax=255)
  # Downsample Image
  ds_corr_test_img = downsample_img(np.reshape(corrupt_test_img.astype(np.uint8),(dim,dim)),int(dim/2),int(dim/2))
  ax[0,2].imshow(np.reshape(ds_corr_test_img,(int(dim/ds_factor),int(dim/ds_factor))),cmap='gray', vmin=0, vmax=255)

  # Find Sparse Vector
  s = classify(np.transpose(ds_training_data_only),ds_corr_test_img.flatten(),0.1, False)
  ax[1,0].plot(s)
  # get recreation
  rec = np.transpose(training_data_only)@s
  ax[1,1].imshow(np.reshape(rec,(int(dim),int(dim))),cmap='gray', vmin=0, vmax=255)

  # Get sparse errors
  err = test_img-rec
  ax[1,2].imshow(np.reshape(err,(int(dim),int(dim))),cmap='gray', vmin=0, vmax=255)
  plt.show()


