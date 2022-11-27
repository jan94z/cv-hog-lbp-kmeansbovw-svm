import pickle
import json
import argparse
import numpy as np
import pandas as pd
from scripts import utils
from scripts.prepare_data import dataprep
from skimage.io import imread
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

def training():
  ### get data
  with open('./data/data.json', 'r') as fp:
    data = json.load(fp)
  trainloader = utils.imageloader(data['X_train'], data['y_train'])
  validloader = utils.imageloader(data['X_valid'], data['y_valid'])

  ### set hyperparameters
  # kmeans
  n_clusters = 5
  k_maxiter = 1000
  # hog
  orient = 8
  ppc = (8,8)
  cpb = (1,1)
  # lbp
  radius = 1
  npoints = 8
  nbins = 128
  range_bins = (0,255)
  # svm
  C = 0.1
  svm_maxiter = 1000

  ### create feature vector of hog features for bovw
  print('Extracting HOG features...')
  first = True
  for img, _ in tqdm(trainloader):
    hog = utils.HOG(img, orient=orient, ppc=ppc, cpb=cpb)
    if first:
      X_train_hog = hog
      first = False
    else:
      X_train_hog = np.vstack((X_train_hog, hog))

  ### train bovw with hog features
  bovw = utils.kmeans_bovw(n_clusters=n_clusters, max_iter=k_maxiter)
  print('Training Kmeans BOVW...')
  bovw.fit(X_train_hog)
  
  ### create training data for classifier
  print('Transforming training data...')
  X_train, y_train = utils.fv(loader=trainloader, bovw=bovw, orient=orient, ppc=ppc, cpb=cpb,
  radius=radius, npoints=npoints, nbins=nbins, range_bins=range_bins)
  # normalize training data
  normalizer = StandardScaler()
  X_train = normalizer.fit_transform(X_train)

  ### train classifier
  svm = LinearSVC(C=C, max_iter=svm_maxiter)
  print('Training SVM...')
  svm.fit(X_train, y_train)

  ### create validation data
  print('Transforming validation data...')
  X_valid, y_valid = utils.fv(loader=validloader, bovw=bovw, orient=orient, ppc=ppc, cpb=cpb,
  radius=radius, npoints=npoints, nbins=nbins, range_bins=range_bins)
  # normalize validation data
  X_valid = normalizer.transform(X_valid)

  ### validation
  y_pred_valid = svm.predict(X_valid)
  cm_valid = confusion_matrix(y_true=y_valid, y_pred=y_pred_valid, normalize='true').round(2)
  acc_valid = np.sum(cm_valid.diagonal())/float(cm_valid.diagonal().shape[0])
  print(f'Performance on validation data:\nAccuracy: {acc_valid:.2f}\nConfusion matrix:\n{cm_valid}')

  ### ask for user input
  print('Do you want to save the model? (y/n)')
  input1 = str(input())

  ### if yes -> continue with evaluation and save the model and the corresponding parameters
  if input1 == 'y':
    # ask for the name to save the model
    print('Type in name to save the model')
    input2 = str(input())

    # create evaluation data
    print('Transforming evaluation data...')
    evalloader = utils.imageloader(data['X_eval'], data['y_eval'])
    X_eval, y_eval = utils.fv(loader=evalloader, bovw=bovw, orient=orient, ppc=ppc, cpb=cpb,
    radius=radius, npoints=npoints, nbins=nbins, range_bins=range_bins)
    # normalize evaluation data
    X_eval = normalizer.transform(X_eval)
    # evaluation
    y_pred_eval = svm.predict(X_eval)
    cm_eval = confusion_matrix(y_true=y_eval, y_pred=y_pred_eval, normalize='true').round(2)
    acc_eval = np.sum(cm_eval.diagonal())/float(cm_eval.diagonal().shape[0])
    print(f'Performance on evaluation data:\nAccuracy: {acc_eval:.2f}\nConfusion matrix:\n{cm_eval}')

    # save the model for re-usage
    model = {'orients': orient, 'ppc': ppc, 'cpb': cpb,
    'radius': radius, 'npoints': npoints, 'nbins': nbins, 'range_bins': range_bins,
    'bovw': bovw, 'svm': svm, 'normalizer': normalizer}
    with open(f'./models/{input2}.pickle', 'wb') as fp:
      pickle.dump(model, fp)

    # add hyperparameters and accuracy of every new model to csv file
    stats = {'name': [input2], 'validation_acc': [acc_valid], 'evaluation_acc': [acc_eval],
       'kmeans_clusters': [n_clusters], 'kmeans_maxiter': [k_maxiter], 'hog_orients': [orient],
       'hog_ppc0': [ppc[0]], 'hog_ppc1': [ppc[1]], 'hog_cpb0': [cpb[0]], 'hog_cpb1': [cpb[1]], 
       'lbp_radius': [radius], 'lbp_npoints': [npoints], 'lbp_nbins': [nbins], 'lbp_rangebins0': [range_bins[0]],
       'lbp_rangebins1': [range_bins[1]], 'svm_c': [C], 'svm_maxiter': [svm_maxiter]}
    df = pd.DataFrame(stats)
    df.to_csv('./models/stats.csv', mode='a', index=False, header=True)

  # if no -> stop
  elif input1 == 'n':
    pass
  # if other input than y or n
  else: 
    print('Input must be y or n')

def classification():
  # get model
  with open(f'./models/{parser.i[0]}.pickle', 'rb') as fp:
    model = pickle.load(fp)
  bovw = model['bovw']
  normalizer = model['normalizer']
  svm = model['svm']

  # get classnames
  with open('./data/data.json', 'r') as fp:
    data = json.load(fp)
  classes = data['classes']

  # get and transform the image
  img = imread(parser.i[1])
  img = img[:,:,0:3] # rgba -> rgb
  hog = utils.HOG(img, orient=model['orients'], ppc=model['ppc'], cpb=model['cpb'])
  hog = bovw.predict(hog)
  lbp = utils.LBP(arr=img, radius=model['radius'], npoints=model['npoints'], nbins=model['nbins'], range_bins=model['range_bins'])
  features = np.hstack((hog,lbp))
  features = normalizer.transform(features)

  # classify
  prediction = int(svm.predict(features))
  print(classes[f'{prediction}'])

if __name__ == '__main__':
  # parser to use terminal
  _parser = argparse.ArgumentParser(description='Extracting command line arguments', add_help=True)
  _parser.add_argument('--d', '--dataprep', action='store_const', const=True)
  _parser.add_argument('--t', '--training', action='store_const', const=True)
  _parser.add_argument('--i', '--image', action='store', nargs='+', type=str, help='First argument: Name of the saved model. Second argument: path to the image to be classified')
  parser = _parser.parse_args()

  if parser.d:
    dataprep()
  elif parser.t:
    training()
  elif parser.i:
    classification()
    