import os 
from sklearn.model_selection import train_test_split
import json
import re

def dataprep():
  path = os.getcwd()
  datapath = path + '/data'

  subdir = [f.path for f in os.scandir(datapath) if f.is_dir()]

  # rename files as spaces and round brackets cause problems in the terminal
  for folder in subdir:
      for file in os.listdir(folder):
        old = os.path.join(folder, file)
        new = re.sub("[\s\(\)]", "", old)
        os.rename(old, new)

  # get files from the folders
  filedict = {p: [os.path.join(p, f) for f in os.listdir(p)] for p in subdir}

  # create lists for files and labels and dict to store classnames and corresponding numbers
  files = []
  labels = []
  classes = {}
  for idx,(k, v) in enumerate(filedict.items()):
    files.extend(v)
    labels.extend([idx]*len(v))
    classes[idx] = k.lstrip(datapath)

  # split 80/10/10
  seed = 0
  X_train, X, y_train, y = train_test_split(files, labels, test_size=0.2, stratify=labels, random_state=seed)
  X_valid, X_eval, y_valid, y_eval = train_test_split(X, y, test_size=0.5, stratify=y, random_state=seed)

  # dump
  data = {
    'X_train': X_train,
    'y_train': y_train,
    'X_valid': X_valid,
    'y_valid': y_valid,
    'X_eval': X_eval,
    'y_eval': y_eval,
    'classes': classes
  }
  with open(f'{datapath}/data.json', 'w') as fp:
    json.dump(data, fp)
  


















