import numpy as np
import skimage.io as skio
import skimage.color as skcol
import skimage.feature as skfeat
from sklearn.cluster import KMeans
from tqdm import tqdm

class imageloader():
    def __init__(self, filenames, labels):
        self.filenames = filenames
        self.labels = labels

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        self.img_path = self.filenames[idx]
        rgba_image = skio.imread(self.img_path)
        image = rgba_image[:,:,0:3] # rgba -> rgb
        label = self.labels[idx]
        return image, label

def HOG(arr, orient, ppc, cpb, vis=True, fv=False):
  """
  Takes an image as an input, converts it to grayscale, extracts features using hog and reshapes it to a (m, orientations matrix).
  """
  gray = skcol.rgb2gray(arr)
  feats, map = skfeat.hog(gray,
                  orientations=orient,
                  pixels_per_cell=ppc,
                  cells_per_block=cpb,
                  visualize=vis,
                  feature_vector=fv)
  
  return feats.reshape(-1, orient)

def LBP(arr, radius, npoints, nbins, range_bins):
  """
  Takes an image as an input, converts it to grayscale, extracts lbp features,
  counts them using a histogramm and returns them as a feature vector.
  """
  gray = skcol.rgb2gray(arr)
  features = skfeat.local_binary_pattern(gray,
                                      R=radius,
                                      P=npoints)

  return np.histogram(features, bins=nbins, range=range_bins)[0].reshape(1,-1)

class kmeans_bovw():
  """
  Bag of visual words using Kmeans clustering.
  """
  def __init__(self, n_clusters:int, **params):
    self.n_clusters = n_clusters
    self.params = params

  def fit(self, X):
    self.kmeans = KMeans(self.n_clusters, **self.params)
    self.kmeans.fit(X)

  def predict(self, X):
    return np.histogram(self.kmeans.predict(X), bins=self.n_clusters)[0].reshape(1,-1)

def fv(loader, bovw, orient, ppc, cpb, radius, npoints, nbins, range_bins):
    """
    Specific helper function. Creates the feature vector and the corresponding labels
    for the final classifier from hog and lbp features.
    """
    first = True
    for img, label in tqdm(loader):
        hog = HOG(img, orient=orient, ppc=ppc, cpb=cpb)
        hog = bovw.predict(hog)
        hog = hog.reshape(1,-1)
        lbp = LBP(img, radius=radius, npoints=npoints, nbins=nbins, range_bins=range_bins)
        features = np.hstack((hog, lbp))
        if first:
            X = features
            y = label
            first = False
        else:
            X = np.vstack((X, features))
            y = np.hstack((y, label))
    return X, y







