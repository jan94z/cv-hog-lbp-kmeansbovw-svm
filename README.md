# cv-hog-lbp-kmeansbovw-svm 
 _Computer Vision - Histogram of Oriented Gradients - Local Binary Patterns - K-Means Bag of Visual Words - Support Vector Machine_
## Objective
This is a project of mine to practice and reiterate traditional computer vision techniques as well as some Python functionalities, that I picked up in a course at the University of Bonn. A classifier is built to distinguish the leaves of five different grapevine varieties, using the data of Koklu et al [^reference]. The Histogram of Oriented Gradients (HOG) and the Local Binary Patterns (LBP) are applied to extract features from the image. After extracting, the HOG features are clustered by the K-Means algorithm. Then, the final feature vector is created by stacking the histogram of cluster occurences and the histogram of the LBP features. Finally, this feature vector is used by the linear support vector machine (SVM) to classify the image.
## Usage
Created on Ubuntu 22.04 and Python 3.10.6.
### Data
Download the data [here](https://www.muratkoklu.com/datasets/vtdhnd10.php) and put it into the data folder in the following structure: <br>
<pre>
|--data
   |--Ak
   |--Ala_Idris
   |--Buzgulu
   |--Dimnit
   |--Nazli
</pre>
### Initialize virtual environment and install packages
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```
### Prepare the data
If you want to change the data split or change the seed: [prepare_data.py](../master/scripts/prepare_data.py)
```bash
python3 main.py --d
```
### Train the model
If you want to change the hyperparameters: [main.py](../master/main.py)
```bash
python3 main.py --t
```
You will be shown the results on the validation data and asked if you want to save the model and how to name it if you choose so.
### Classify an image
[model] = the name of the saved model <br>
[img] = the path of the image to be classified
```bash
python3 main.py --i [model] [img]
```
The repository comes with one trained model named 'example'.

[^reference]: Koklu, M., Unlersen, M. F., Ozkan, I. A., Aslan, M. F., & Sabanci, K. (2022). A CNN-SVM study based on selected deep features for grapevine leaves classification. Measurement, 188, 110425. Doi:https://doi.org/10.1016/j.measurement.2021.110425
