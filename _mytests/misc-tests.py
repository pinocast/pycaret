# Importing dataset
from pycaret.datasets import get_data
diabetes = get_data('diabetes')
# Importing module and initializing setup
from pycaret.classification import *
clf1 = setup(data = diabetes, target = 'Class variable')
# train logistic regression model
lr = create_model('lightgbm') #lr is the id of the model