
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import roc_auc_score, matthews_corrcoef
from os.path import join
import os
import pandas as pd
import re
import warnings
from additional_code.helper_functions import *
CURRENT_DIR = os.getcwd()
print(CURRENT_DIR)

import warnings
warnings.filterwarnings("ignore")



log_directory = '/Users/vahidatabaigi/SIP/ESP_HardSplits/data/Reports/hyperOp_report'


color_map = {'ESPC1f': 'red', 'C1e': 'blue', 'C1f': 'green', 'I1e': 'black', 'I1f': 'magenta'}



plotting("PreGNN", experiment="1D", log_directory=log_directory, color_map=color_map, split_number=3)



