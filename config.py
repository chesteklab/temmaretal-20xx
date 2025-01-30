import pdb

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import torch
from matplotlib import font_manager
import seaborn as sns
import os
#some basic text parameters for figures
mpl.rcParams['font.family'] = "Atkinson Hyperlegible" # if installed but not showing up, rebuild mpl cache
mpl.rcParams['font.size'] = 12
mpl.rcParams['savefig.format'] = 'pdf'
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.titlesize'] = 18
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlelocation'] = 'left'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['figure.constrained_layout.use'] = True
mpl.rcParams['figure.titlesize'] = 18
mpl.rcParams['figure.titleweight'] = 'bold'
mpl.rcParams['pdf.fonttype'] = 42

# Sets up some parameters used throughout the project

# set up standard plot colors
hcColor = np.asarray([0.5, 0.5, 0.5])
kfColor = np.asarray([64, 64, 192]) / 255
dsColor = np.asarray([64, 192, 64]) / 255
nnColor = np.asarray([192, 64, 64]) / 255

noregColor = np.asarray([64, 192, 192]) / 255
nobnColor = np.asarray([192, 64, 192]) / 255
nodpColor = np.asarray([128, 128, 224]) / 255


offlinePalette = np.stack((kfColor, dsColor, nnColor))
onlinePalette = np.stack((hcColor, kfColor, nnColor))
offlineVariancePalette = {'hc':hcColor, 'tcfnn':nnColor, 'nodp':nodpColor,
                          'nobn':nobnColor, 'noreg':noregColor}

variance_models = ('TCN', 'TCN_nobn', 'TCN_nodp', 'TCN_noreg')
varianceLabels = ('tcfnn', 'dropout only (noBN)', 'batchnorm only (noDP)', 'noReg')
varianceTicks = ('tcFNN', 'noBN', 'noDP', 'noReg')

contextPalette = np.asarray([[255, 200, 50],
                             [255, 155, 40],
                             [255, 100, 45],
                             [225, 25, 30],
                             [60, 200, 255],
                             [30, 170, 225]])/255

contextGroupPalette = np.asarray([[250, 190, 0],
                                   [155, 20, 115],
                                   [60, 200, 255]])/255

contextOrder = ['Normal', 'Spring', 'Wrist','SprWrst', 'Mixed','Mixed_Full']
onlineTracesPalette = contextGroupPalette[0:2,:]

dsmap = mpl.colors.LinearSegmentedColormap.from_list('dsmap',[[0,0,0],dsColor])
# standard binsize (when not already used)
binsize = 50
numChans = 96
batch_size = 64

# get cwd for making the folders - so that all the data should generate/get saved into the right places
raw_data_dir = 'Z:\Data\Monkeys'
output_dir = 'Z:\\Student Folders\\Hisham_Temmar\\tcFNNPaper\\Revision'
model_dir = os.path.join(output_dir,'Models')
results_dir = os.path.join(output_dir,'Results')
data_dir = os.path.join(output_dir,'Data')
architectures_path = os.path.join(os.getcwd(), 'network_architectures.yaml')
training_params_path = os.path.join(os.getcwd(), 'training_params.yaml')


device = torch.device('cuda:0')
dtype = torch.float