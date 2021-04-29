from Hate import *
from Sentiment import *
from Offensive import *
from preprocessing import *
import sys
import torch
from flask import Blueprint, request

classify_api = Blueprint('classify_api', __name__)

@classify_api.route('/classify/<classification_task>', methods=['POST'])
def start_classification(classification_task):
    # classification_task = request.headers.get('Task', type=str)
    ''' Classification_task is specified from the command line '''
    # classification_task = str(sys.argv[1])
    print(classification_task)
    task = None
    config = None
    freeze_layers = None

    ''' Creating task specific configurations and data. '''
    if classification_task == "sentiment":
        task = sentiment()
        freeze_layers = 1
        config = {'batch_size': 15, 'lr': 1e-5, 'epochs': 1}
    if classification_task == "hate":
        task = hate()
        freeze_layers = True
        config = {'batch_size': 24, 'lr': 6e-5, 'epochs': 3}
    if classification_task == "offensive":
        task = offensive()
        freeze_layers = 0
        config = {'batch_size': 34, 'lr': 5e-5, 'epochs': 2}

    ''' Creating object of Class preprocessing
    the preprocessing class fetches data and cleanst it.
    Next it converts the data in to data frame and returns
    '''
    data = preprocessing()
    train, val, test = data.prepare_dataset(classification_task)

    print('=========================================')
    print('CLASSIFICATION TASK: {}'.format(classification_task))
    print('=========================================')

    ''' Celaring cuda cache for better memory management during training '''
    torch.cuda.empty_cache()

    ''' Task specifing api will be called with configured parameters '''
    task.fineTune(config['batch_size'], config['lr'], config['epochs'], data, freeze_layers)
