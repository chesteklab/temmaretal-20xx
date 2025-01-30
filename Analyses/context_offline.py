import pdb

import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import pickle
import seaborn as sns
import scipy.stats as stats

from utils.ztools import ZStructTranslator, getZFeats, sliceMiddleTrials
from utils import offline_metrics, offline_training
from utils import nn_decoders
from utils.offline_data import datacleanup, truncateToShortest, splitContextData
import config


def context_offline(serverpath, mk_name, date, runs, labels, preprocess=True, train_rr=True, train_nn=True):
    numFolds = 5
    if preprocess:
        trainData = {}
        testData = {}
        data_lengths = []
        for context, runi in zip(labels, runs):
            runi = f'Run-{str(runi).zfill(3)}'
            fpath = os.path.join(serverpath, mk_name, date, runi)
            z = ZStructTranslator(fpath, numChans=config.numChans)
            z = z.asdataframe()
            z = z[z['TrialSuccess'] != 0] # remove unsuccessful trials
            z = sliceMiddleTrials(z, 600) #get 600 trials from each run
            trainDD = getZFeats(z.iloc[0:500,:], config.binsize, featList=['FingerAnglesTIMRL','NeuralFeature'])
            testDD = getZFeats(z.iloc[500:,:], config.binsize, featList=['FingerAnglesTIMRL','NeuralFeature'])
            print(len(z.iloc[500:]))
            trainData[context] = datacleanup(trainDD)
            testData[context] = datacleanup(testDD)
            data_lengths.append([len(trainData[context]["vel"]),
                                 len(testData[context]["vel"])])
        #truncate the data to the length of the shortest dataset (I don't like how I'm doing this but it works)
        data_lengths = np.asarray(data_lengths)
        trainData = truncateToShortest(trainData,np.min(data_lengths[:,0]))
        testData = truncateToShortest(testData, np.min(data_lengths[:,1])) #maybe don't need to do this

        # now, we'll get all the indices we'll need for splitting up the data first into 5 folds, and then
        # within set of 4 folds (leave one out) the indices for the mixed context set.
        trainData, inIDXList, outIDXList, mixIDXList = splitContextData(trainData, numFolds)

        # we should now be ready for training decoders, predicting, etc.
        with open(os.path.join(config.data_dir, 'context_offline', f'data_{date}.pkl'),'wb') as f:
            pickle.dump((trainData, testData, inIDXList, outIDXList, mixIDXList), f)
        print('Data Pre-Processed and Saved')

        #if we re-preprocessed data, we should retrain the decoders.
        print('Overriding train_rr and train_nn')
        train_nn = True
        train_rr = True
    else:
        with open(os.path.join(config.data_dir, 'context_offline', f'data_{date}.pkl'),'rb') as f:
            trainData, testData, inIDXList, outIDXList, mixIDXList = pickle.load(f)
        print("Data Loaded")

    #starting with ridge regression
    if train_rr:
        lbda = 0.001
        rr_models = {}
        for i, context in enumerate(trainData.keys()):
            #create a list where each model trained on each fold will go.
            rr_models[context] = []
            for k in np.arange(numFolds):
                neu = trainData[context]['neu2D'][k]
                vel = trainData[context]['vel'][k]

                rr_models[context].append(offline_training.rrTrain(neu, vel, lbda=lbda))
            print(f'RR models for {context} trained.')
        with open(os.path.join(config.model_dir, 'context_offline', f'RRModels_{date}.pkl'), 'wb') as f:
            pickle.dump(rr_models, f)
        print('RR Decoders Saved.')
    else:
        with open(os.path.join(config.model_dir, 'context_offline', f'RRModels_{date}.pkl'), 'rb') as f:
            rr_models = pickle.load(f)
        print('RR Decoders Loaded.')

    #Now we train the tcFNN
    device = torch.device('cuda:0')
    dtype = torch.float
    if train_nn:
        epochs = 10
        nn_models = {}
        scalers = {}

        for i, context in enumerate(trainData.keys()):
            nn_models[context] = []
            scalers[context] = []

            for k in np.arange(numFolds):
                #need to make these tensors
                neu = torch.from_numpy(trainData[context]['neu3D'][k]).to(device, dtype)
                vel = torch.from_numpy(trainData[context]['vel'][k]).to(device, dtype)

                ds = TensorDataset(neu, vel)

                #since we know how long we're training, val dataset can just be the same
                # as training
                dl = DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)
                dl2 = DataLoader(ds, batch_size=len(ds),drop_last=True)

                #Model instance and architecture
                in_size = neu.shape[1]
                num_states = 2
                # initialize the decoder and optimizer
                model = offline_training.init_model(nn_decoders.tcFNN, 'TCN', in_size, num_states)
                
                # TODO - update the configs for the different decoder types
                opt, scheduler = offline_training.init_opt(model, 'TCN')

                # train network
                loss_h, vloss_h = offline_training.fit(epochs, nn_model, opt, dl, dl2, print_every=15,
                                                      print_results=False)
                
                nn_model = nn_model.cpu()
                nn_models[context].append(nn_model)

                #generate the scaler
                dl_scale = DataLoader(ds, batch_size=len(ds),shuffle=True) #shuffle doesn't matter

                scalers[context].append(offline_training.generate_output_scaler(nn_model.to(device), dl_scale,
                                                                                num_outputs=num_states))
            print(f'tcFNN models for {context} trained.')
        with open(os.path.join(config.model_dir, 'context_offline', f'NNModels_{date}.pkl'), 'wb') as f:
            pickle.dump((nn_models, scalers), f)
        print('tcFNN Decoders Saved.')
    else:
        with open(os.path.join(config.model_dir, 'context_offline', f'NNModels_{date}.pkl'), 'rb') as f:
            nn_models, scalers = pickle.load(f)
        print('tcFNN Decoders Loaded.')

    # get predictions
    rr_predictions = {}
    nn_predictions = {}

    # iterate through the folds (we'll operate within it)
    for k in np.arange(numFolds):
        # go through each test set
        for i, testcontext in enumerate(config.contextOrder[0:-2]): #don't include mixed
            neu_test = testData[testcontext]['neu2D']
            neu2_test = testData[testcontext]['neu3D']
            neu2_test = torch.from_numpy(neu2_test).to(device, dtype)

            if k == 0:
                #we'll have a dict of dicts, organized as:
                # rr_predictions[testcontext][traincontext]
                rr_predictions[testcontext] = {}
                nn_predictions[testcontext] = {}

            #we'll iterate through the models in this fold and predict
            for j, modelcontext in enumerate(nn_models.keys()):
                if k == 0:
                    # if we're on the first fold, need to make a new list
                    rr_predictions[testcontext][modelcontext] = []
                    nn_predictions[testcontext][modelcontext] = []

                #get the rr prediction
                rr_predictions[testcontext][modelcontext].append(
                    offline_training.rrPredict(neu_test, rr_models[modelcontext][k]))

                #get the nn prediction
                model = nn_models[modelcontext][k]
                model.to(device=device)
                model.eval()

                nn_predictions[testcontext][modelcontext].append(
                    scalers[modelcontext][k].scale(model(neu2_test)).detach().numpy()
                )
    # scale velocities so they are in AU/sec not AU/bin
    sec = 1000
    testcontexts = list(nn_predictions.keys())
    modelcontexts = list(nn_predictions[testcontexts[0]].keys())
    test_velocities = {} #scale the test data as well
    for k in np.arange(numFolds):
        for testc in testcontexts:
            for modelc in modelcontexts:
                nn_predictions[testc][modelc][k] = nn_predictions[testc][modelc][k]/config.binsize * sec
                rr_predictions[testc][modelc][k] = rr_predictions[testc][modelc][k]/config.binsize * sec
            test_velocities[testc] = testData[testc]['vel']/config.binsize * sec

    # Calculate MSE for each decoder prediction (iter through folds, test contexts, and model contexts)
    metrics = {'decoder':[],'test_context':[],'train_context':[], 'on_off':[],
               'fold':[],'mse':[], 'date':[]}
    for k in np.arange(numFolds):
        for testc in testcontexts:
            for modelc in modelcontexts:
                # calculate MSE on a prediction, add to metrics dict (will become a pd dataframe)
                def mse_calc(decoder, pred):
                    # params
                    metrics['decoder'].append(decoder)
                    metrics['test_context'].append(testc)
                    metrics['train_context'].append(modelc)
                    if modelc == testc:
                        metrics['on_off'].append('on')
                    elif modelc == 'Mixed':
                        metrics['on_off'].append('mix')
                    elif modelc == 'Mixed_Full':
                        metrics['on_off'].append('mix')
                    else:
                        metrics['on_off'].append('off')
                    metrics['fold'].append(k)
                    metrics['date'].append(date)

                    #calculate mse
                    mse = offline_metrics.mse(pred[testc][modelc][k].flatten(), test_velocities[testc].flatten())
                    metrics['mse'].append(mse)

                # tcfNN MSE
                mse_calc('tcfnn', nn_predictions)

                # rr MSE
                mse_calc('rr', rr_predictions)

    metrics = pd.DataFrame(metrics)
    return metrics

def context_offline_partII(metrics, figdate): #metrics on all days
    # create columns in the bottom row of the figure for the amount of days

    #Create figure, add example traces
    cont_fig = plt.figure(figsize=(16,6))
    subfigs = cont_fig.subfigures(1,2, width_ratios=[6, 2])
    ax = subfigs[0].subplots(1,2, sharey=True)
    groupax = subfigs[1].subplots(1)
    subfigs[0].suptitle('A. Decoder accuracy across contexts (on 1 day)')
    subfigs[1].suptitle('B. Grouped in-day accuracy')
    # first two, MSE breakdowns for RR and NN

    for i, (decoder, met) in enumerate(metrics.groupby('decoder')):
        # get the initial barplot
        width = 0.8
        sns.barplot(met.loc[met['date'] == figdate,:], x='test_context', y='mse', hue='train_context',
                    hue_order=config.contextOrder, palette=config.contextPalette, errorbar = 'se',
                    width = width, ax=ax[i])

        # add arrows above the on context
        xtick_loc = {v.get_text(): v.get_position()[0] for v in ax[i].get_xticklabels()}
        onx = np.zeros((len(xtick_loc.keys()),1))
        ony = np.zeros_like(onx)
        for j, tickkey in enumerate(xtick_loc.keys()):
            context = tickkey
            mask = (met['date'] == figdate) & (met['train_context'] == context) & (met['test_context'] == context)
            barwidth = width/len(config.contextOrder)
            onx[j] = xtick_loc[tickkey] - (width/2) + barwidth/2 + barwidth * j
            ony[j] = met.loc[mask, 'mse'].mean() + 0.03

        ax[i].scatter(onx, ony, marker='v', c='k', s=30)
        ax[i].set(title='Ridge Regression' if decoder == 'rr' else 'tcFNN', xlabel='Test Context',
                  ylabel='Mean-Squared Error')
        ax[i].legend(title='Training Context', ncol=3)

    # third plot - MSE grouped across days
    sns.barplot(metrics, x='decoder', y='mse', hue='on_off', ax=groupax, hue_order=['on', 'off', 'mix'],
                errorbar='se', palette=config.contextGroupPalette)

    ax[1].get_legend().remove()
    ax[1].set_ylabel(None)
    groupax.legend(title='Prediction Type')
    groupax.set(ylabel='Mean-Squared Error', title='spacing', xlabel='Decoder')

    # Difference in means across days
    comparisons = {'comparison':[], 'diff':[], 'diffpct':[], 'pctrelative':[], 'pvalue':[]}
    # per day means and stats

    nn_mses = metrics.loc[metrics['decoder'] == 'tcfnn',:].reset_index()
    rr_mses = metrics.loc[metrics['decoder'] == 'rr', :].reset_index()

    # check to make sure the folds, train context, and test context match up
    droplist = ['mse', 'decoder', 'index', 'level_0']
    if not nn_mses.drop(droplist, axis=1).equals(rr_mses.drop(droplist, axis=1)):
        Exception('uh oh')

    # pairwise comparison between respective nn and rr predictions
    comparisons['comparison'].append('rr > nn')
    comparisons['diff'].append((rr_mses['mse'] - nn_mses['mse']).mean())
    comparisons['diffpct'].append((rr_mses['mse'] - nn_mses['mse']).mean()/rr_mses['mse'].mean())
    comparisons['pctrelative'].append('rr')
    comparisons['pvalue'].append(stats.ttest_rel(rr_mses['mse'], nn_mses['mse'], alternative='greater').pvalue)

    # get  MSEs for on, off, mixed
    grouped_mses = metrics.groupby(['decoder','on_off'])
    grouped_mses_means = grouped_mses['mse'].agg(['mean','std'])

    # get differences between each group (separated by decoder)
    combos = [['off', 'on'],
              ['off', 'mix'],
              ['mix', 'on']]

    for comb in combos:
        for decoder in ('rr','tcfnn'):
            df1 = grouped_mses.get_group((decoder,comb[0]))
            df2 = grouped_mses.get_group((decoder,comb[1]))

            m1 = grouped_mses_means.loc[(decoder, comb[0]),'mean']
            m2 = grouped_mses_means.loc[(decoder, comb[1]),'mean']

            diff = m1 - m2
            if comb[1] == 'mix':
                diffpct = (m1 - m2)/m1
                comparisons['pctrelative'].append(comb[0])
            else:
                diffpct = (m1 - m2)/m2
                comparisons['pctrelative'].append(comb[1])
            
            comparisons['comparison'].append(f'{decoder}: {comb[0]} > {comb[1]}')
            comparisons['diff'].append(diff)
            comparisons['diffpct'].append(diffpct)
            comparisons['pvalue'].append(stats.ttest_ind(df1['mse'], df2['mse'], alternative='greater').pvalue)


    df1 = grouped_mses.get_group(('rr','on'))
    df2 = grouped_mses.get_group(('tcfnn','off'))
    m1 = grouped_mses_means.loc[('rr','on'),'mean']
    m2 = grouped_mses_means.loc[('tcfnn','off'),'mean']

    diff = m1 - m2
    diffpct = (m1-m2)/m1
    comparisons['comparison'].append(f'rr on > tcfnn off')
    comparisons['diff'].append(diff)
    comparisons['diffpct'].append(diffpct)
    comparisons['pctrelative'].append('rr on')
    comparisons['pvalue'].append(stats.ttest_ind(df1['mse'], df2['mse'], alternative='greater').pvalue)

    # compare short and full
    nn_mses = metrics.loc[metrics['decoder'] == 'tcfnn',:].reset_index()
    rr_mses = metrics.loc[metrics['decoder'] == 'rr', :].reset_index()

    short_nn = nn_mses.loc[metrics['train_context'] == 'Mixed',:].drop('level_0',axis=1).reset_index()
    short_rr = rr_mses.loc[metrics['train_context'] == 'Mixed',:].drop('level_0',axis=1).reset_index()

    full_nn = nn_mses.loc[metrics['train_context'] == 'Mixed_Full',:].drop('level_0',axis=1).reset_index()
    full_rr = rr_mses.loc[metrics['train_context'] == 'Mixed_Full',:].drop('level_0',axis=1).reset_index()

    comparisons['comparison'].append('nn: short v full')
    comparisons['diff'].append((short_nn['mse'] - full_nn['mse']).mean())
    comparisons['diffpct'].append((short_nn['mse'] - full_nn['mse']).mean()/full_nn['mse'].mean())
    comparisons['pctrelative'].append('full')
    comparisons['pvalue'].append(stats.ttest_rel(short_nn['mse'], full_nn['mse'], alternative='two-sided').pvalue)

    comparisons['comparison'].append('rr: short v full')
    comparisons['diff'].append((short_rr['mse'] - full_rr['mse']).mean())
    comparisons['diffpct'].append((short_rr['mse'] - full_rr['mse']).mean()/full_rr['mse'].mean())
    comparisons['pctrelative'].append('full')
    comparisons['pvalue'].append(stats.ttest_rel(short_rr['mse'], full_rr['mse'], alternative='two-sided').pvalue)

    # save results
    grouped_mses_means.to_csv(os.path.join(config.results_dir,'context_offline','groupmeans.csv'))
    pd.DataFrame(comparisons).to_csv(os.path.join(config.results_dir, 'context_offline', 'comparisons.csv'))
    cont_fig.savefig(os.path.join(config.results_dir, 'context_offline', 'context_offlineFigure.pdf'))