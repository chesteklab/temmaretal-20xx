import pdb

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import pickle
import seaborn as sns
import scipy.stats as stats

from utils.ztools import ZStructTranslator, getZFeats, sliceMiddleTrials
from utils import offline_metrics, offline_training
from utils import nn_decoders
from utils.offline_data import data_cleanup, truncate_to_shortest, split_context_data
import config


def context_offline(mk_name, date, runs, labels, preprocess=True, train_rr=True, train_tcn=True, train_rnn=True):
    numFolds = 5
    if preprocess:
        trainData = {}
        testData = {}
        data_lengths = []
        for context, runi in zip(labels, runs):
            runi = f'Run-{str(runi).zfill(3)}'
            fpath = os.path.join(config.raw_data_dir, mk_name, date, runi)
            z = ZStructTranslator(fpath, os.path.join(config.data_dir, 'context_offline'), numChans=config.numChans)
            z = z.asdataframe()
            z = z[z['TrialSuccess'] != 0] # remove unsuccessful trials
            z = sliceMiddleTrials(z, 600) #get 600 trials from each run
            trainDD = getZFeats(z.iloc[0:500,:], config.binsize, featList=['FingerAnglesTIMRL','NeuralFeature'])
            testDD = getZFeats(z.iloc[500:,:], config.binsize, featList=['FingerAnglesTIMRL','NeuralFeature'])
            
            trainData[context] = data_cleanup(trainDD)
            testData[context] = data_cleanup(testDD)
            data_lengths.append([len(trainData[context]["vel"]),
                                 len(testData[context]["vel"])])
        #truncate the data to the length of the shortest dataset (I don't like how I'm doing this but it works)
        data_lengths = np.asarray(data_lengths)
        trainData = truncate_to_shortest(trainData,np.min(data_lengths[:,0]))
        testData = truncate_to_shortest(testData, np.min(data_lengths[:,1])) #maybe don't need to do this

        # now, we'll get all the indices we'll need for splitting up the data first into 5 folds, and then
        # within set of 4 folds (leave one out) the indices for the mixed context set.
        trainData, inIDXList, outIDXList, mixIDXList = split_context_data(trainData, numFolds)

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
        with open(os.path.join(config.model_dir, 'context_offline', f'rr_models_{date}.pkl'), 'wb') as f:
            pickle.dump(rr_models, f)
        print('RR Decoders Saved.')
    else:
        with open(os.path.join(config.model_dir, 'context_offline', f'rr_Models_{date}.pkl'), 'rb') as f:
            rr_models = pickle.load(f)
        print('RR Decoders Loaded.')

    #TCN Training
    if train_tcn:
        epochs = 10
        tcn_models = {}
        scalers = {}

        for i, context in enumerate(trainData.keys()):
            tcn_models[context] = []
            scalers[context] = []

            for k in np.arange(numFolds):
                model, scaler, _, _ = offline_training.train_nn(trainData[context]['neu3D'][k],
                                                                trainData[context]['vel'][k],
                                                                nn_decoders.TCN,
                                                                'TCN',
                                                                normalize=False)
                
                tcn_models[context].append(model)
                scalers[context].append(scaler)

            print(f'tcn models for {context} trained.')
        with open(os.path.join(config.model_dir, 'context_offline', f'tcn_models_{date}.pkl'), 'wb') as f:
            pickle.dump((tcn_models, scalers), f)
        print('tcn Decoders Saved.')
    else:
        with open(os.path.join(config.model_dir, 'context_offline', f'tcn_models_{date}.pkl'), 'rb') as f:
            tcn_models, scalers = pickle.load(f)
        print('tcn Decoders Loaded.')

    #LSTM Training
    if train_rnn:
        epochs = 10
        rnn_models = {}
        rnn_scalers = {}
        norm_params = {}
        for i, context in enumerate(trainData.keys()):
            rnn_models[context] = []
            rnn_scalers[context] = []
            norm_params[context] = []

            for k in np.arange(numFolds):
                model, scaler, _, norms = offline_training.train_nn(trainData[context]['neu3D'][k],
                                                                    trainData[context]['vel'][k],
                                                                    nn_decoders.RecurrentModel,
                                                                    'LSTM_xnorm_ynorm',
                                                                    normalize=True)
                
                rnn_models[context].append(model)
                rnn_scalers[context].append(scaler)
                norm_params[context].append(norms)

            print(f'RNN models for {context} trained.')
        with open(os.path.join(config.model_dir, 'context_offline', f'rnn_models_{date}.pkl'), 'wb') as f:
            pickle.dump((rnn_models, rnn_scalers, norm_params), f)
        print('RNN Decoders Saved.')
    else:
        with open(os.path.join(config.model_dir, 'context_offline', f'rnn_models_{date}.pkl'), 'rb') as f:
            rnn_models, rnn_scalers, norm_params = pickle.load(f)
        print('RNN Decoders Loaded.')
    
    # get predictions
    rr_predictions = {}
    tcn_predictions = {}
    rnn_predictions = {}

    # iterate through the folds (we'll operate within it)
    for k in np.arange(numFolds):
        # go through each test set
        for i, testcontext in enumerate(config.context_order[0:-2]): #don't include mixed
            neu_test = testData[testcontext]['neu2D']
            neu3_test = torch.from_numpy(testData[testcontext]['neu3D']).to(config.device, config.dtype)
            
            if k == 0:
                #we'll have a dict of dicts, organized as:
                # rr_predictions[testcontext][traincontext]
                rr_predictions[testcontext] = {}
                tcn_predictions[testcontext] = {}
                rnn_predictions[testcontext] = {}

            #we'll iterate through the models in this fold and predict
            for j, modelcontext in enumerate(tcn_models.keys()):
                if k == 0:
                    # if we're on the first fold, need to make a new list
                    rr_predictions[testcontext][modelcontext] = []
                    tcn_predictions[testcontext][modelcontext] = []
                    rnn_predictions[testcontext][modelcontext] = []

                #get the rr prediction
                rr_predictions[testcontext][modelcontext].append(
                    offline_training.rrPredict(neu_test, rr_models[modelcontext][k]))

                #get the nn prediction
                tcn = tcn_models[modelcontext][k].to(config.device)
                tcn.eval()
                
                tcn_yh = tcn(neu3_test)
                tcn_predictions[testcontext][modelcontext].append(scalers[modelcontext][k].scale(tcn_yh).detach().numpy())
                
                rnn = rnn_models[modelcontext][k].to(config.device)
                rnn.eval()
                rnn_scaler = rnn_scalers[modelcontext][k]
                norms = norm_params[modelcontext][k]

                neu3_test_norm = (neu3_test - norms[0]) / (norms[1] + 1e-6)
                rnn_yh = rnn(neu3_test_norm)
                rnn_yh = rnn_scaler.scale(rnn_yh[0].cpu().detach().numpy())
                rnn_yh = rnn_yh * norms[3].cpu().detach().numpy() + norms[2].cpu().detach().numpy()

                rnn_predictions[testcontext][modelcontext].append(rnn_yh)
            
    # scale velocities so they are in AU/sec not AU/bin
    sec = 1000
    testcontexts = list(tcn_predictions.keys())
    modelcontexts = list(tcn_predictions[testcontexts[0]].keys())
    test_velocities = {} #scale the test data as well
    for k in np.arange(numFolds):
        for testc in testcontexts:
            for modelc in modelcontexts:
                tcn_predictions[testc][modelc][k] = tcn_predictions[testc][modelc][k]/config.binsize * sec
                rnn_predictions[testc][modelc][k] = rnn_predictions[testc][modelc][k]/config.binsize * sec
                rr_predictions[testc][modelc][k] = rr_predictions[testc][modelc][k]/config.binsize * sec
            test_velocities[testc] = testData[testc]['vel']/config.binsize * sec

    with open(os.path.join(config.results_dir, 'context_offline', f'predictions_{date}.pkl'), 'wb') as f:
        pickle.dump((tcn_predictions, rnn_predictions, rr_predictions, test_velocities), f)
    
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

                # tcn MSE
                mse_calc('tcn', tcn_predictions)

                # rr MSE
                mse_calc('rr', rr_predictions)

                # rnn mse
                mse_calc('rnn', rnn_predictions)

    metrics = pd.DataFrame(metrics)
    return metrics

def context_offline_partII(results, figdate): #metrics on all days

    # create the figure
    make_context_figure(results, figdate)
    #do stats
    do_context_stats(results)

def do_context_stats(metrics):
    # Difference in means across days
    comparisons = {'comparison':[], 'diff':[], 'diff_pct':[], 'pct_relative_to':[], 'p_value':[]}
    # per day means and stats

    droplist = ['mse', 'decoder', 'index', 'level_0']
    tcn_mses = metrics.loc[metrics['decoder'] == 'tcn', :].reset_index()
    rnn_mses = metrics.loc[metrics['decoder'] == 'rnn', :].reset_index()
    rr_mses = metrics.loc[metrics['decoder'] == 'rr', :].reset_index()

    # pairwise comparisons between decoders
    combos = [['rr','tcn'],
              ['rr','rnn'],
              ['tcn','rnn']]
    for comb in combos:
        mse1 = metrics.loc[metrics['decoder'] == comb[0], :].reset_index()
        mse2 = metrics.loc[metrics['decoder'] == comb[1], :].reset_index()

        comparisons['comparison'].append(f'{comb[0]} > {comb[1]}')
        comparisons['diff'].append((mse1['mse'] - mse2['mse']).mean())
        comparisons['diff_pct'].append((mse1['mse'] - mse2['mse']).mean()/mse1['mse'].mean())
        comparisons['pct_relative_to'].append(comb[0])
        comparisons['p_value'].append(stats.ttest_rel(mse1['mse'], mse2['mse'], alternative='greater').pvalue)
    
    # get  MSEs for on, off, mixed
    grouped_mses = metrics.groupby(['decoder','on_off'])
    grouped_mses_means = grouped_mses['mse'].agg(['mean','std'])
    # get differences between each group (separated by decoder)
    combos = [['off', 'on'],
              ['off', 'mix'],
              ['mix', 'on']]

    for comb in combos:
        for decoder in ('rr','tcn','rnn'):
            df1 = grouped_mses.get_group((decoder,comb[0]))
            df2 = grouped_mses.get_group((decoder,comb[1]))

            m1 = grouped_mses_means.loc[(decoder, comb[0]),'mean']
            m2 = grouped_mses_means.loc[(decoder, comb[1]),'mean']

            diff = m1 - m2
            if comb[1] == 'mix':
                diffpct = (m1 - m2)/m1
                comparisons['pct_relative_to'].append(comb[0])
            else:
                diffpct = (m1 - m2)/m2
                comparisons['pct_relative_to'].append(comb[1])
            
            comparisons['comparison'].append(f'{decoder}: {comb[0]} > {comb[1]}')
            comparisons['diff'].append(diff)
            comparisons['diff_pct'].append(diffpct)
            comparisons['p_value'].append(stats.ttest_ind(df1['mse'], df2['mse'], alternative='greater').pvalue)


    df1 = grouped_mses.get_group(('rr','on'))
    df2 = grouped_mses.get_group(('tcn','off'))
    m1 = grouped_mses_means.loc[('rr','on'),'mean']
    m2 = grouped_mses_means.loc[('tcn','off'),'mean']

    diff = m1 - m2
    diffpct = (m1-m2)/m1
    comparisons['comparison'].append(f'rr on > tcn off')
    comparisons['diff'].append(diff)
    comparisons['diff_pct'].append(diffpct)
    comparisons['pct_relative_to'].append('rr on')
    comparisons['p_value'].append(stats.ttest_ind(df1['mse'], df2['mse'], alternative='greater').pvalue)

    # compare short and full
    for decoder in ('rr', 'tcn', 'rnn'):
        mses = metrics.loc[metrics['decoder'] == decoder,:].reset_index()
        short = mses.loc[mses['train_context'] == 'Mixed',:].drop('level_0', axis=1).reset_index()
        full = mses.loc[mses['train_context'] == 'Mixed_Full',:].drop('level_0', axis=1).reset_index()

        comparisons['comparison'].append(f'{decoder}: short v full')

        comparisons['diff'].append((short['mse'].mean() - full['mse']).mean())
        comparisons['diff_pct'].append((short['mse'].mean() - full['mse']).mean()/full['mse'].mean())
        comparisons['pct_relative_to'].append('full')
        comparisons['p_value'].append(stats.ttest_rel(short['mse'], full['mse'], alternative='greater').pvalue)
    
    # save results
    grouped_mses_means.to_csv(os.path.join(config.results_dir,'context_offline','groupmeans.csv'))
    pd.DataFrame(comparisons).to_csv(os.path.join(config.results_dir, 'context_offline', 'comparisons.csv'))
    

def make_context_figure(results, date):
    with open(os.path.join(config.results_dir, 'context_offline', f'predictions_{date}.pkl'), 'rb') as f:
        tcn_predictions, rnn_predictions, rr_predictions, test_velocities = pickle.load(f)
    context_fig = plt.figure(figsize=(10,6), layout='constrained')
    sfs = context_fig.subfigures(2,1)

    bar_axs = sfs[1].subplots(1,3, sharey=True)

    top_row = sfs[0].subfigures(1,2)
    examples = top_row[0].subplots(3,1, sharex=True)
    group_mses = top_row[1].add_subplot()

    ## example traces of an on and off context prediction maybe?
    predictions = (tcn_predictions, rnn_predictions, rr_predictions)
    decoders = ('tcn', 'rnn', 'rr')
    timeslice = slice(425,550)
    truth = test_velocities['Normal'][timeslice,1]
    for i in range(3):
        on_example = predictions[i]['Normal']['Normal'][0][timeslice,1]
        off_example = predictions[i]['Normal']['SprWrst'][0][timeslice,1]

        examples[i].plot(truth, 'k')
        examples[i].plot(on_example, color=config.context_palette[0,:])
        examples[i].plot(off_example, color=config.context_palette[3,:])
        examples[i].set(yticks=[-2,0,2])

    examples[0].set(title='on vs. off context predictions', ylabel='vel')

    results_date = results[results['date'] == date]
    for i, (decoder, res) in enumerate(results_date.groupby('decoder')):
        # barplot
        width = .8
        sns.barplot(res, x='test_context', y='mse', hue='train_context', 
                    hue_order=config.context_order, palette=config.context_palette,
                    width=width, ax=bar_axs[i], alpha=0.6)
            
        sns.stripplot(res, x='test_context', y='mse', hue='train_context', 
                    hue_order=config.context_order, palette=config.context_palette,
                    dodge=True, ax=bar_axs[i], alpha=0.7, zorder=1)
            
        # add arrows above on context
        xtick_loc = {v.get_text(): v.get_position()[0] for v in bar_axs[i].get_xticklabels()}
        onx = np.zeros((len(xtick_loc.keys())))
        ony = np.zeros_like(onx)
        for j, tickkey in enumerate(xtick_loc.keys()):
            context = tickkey
            mask = (res['train_context'] == context) & (res['test_context'] == context)
            barwidth = width/len(config.context_order)
            onx[j] = xtick_loc[tickkey] - (width/2) + barwidth/2 + barwidth * j
            ony[j] = res.loc[mask, 'mse'].mean() + 0.03
        bar_axs[i].scatter(onx, ony, marker='v', c='k', s=20, alpha=0.5)
        bar_axs[i].set_title(decoder)
        
        if i != 0:
            bar_axs[i].set(ylabel=None)
        bar_axs[i].get_legend().remove()
        bar_axs[i].set(title=decoder, xlabel='Test Context', ylabel='MSE')


    sns.barplot(results_date, x='decoder', y='mse', hue='on_off', hue_order=['on','off','mix'],
                ax=group_mses, palette=config.context_group_palette, alpha=.6, errorbar='se')
    sns.stripplot(results_date, x='decoder', y='mse', hue='on_off', hue_order=['on','off','mix'],
                ax=group_mses, palette=config.context_group_palette, dodge=True, alpha=0.7, zorder=1)

    group_mses.set(title='grouped', xlabel=None, ylabel='MSE')

    context_fig.savefig(os.path.join(config.results_dir, 'context_offline', 'context_offlineFigure.pdf'))
    