import numpy as np
import os
import pdb
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import pickle
import pandas as pd
import seaborn as sns
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib as mpl

import config
import utils.online_metrics
from utils.ztools import ZStructTranslator, getZFeats, sliceMiddleTrials
from utils import offline_training, nn_decoders, offline_metrics
from utils.offline_data import datacleanup, splitOfflineData

'''
Addressing the first block of the paper: Do NNs fit better than other methods?
- Offline Figure showing fit of predicted velocities of various approaches to hand control
'''
def fits_offline(mk_name, date, runs, preprocess=True, train_rr=True, train_ds=True,
                         train_nn=True, genfig=True):
    #setup pytorch stuff
    device = torch.device('cuda:0')
    dtype = torch.float

    numFolds = 5

    # load and preprocess data if needed
    if preprocess:
        print(type(runs))
        for i in range(len(runs)):
            run = 'Run-{}'.format(str(runs[i]).zfill(3))
            fpath = os.path.join(config.serverpath, mk_name, date, run)
            zadd = ZStructTranslator(fpath, numChans=config.numChans)
            # remove unsuccessful trials
            zadd = zadd.asdataframe()
            zadd = zadd[zadd['TrialSuccess'] != 0]
            if i == 0:
                z = zadd
            else:
                z = pd.concat([z,zadd])

        #take middle 1000 trials and get z feats
        if mk_name == 'Batman':
            zsliced = sliceMiddleTrials(z, 400)
            trainDD = getZFeats(zsliced[0:300], config.binsize, featList=['FingerAnglesTIMRL', 'NeuralFeature'])
            testDD = getZFeats(zsliced[300:], config.binsize, featList=['FingerAnglesTIMRL', 'NeuralFeature','TrialNumber'])
        else:
            zsliced = sliceMiddleTrials(z, 600)
            trainDD = getZFeats(zsliced[0:500], config.binsize, featList=['FingerAnglesTIMRL', 'NeuralFeature'])
            testDD = getZFeats(zsliced[500:], config.binsize, featList=['FingerAnglesTIMRL', 'NeuralFeature','TrialNumber'])

        # separate feats, add time history, add a column of ones for RR, and reshape data for NN.
        pretrainData = datacleanup(trainDD)
        testData = datacleanup(testDD)

        #split the training data into folds
        trainData, inIDXList, outIDXList = splitOfflineData(pretrainData,numFolds)

        # get trialnumber for testDD
        trial_num = testDD['TrialNumber'][3:,0].astype(int)
        with open(os.path.join(config.datadir,'fits_offline',f'data_{date}_{mk_name}.pkl'), 'wb') as f:
            pickle.dump((trainData, testData, inIDXList, outIDXList, trial_num), f)
    else:
        ## Load in saved data
        print('loading data')
        with open(os.path.join(config.datadir,'fits_offline',f'data_{date}_{mk_name}.pkl'), 'rb') as f:
            trainData, testData, inIDXList, outIDXList, trial_num = pickle.load(f)
    print('data loaded')

    ## Train RR Decoders
    if train_rr:
        lbda = 0.001 #.001 from matt's paper
        rr_models = []
        for k in np.arange(numFolds):
            neu = trainData['neu2D'][k]
            vel = trainData['vel'][k]
            rr_models.append(offline_training.rrTrain(neu, vel, lbda=lbda))
        #save model
        with open(os.path.join(config.modeldir,'fits_offline',f'RRmodel_{date}_{mk_name}.pkl'), 'wb') as f:
            pickle.dump(rr_models, f)
        print('RR Decoders Saved')
    else:
        with open(os.path.join(config.modeldir,'fits_offline',f'RRmodel_{date}_{mk_name}.pkl'), 'rb') as f:
            rr_models = pickle.load(f)
        print('RR Decoders Loaded')

    # Train Dual-State Decoders
    if train_ds:
        ds_models = []
        for k in np.arange(numFolds):
            neu = trainData['neu2D'][k]
            vel = trainData['vel'][k]

            ds_models.append(offline_training.dsTrain(neu, vel))
        with open(os.path.join(config.modeldir, 'fits_offline', f'DSmodel_{date}_{mk_name}.pkl'), 'wb') as f:
            pickle.dump(ds_models, f)
        print('DS Decoders Saved')
    else:
        with open(os.path.join(config.modeldir, 'fits_offline', f'DSmodel_{date}_{mk_name}.pkl'), 'rb') as f:
            ds_models = pickle.load(f)
        print('DS Decoders Loaded')

    ## Train tcFNN decoder
    if train_nn:
        if mk_name == 'Batman':
            epochs = 10
        else:
            epochs = 10
        nn_models = []
        scalers = []
        for k in np.arange(numFolds):
            neu = torch.from_numpy(trainData['neu3D'][k]).to(device, dtype)
            vel = torch.from_numpy(trainData['vel'][k]).to(device, dtype)

            #create pytorch datasets and then dataloaders
            ds = TensorDataset(neu, vel)

            #since we know how long we're training, val dataset can just be the same as training
            dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
            dl2 = DataLoader(ds, batch_size=len(ds), shuffle=False, drop_last=True)

            #Model Instance and architecture
            in_size = neu.shape[1]
            layer_size = 256
            ConvSize = 3
            ConvSizeOut = 16
            num_states = 2
            model = nn_decoders.tcFNN(in_size, layer_size, ConvSize, ConvSizeOut, num_states).to(device)

            #training hyperparams as determined previously (not finalized yet)
            learning_rate = 1e-4
            weight_decay = 1e-2
            opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

            #run fit function (model is trained here)

            loss_h, vloss_h = offline_training.fit(epochs, model, opt, dl, dl2, print_every=1, print_results=True)

            # generate scaler (need diff organized Dataset since scaler is online code)
            ds_scale = offline_training.BasicDataset(neu, vel)
            dl_scale = DataLoader(ds_scale, batch_size=len(ds_scale), shuffle=True)
            scalers.append(offline_training.generate_output_scaler(model, dl_scale, num_outputs=num_states))
            model = model.cpu()
            nn_models.append(model)

        print(f'tcFNN models trained.')
        # save decoders and scalers
        with open(os.path.join(config.modeldir, 'fits_offline', f'tcFNNmodels_{date}_{mk_name}.pkl'), 'wb') as f:
            pickle.dump((nn_models, scalers), f)
        print('tcFNN models Saved')
    else:
        with open(os.path.join(config.modeldir, 'fits_offline', f'tcFNNmodels_{date}_{mk_name}.pkl'), 'rb') as f:
            nn_models, scalers = pickle.load(f)
        print('NN Decoders Loaded')

    # Get predictions for each decoder
    rr_predictions = np.zeros((testData['vel'].shape[0], testData['vel'].shape[1], numFolds))
    nn_predictions = np.zeros_like(rr_predictions)
    ds_predictions = np.zeros_like(rr_predictions)
    ds_probabilities = np.zeros((testData['vel'].shape[0], numFolds))
    for i in range(numFolds):
        rr = rr_models[i]
        tcfnn = nn_models[i].to(device)
        tcfnnscaler = scalers[i]
        ds = ds_models[i]

        tcfnn.eval()

        neu3 = torch.from_numpy(testData['neu3D']).to(device, dtype)
        nn_predictions[:,:,i] = tcfnnscaler.scale(tcfnn(neu3)).cpu().detach().numpy()
        rr_predictions[:,:,i] = offline_training.rrPredict(testData['neu2D'], rr)
        ds_predictions[:,:,i], pr = offline_training.dsPredict(testData['neu2D'], ds)
        ds_probabilities[:, i:i+1] = pr

    #scale predictions for au/sec
    binsize = config.binsize
    sec = 1000

    nn_predictions = nn_predictions/binsize * sec
    rr_predictions = rr_predictions/binsize * sec
    ds_predictions = ds_predictions/binsize * sec
    vel_test = testData['vel']/binsize * sec

    # Calculate Metrics
    sortidx = np.argsort(np.abs(vel_test.flatten()))  # sort absolute values of velocities.
    hi_vel_idx = sortidx[np.floor(len(sortidx) * 9 / 10).astype(int):]  # take top 10%
    lo_vel_idx = sortidx[0:np.ceil(len(sortidx) / 10).astype(int)]  # take bottom 10%

    hi_thr= vel_test.flatten()[hi_vel_idx[0]]
    lo_thr = vel_test.flatten()[lo_vel_idx[-1]]

    binedges = np.linspace(-9, 9, 100)
    decoders = ('rr','ds','nn')
    preds = (rr_predictions, ds_predictions, nn_predictions)
    metrics = {'cc':[], 'mse':[], 'vaf':[],'mse_hi':[],'mse_lo':[],
                  'mean_hi':[],'mean_lo':[],'kl_div':[], 'decoder':[], 'fold':[]}

    # TODO: FOR NEETHAN - USE THE FILE SAVED HERE TO LOAD PREDICTIONS AND TRUE MOVEMENTS AND CALCULATE COHERENCE
    with open(os.path.join(config.resultsdir, 'fits_offline', f'offlineFitPrediction_{date}_{mk_name}.pkl'), 'wb') as f:
        pickle.dump((preds, vel_test), f)

    for k in np.arange(numFolds):
        for i,decoder in enumerate(decoders):
            prediction = preds[i][:,:,k].flatten()
            truth = vel_test.flatten()

            # get overall metrics
            # pdb.set_trace()
            metrics['mse'].append(offline_metrics.mse(truth, prediction))
            metrics['vaf'].append(offline_metrics.vaf(truth, prediction))
            metrics['cc'].append(offline_metrics.corrcoef(truth, prediction))

            pred_hi = np.abs(prediction[hi_vel_idx])
            pred_lo = np.abs(prediction[lo_vel_idx])
            pv_hi = np.abs(truth[hi_vel_idx])
            pv_lo = np.abs(truth[lo_vel_idx])
            metrics['mean_hi'].append(np.mean(pred_hi))
            metrics['mean_lo'].append(np.mean(pred_lo))
            metrics['mse_hi'].append(offline_metrics.mse(pv_hi, pred_hi))
            metrics['mse_lo'].append(offline_metrics.mse(pv_lo, pred_lo))

            #kl div will be duplicate for each trial - for now
            pv_hist, _ = np.histogram(truth, density=True, bins=binedges)
            pred_hist, _ = np.histogram(prediction, density=True, bins=binedges)

            # calculate kl divergence between hand control and decoder bins.
            # Needs to sum to 1, not integrate to 1. PMFs not PDFs
            f = pv_hist / np.sum(pv_hist)
            g = pred_hist / np.sum(pred_hist)
            metrics['kl_div'].append(offline_metrics.kldiv(f, g))
            metrics['fold'].append(k)
            metrics['decoder'].append(decoder)

    metrics = pd.DataFrame(metrics)

    fitFig = None
    mseax = None
    klax = None
    if genfig:
        # Creating the Figure:
        fitFig = plt.figure(figsize=(16,8))

        subfigs = fitFig.add_gridspec(2,2, width_ratios = [3, 1])
        tracespec = subfigs[0,0].subgridspec(1,3)
        mseax = fitFig.add_subplot(subfigs[0,1])

        distspec = subfigs[1,0].subgridspec(2,3)
        klax = fitFig.add_subplot(subfigs[1,1])
        if mk_name == 'Batman':
            plotrange = np.arange(899, 962)
        else:
            plotrange = np.arange(1499, 1562)
        times = plotrange * config.binsize / sec
        histwidth = 3

        # choose one fold's decoders to look at (as to not choose between 5 graphs a day)
        fold = 2
        traceid = 0 #finger to use in traces
        predLabels = ('Ridge Regression RR', 'Sachs et al. 2016 DS', 'Willsey et al. 2022 tcFNN')

        for i, pred in enumerate(preds):
            ax = fitFig.add_subplot(tracespec[i])
            ax.plot(times, vel_test[plotrange, traceid], color=config.hcColor, lw=histwidth)
            ax.plot(times, pred[plotrange, traceid, fold], color=config.offlinePalette[i, :], lw=histwidth)

            if i == 1:
                ax.scatter(times, pred[plotrange, traceid, fold], c=ds_probabilities[plotrange,fold],
                           cmap=config.dsmap, zorder=10, vmin=0, vmax=1)
                cb_ax = inset_axes(ax, width="40%", height = "5%", loc=2)
                plt.colorbar(mappable=mpl.cm.ScalarMappable(cmap=config.dsmap), cax=cb_ax, orientation='horizontal',
                             label='Movement Likelihood')
                for spine in cb_ax.spines.values():
                    spine.set_visible(False)
            if i == 0:
                ax.set(ylabel='Velocity (Flex/Sec)', yticks=(-1,0,1,2))
            else:
                ax.set_yticks((-1,0,1,2), labels=[])
            if mk_name == 'Batman':
                ax.set(xlabel='Time (sec)',title=predLabels[i], ylim=(-2,3),
                   xlim=(times[0],times[-1]),xticks=(times[1],times[-2]))
            else:
                ax.set(xlabel='Time (sec)',title=predLabels[i], ylim=(-1.5,2.5),
                    xlim=(times[0],times[-1]),xticks=(times[1],times[-2]))

            topax = fitFig.add_subplot(distspec[0,i])
            botax = fitFig.add_subplot(distspec[1,i])

            #plot the same data on both axes
            def histplot(ax, top=True, addlines=True):
                ax.hist(vel_test.flatten(), color=config.hcColor, density=True, bins=binedges)
                ax.hist(pred[:,:,:].flatten(), color=config.offlinePalette[i,:], density=True, histtype='step',
                        bins=binedges, linewidth=histwidth)
                lineargs = {'linestyle':'-','color':'k', 'lw':3}
                arrowargs = {'color':'k','width':0.01}
                if top and addlines:
                    ax.annotate("", xy=(lo_thr, 3), xytext=(lo_thr+1, 3),
                                arrowprops=dict(arrowstyle="-|>", lw=3))
                    ax.annotate("", xy=(0-lo_thr, 3), xytext=(0-lo_thr-1, 3),
                                arrowprops=dict(arrowstyle="-|>", lw=3))

                elif addlines:
                    ax.annotate("", xy=(hi_thr, .05), xytext=(hi_thr+1, .05),
                                arrowprops=dict(arrowstyle="<|-", lw=3))
                    ax.annotate("", xy=(0-hi_thr, .05), xytext=(0-hi_thr-1, .05),
                                arrowprops=dict(arrowstyle="<|-", lw=3))

            addlines = True if (i == 0) else False

            histplot(topax, addlines=addlines)
            histplot(botax, top=False, addlines=addlines)

            topax.set(ylim=(0.5, 3.5),xlim=(-4, 4), title='Velocity Distribution', yticks=[1, 2, 3])
            botax.set(ylim=(0,0.2),xlim=(-4,4), xlabel='Velocity (Flex/Sec)', yticks=[0,0.1,0.2])

            if i != 0:
                topax.yaxis.set_ticklabels([])
                botax.yaxis.set_ticklabels([])
            else:
                topax.set(ylabel='Estimated Density')

            utils.online_metrics.drawBrokenAxes(topax, botax, d=0.015)

    return metrics, fitFig, mseax, klax

def fits_offline_partII(mk_name, results, mseax, klax):
    # summarize results within days

    rr_summary = results.loc[results['decoder'] == 'rr', :].groupby(level='date').describe()
    ds_summary = results.loc[results['decoder'] == 'ds', :].groupby(level='date').describe()
    nn_summary = results.loc[results['decoder'] == 'nn', :].groupby(level='date').describe()

    rr_summary.to_csv(os.path.join(config.resultsdir, 'fits_offline', f'rr_summary_{mk_name}.csv'))
    ds_summary.to_csv(os.path.join(config.resultsdir, 'fits_offline', f'ds_summary_{mk_name}.csv'))
    nn_summary.to_csv(os.path.join(config.resultsdir, 'fits_offline', f'nn_summary_{mk_name}.csv'))

    def dopairedstats(metric, althypo, ):
        rrm = results.loc[results['decoder'] == 'rr', metric].droplevel('indayidx')
        nnm = results.loc[results['decoder'] == 'nn', metric].droplevel('indayidx')
        dsm = results.loc[results['decoder'] == 'ds', metric].droplevel('indayidx')

        rrnn_difference = np.mean((rrm - nnm)/rrm)
        rrds_difference = np.mean((rrm - dsm)/rrm)

        rrnn_testresult = stats.ttest_rel(rrm, nnm, alternative=althypo)
        rrds_testresult = stats.ttest_rel(rrm, dsm, alternative=althypo)

        return rrnn_difference, rrds_difference, rrnn_testresult, rrds_testresult

    metricstotest = ('mse', 'mse_lo', 'mse_hi', 'mean_hi', 'mean_lo', 'kl_div')
    althypo = ('greater', 'greater', 'greater', 'less', 'greater', 'greater')

    offlineFitResults = {'diff_rrnn':[], 'pval_rrnn':[], 'diff_rrds':[], 'pval_rrds':[]}
    for metric, alt in zip(metricstotest, althypo):
        a,b,c,d = dopairedstats(metric, alt)
        offlineFitResults['diff_rrnn'].append(a)
        offlineFitResults['pval_rrnn'].append(c.pvalue)
        offlineFitResults['diff_rrds'].append(b)
        offlineFitResults['pval_rrds'].append(d.pvalue)

    # Plot MSE over folds and over days
    sns.barplot(data=results, x='decoder', y='mse', palette=config.offlinePalette,
                hue_order=('rr','ds','nn'), ax=mseax, errorbar='se')
    # Plot KL-Divergence over folds and over days
    sns.barplot(data=results, x='decoder',y='kl_div',palette=config.offlinePalette,
                hue_order=('rr','ds','nn'), ax=klax, errorbar='se')
    if mk_name == 'Batman':
        mseax.set(ylim=[0, .45], yticks=[0, 0.3, 0.6],title='B. Open-loop error', ylabel='Mean-Squared Error',
                xticklabels=['RR', 'DS', 'tcFNN'])
    else:
        mseax.set(ylim=[0, .45], yticks=[0, 0.2, 0.4],title='B. Open-loop error', ylabel='Mean-Squared Error',
                xticklabels=['RR', 'DS', 'tcFNN'])
    klax.set(ylim=[0, .45], yticks=[0, 0.2, 0.4], title='D. Decoder fit to true distribution', ylabel='KL-Divergence',
             xticklabels=['RR', 'DS', 'tcFNN'])

    offlineFitResults = pd.DataFrame(offlineFitResults, index=metricstotest)
    offlineFitResults.to_csv(os.path.join(config.resultsdir, 'fits_offline', f'offlineFitResults_{mk_name}.csv'))

    return