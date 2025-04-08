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
import yaml
import config
import utils.online_metrics
from utils.ztools import ZStructTranslator, getZFeats, sliceMiddleTrials
from utils import offline_training, nn_decoders, offline_metrics
from utils.offline_data import data_cleanup, split_offline_data

'''
Addressing the first block of the paper: Do NNs fit better than other methods?
- Offline Figure showing fit of predicted velocities of various approaches to hand control
'''
def fits_offline(mk_name, date, runs, preprocess=True, train_rr=True, train_ds=True, train_tcn=True, train_rnn = True, genfig=True,
                 short_day=False):
    #setup pytorch stuff
    device = torch.device('cuda:0')
    dtype = torch.float

    numFolds = 5

    # load and preprocess data if needed
    if preprocess:
        for i in range(len(runs)):
            run = 'Run-{}'.format(str(runs[i]).zfill(3))
            fpath = os.path.join(config.raw_data_dir, mk_name, date, run)
            zadd = ZStructTranslator(fpath, os.path.join(config.data_dir, 'fits_offline'), numChans=config.numChans)
            # remove unsuccessful trials
            zadd = zadd.asdataframe()
            zadd = zadd[zadd['TrialSuccess'] != 0]
            if i == 0:
                z = zadd
            else:
                z = pd.concat([z,zadd])

        #take middle 1000 trials and get z feats
        if short_day:
            zsliced = sliceMiddleTrials(z, 400)
            trainDD = getZFeats(zsliced[0:300], config.binsize, featList=['FingerAnglesTIMRL', 'NeuralFeature'])
            testDD = getZFeats(zsliced[300:], config.binsize, featList=['FingerAnglesTIMRL', 'NeuralFeature','TrialNumber'])
        else:
            zsliced = sliceMiddleTrials(z, 600)
            trainDD = getZFeats(zsliced[0:500], config.binsize, featList=['FingerAnglesTIMRL', 'NeuralFeature'])
            testDD = getZFeats(zsliced[500:], config.binsize, featList=['FingerAnglesTIMRL', 'NeuralFeature','TrialNumber'])

        # separate feats, add time history, add a column of ones for RR, and reshape data for NN.
        pretrainData = data_cleanup(trainDD)
        testData = data_cleanup(testDD)

        #split the training data into folds
        trainData, inIDXList, outIDXList = split_offline_data(pretrainData,numFolds)

        # get trialnumber for testDD
        trial_num = testDD['TrialNumber'][3:,0].astype(int)
        with open(os.path.join(config.data_dir,'fits_offline',f'data_{date}_{mk_name}.pkl'), 'wb') as f:
            pickle.dump((trainData, testData, inIDXList, outIDXList, trial_num), f)
    else:
        ## Load in saved data
        print('loading data')
        with open(os.path.join(config.data_dir,'fits_offline',f'data_{date}_{mk_name}.pkl'), 'rb') as f:
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
        with open(os.path.join(config.model_dir,'fits_offline',f'rr_models_{date}_{mk_name}.pkl'), 'wb') as f:
            pickle.dump(rr_models, f)
        print('RR Decoders Saved')
    else:
        with open(os.path.join(config.model_dir,'fits_offline',f'rr_models_{date}_{mk_name}.pkl'), 'rb') as f:
            rr_models = pickle.load(f)
        print('RR Decoders Loaded')

    # Train Dual-State Decoders
    if train_ds:
        ds_models = []
        for k in np.arange(numFolds):
            neu = trainData['neu2D'][k]
            vel = trainData['vel'][k]

            ds_models.append(offline_training.dsTrain(neu, vel))
        with open(os.path.join(config.model_dir, 'fits_offline', f'ds_models_{date}_{mk_name}.pkl'), 'wb') as f:
            pickle.dump(ds_models, f)
        print('DS Decoders Saved')
    else:
        with open(os.path.join(config.model_dir, 'fits_offline', f'ds_models_{date}_{mk_name}.pkl'), 'rb') as f:
            ds_models = pickle.load(f)
        print('DS Decoders Loaded')

    ## Train tcFNN decoder
    if train_tcn:
        epochs = 10
        nn_models = []
        scalers = []
        for k in np.arange(numFolds):
            model, scaler, _, _ = offline_training.train_nn(trainData['neu3D'][k], 
                                                            trainData['vel'][k], 
                                                            nn_decoders.TCN, 
                                                            'TCN',
                                                            normalize=False)
            nn_models.append(model)
            scalers.append(scaler)

        print(f'tcn models trained.')
        # save decoders and scalers
        with open(os.path.join(config.model_dir, 'fits_offline', f'tcnmodels_{date}_{mk_name}.pkl'), 'wb') as f:
            pickle.dump((nn_models, scalers), f)
        print('tcn models Saved')
    else:
        with open(os.path.join(config.model_dir, 'fits_offline', f'tcnmodels_{date}_{mk_name}.pkl'), 'rb') as f:
            nn_models, scalers = pickle.load(f)
        print('NN Decoders Loaded')
    
    # train rnn
    if train_rnn:
        rnn_models = []
        rnn_scalers = []
        norm_params = []
        for k in np.arange(numFolds):
            model, scaler, _, norms = offline_training.train_nn(trainData['neu3D'][k],
                                                               trainData['vel'][k], 
                                                               nn_decoders.RecurrentModel,
                                                               'LSTM_xnorm_ynorm')
            rnn_models.append(model)
            rnn_scalers.append(scaler)
            norm_params.append(norms)
        print(f'rnn models trained.')
        # save decoders and scalers
        with open(os.path.join(config.model_dir, 'fits_offline', f'rnn_models_{date}_{mk_name}.pkl'), 'wb') as f:
            pickle.dump((rnn_models, rnn_scalers, norm_params), f)
        print('rnn models saved, means and stds saved')
    else:
        with open(os.path.join(config.model_dir, 'fits_offline', f'rnn_models_{date}_{mk_name}.pkl'), 'rb') as f:
            rnn_models, rnn_scalers, norm_params = pickle.load(f)
        print('rnn models loaded, means and stds loaded')

    # Get predictions for each decoder
    rr_predictions = np.zeros((testData['vel'].shape[0], testData['vel'].shape[1], numFolds))
    nn_predictions = np.zeros_like(rr_predictions)
    rnn_predictions = np.zeros_like(rr_predictions)
    ds_predictions = np.zeros_like(rr_predictions)
    ds_probabilities = np.zeros((testData['vel'].shape[0], numFolds))
    for k in range(numFolds):
        rr = rr_models[k]
        ds = ds_models[k]
        tcn = nn_models[k].to(config.device)
        tcn_scaler = scalers[k]
        rnn = rnn_models[k].to(config.device)
        rnn_scaler = rnn_scalers[k]
        norms = norm_params[k] # [neu_mean, neu_std, vel_mean, vel_std]

        tcn.eval()
        rnn.eval()

        neu_test = torch.from_numpy(testData['neu3D']).to(config.device, config.dtype)
        tcn_yh = tcn(neu_test)

        neu_test_norm = (neu_test - norms[0]) / (norms[1] + 1e-6)
        rnn_yh = rnn(neu_test_norm)
        rnn_yh = rnn_scaler.scale(rnn_yh[0].cpu().detach().numpy())
        rnn_yh = rnn_yh * norms[3].cpu().detach().numpy() + norms[2].cpu().detach().numpy()

        nn_predictions[:,:,k] = tcn_scaler.scale(tcn_yh).cpu().detach().numpy()
        rnn_predictions[:,:,k] = rnn_yh
        rr_predictions[:,:,k] = offline_training.rrPredict(testData['neu2D'], rr)
        ds_predictions[:,:,k], pr = offline_training.dsPredict(testData['neu2D'], ds)
        ds_probabilities[:, k:k+1] = pr

    #scale predictions for au/sec
    binsize = config.binsize
    sec = 1000

    nn_predictions = nn_predictions/binsize * sec
    rnn_predictions = rnn_predictions/binsize * sec
    rr_predictions = rr_predictions/binsize * sec
    ds_predictions = ds_predictions/binsize * sec
    vel_test = testData['vel']/binsize * sec

    # Calculate Metrics
    sortidx = np.argsort(np.abs(vel_test.flatten()))  # sort absolute values of velocities.
    hi_vel_idx = sortidx[np.floor(len(sortidx) * 9 / 10).astype(int):]  # take top 10%
    lo_vel_idx = sortidx[0:np.ceil(len(sortidx) / 10).astype(int)]  # take bottom 10%

    hi_thr= np.abs(vel_test.flatten()[hi_vel_idx[0]])
    lo_thr = np.abs(vel_test.flatten()[lo_vel_idx[-1]])

    nbins = 99
    binmin = -9
    binmax = 9
    binsize = (binmax - binmin) / nbins
    binedges = np.linspace(-9, 9, nbins+1)
    decoders = ('rr','ds','tcn', 'rnn')
    preds = (rr_predictions, ds_predictions, nn_predictions, rnn_predictions)
    metrics = {'cc':[], 'mse':[], 'vaf':[],'mse_hi':[],'mse_lo':[],
                  'mean_hi':[],'mean_lo':[],'kl_div':[], 'decoder':[], 'fold':[]}

    with open(os.path.join(config.results_dir, 'fits_offline', f'predictions_all_models_{date}_{mk_name}.pkl'), 'wb') as f:
        pickle.dump((preds, vel_test), f)
    
    for k in np.arange(numFolds):
        for i, decoder in enumerate(decoders):
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

    fitFig = None
    mseax = None
    klax = None

    if genfig:
        dist_stats = {'decoder': [], 'shapiro_p': [], 'kl_div_gaussian': []}

        # Creating the Figure:
        fitFig = plt.figure(figsize=(20,8))

        subfigs = fitFig.add_gridspec(2,2, width_ratios = [4, 1])
        tracespec = subfigs[0,0].subgridspec(1,4)
        mseax = fitFig.add_subplot(subfigs[0,1])

        distspec = subfigs[1,0].subgridspec(2,4)
        klax = fitFig.add_subplot(subfigs[1,1])
        if mk_name == 'Batman':
            plotrange = np.arange(599, 662)
        else:
            plotrange = np.arange(1499, 1562)
        times = plotrange * config.binsize / sec
        histwidth = 3

        # choose one fold's decoders to look at (as to not choose between 5 graphs a day)
        fold = 2
        traceid = 0 #finger to use in traces
        predLabels = ('Ridge Regression RR', 'Sachs et al. 2016 DS', 'Willsey et al. 2022 tcFNN', 'LSTM')

        for i, pred in enumerate(preds):
            ax = fitFig.add_subplot(tracespec[i])
            ax.plot(times, vel_test[plotrange, traceid], color=config.hcColor, lw=histwidth)
            ax.plot(times, pred[plotrange, traceid, fold], color=config.offline_palette[i, :], lw=histwidth)

            if i == 1:
                ax.scatter(times, pred[plotrange, traceid, fold], c=ds_probabilities[plotrange,fold],
                           cmap=config.dsmap, zorder=10, vmin=0, vmax=1)
                cb_ax = inset_axes(ax, width="40%", height = "5%", loc=2)
                plt.colorbar(mappable=mpl.cm.ScalarMappable(cmap=config.dsmap), cax=cb_ax, orientation='horizontal',
                             label='Movement Likelihood')
                for spine in cb_ax.spines.values():
                    spine.set_visible(False)
            if i == 0:
                if mk_name == 'Batman':
                    ax.set(ylabel='Velocity (Flex/Sec)', yticks=(-2,0,2))
                else:
                    ax.set(ylabel='Velocity (Flex/Sec)', yticks=(-1,0,1,2))
            else:
                if mk_name == 'Batman':
                    ax.set_yticks((-2,0,2), labels=[])
                else:
                    ax.set_yticks((-1,0,1,2), labels=[])
            if mk_name == 'Batman':
                ax.set(xlabel='Time (sec)',title=predLabels[i], ylim=(-2.25,2.25),
                   xlim=(times[0],times[-1]),xticks=(30,31,32,33))
            else:
                ax.set(xlabel='Time (sec)',title=predLabels[i], ylim=(-1.5,2.5), xlim=(times[0],times[-1]), xticks=(75, 76, 77, 78))

            topax = fitFig.add_subplot(distspec[0,i])
            botax = fitFig.add_subplot(distspec[1,i])

            #plot the same data on both axes
            def histplot(ax, top=True, addlines=True):
                ax.hist(vel_test.flatten(), color=config.hcColor, density=True, bins=binedges)
                ax.hist(pred[:,:,:].flatten(), color=config.offline_palette[i,:], density=True, histtype='step',
                        bins=binedges, linewidth=histwidth)
                lineargs = {'linestyle':'-','color':'k', 'lw':3}
                arrowargs = {'color':'k','width':0.01}
                if top and addlines:
                    ax.annotate("", xy=(lo_thr, 3), xytext=(lo_thr+1, 3),
                                arrowprops=dict(arrowstyle="-|>", lw=3)) # from xytext to xy
                    ax.annotate("", xy=(0-lo_thr, 3), xytext=(0-lo_thr-1, 3),
                                arrowprops=dict(arrowstyle="-|>", lw=3))

                elif addlines:
                    ax.annotate("", xy=(hi_thr, .05), xytext=(hi_thr+1, .05), # from xy to xytext
                                arrowprops=dict(arrowstyle="<|-", lw=3))
                    ax.annotate("", xy=(0-hi_thr, .05), xytext=(0-hi_thr-1, .05),
                                arrowprops=dict(arrowstyle="<|-", lw=3))
                    
            def plotGauss(ax):
                mean = np.mean(pred[:,:,:].flatten())
                std = np.std(pred[:,:,:].flatten())
                """ # for binned visualization
                x = np.linspace(binmin + binsize/2, binmax - binsize/2, nbins)
                gauss = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
                ploty_gauss = np.zeros_like(plotx_gauss)
                for i in range(len(gauss)):
                    bin_range = [x[i] - binsize / 2, x[i] + binsize / 2]
                    curx = np.argwhere((plotx_gauss >= bin_range[0]) & (plotx_gauss <= bin_range[1]))
                    ploty_gauss[curx] = gauss[i]
                """
                plotx_gauss = np.linspace(binmin, binmax, 1000000)
                ploty_gauss = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((plotx_gauss - mean) ** 2) / (2 * std ** 2))
                topax.plot(plotx_gauss, ploty_gauss, color='darkorange', lw=histwidth, linestyle=':', alpha=1.0)
                botax.plot(plotx_gauss, ploty_gauss, color='darkorange', lw=histwidth, linestyle=':', alpha=1.0)

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
                plotGauss(topax)
                plotGauss(botax)
            
            idx = np.random.randint(0, len(pred[:,:,:].flatten()), 5000) # sample 5000 points from the distribution, maximum for shapiro test
            dist_stats['shapiro_p'].append(stats.shapiro(pred[:,:,:].flatten()[idx])[1])
            dist_stats['kl_div_gaussian'].append(offline_metrics.kl_div_gaussian(pred[:,:,:].flatten()[idx], binedges, binsize))
            dist_stats['decoder'].append(decoders[i])
            

            utils.online_metrics.drawBrokenAxes(topax, botax, d=0.015)
        dist_stats = pd.DataFrame(dist_stats)
        dist_stats.to_csv(os.path.join(config.results_dir, 'fits_offline', f'normality_results_{mk_name}.csv'))

    metrics = pd.DataFrame(metrics)

    return metrics, fitFig, mseax, klax

def fits_offline_partII(mk_name, results, mseax, klax):
    # summarize results within days

    rr_summary = results.loc[results['decoder'] == 'rr', :].groupby(level='date').describe()
    ds_summary = results.loc[results['decoder'] == 'ds', :].groupby(level='date').describe()
    tcn_summary = results.loc[results['decoder'] == 'tcn', :].groupby(level='date').describe()
    rnn_summary = results.loc[results['decoder'] == 'rnn', :].groupby(level='date').describe()

    rr_summary.to_csv(os.path.join(config.results_dir, 'fits_offline', f'rr_summary_{mk_name}.csv'))
    ds_summary.to_csv(os.path.join(config.results_dir, 'fits_offline', f'ds_summary_{mk_name}.csv'))
    tcn_summary.to_csv(os.path.join(config.results_dir, 'fits_offline', f'tcn_summary_{mk_name}.csv'))
    rnn_summary.to_csv(os.path.join(config.results_dir, 'fits_offline', f'rnn_summary_{mk_name}.csv'))

    def dopairedstats(metric, althypo, ):
        rrm = results.loc[results['decoder'] == 'rr', metric].droplevel('indayidx')
        tcnm = results.loc[results['decoder'] == 'tcn', metric].droplevel('indayidx')
        dsm = results.loc[results['decoder'] == 'ds', metric].droplevel('indayidx')
        rnnm = results.loc[results['decoder'] == 'rnn', metric].droplevel('indayidx')

        rrtcn_difference = np.mean((rrm - tcnm)/rrm)
        rrds_difference = np.mean((rrm - dsm)/rrm)
        rrrnn_difference = np.mean((rrm - rnnm)/rrm)

        rrtcn_testresult = stats.ttest_rel(rrm, tcnm, alternative=althypo)
        rrds_testresult = stats.ttest_rel(rrm, dsm, alternative=althypo)
        rrrnn_testresult = stats.ttest_rel(rrm, rnnm, alternative=althypo)

        return rrtcn_difference, rrds_difference, rrrnn_difference, rrtcn_testresult, rrds_testresult, rrrnn_testresult

    metricstotest = ('mse', 'mse_lo', 'mse_hi', 'mean_hi', 'mean_lo', 'kl_div')
    althypo = ('greater', 'greater', 'greater', 'less', 'greater', 'greater')

    offlineFitResults = {'diff_rr_tcn':[], 'pval_rr_tcn':[], 'diff_rr_ds':[], 'pval_rr_ds':[], 'diff_rr_rnn':[],'pval_rr_rnn':[]}
    for metric, alt in zip(metricstotest, althypo):
        a,b,c,d,e,f = dopairedstats(metric, alt)
        offlineFitResults['diff_rr_tcn'].append(a)
        offlineFitResults['pval_rr_tcn'].append(d.pvalue)
        offlineFitResults['diff_rr_ds'].append(b)
        offlineFitResults['pval_rr_ds'].append(e.pvalue)
        offlineFitResults['diff_rr_rnn'].append(c)
        offlineFitResults['pval_rr_rnn'].append(f.pvalue)

    # Plot MSE over folds and over days
    sns.barplot(data=results, x='decoder', y='mse', palette=config.offline_palette,
                ax=mseax, alpha=0.6, errorbar='se', legend=False)
    sns.stripplot(results, x='decoder', y='mse', palette=config.offline_palette,
                ax=mseax, alpha=0.7, zorder=1)
    # # Plot KL-Divergence over folds and over days
    sns.barplot(data=results, x='decoder', y='kl_div', palette=config.offline_palette,
                ax=klax, alpha=0.6, errorbar='se', legend=False)
    sns.stripplot(results, x='decoder', y='kl_div', palette=config.offline_palette,
                  ax=klax, alpha=0.7, zorder=1)

    mseax.set(title='B. Open-loop prediction error', xlabel='decoder', xticklabels=['RR','DS','TCN','LSTM'], ylabel='Mean-Squared Error')
    klax.set(title='D. Decoder fit to true distribution', xlabel='decoder', xticklabels=['RR','DS','TCN','LSTM'], ylabel='KL-Divergence')

    offlineFitResults = pd.DataFrame(offlineFitResults, index=metricstotest)
    offlineFitResults.to_csv(os.path.join(config.results_dir, 'fits_offline', f'offlineFitResults_{mk_name}.csv'))

    return