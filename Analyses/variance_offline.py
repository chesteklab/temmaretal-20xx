import pdb
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import pickle
import seaborn as sns
import scipy.stats as stats
import yaml

from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import normalize
from utils import offline_training
from utils import nn_decoders
from utils import offline_metrics
import config


def variance_offline(mk_name, date, genfig, train_models=True, calculate_results=False, normalize_data = False):
    #setup pytorch stuff
    device = torch.device('cuda:0')
    dtype = torch.float

    # for now, run after decoder_fits_offline
    print('loading data')
    with open(os.path.join(config.data_dir,'fits_offline',f'data_{date}_{mk_name}.pkl'), 'rb') as f:
        trainData, testData, inIDXList, outIDXList, trial_num = pickle.load(f)
    print('data loaded')

    #separate and turn into tensors (no rr here)
    neu_train = torch.from_numpy(trainData['neu3D'][2]).to(device, dtype)
    vel_train = torch.from_numpy(trainData['vel'][2]).to(device, dtype)

    neu_test = torch.from_numpy(testData['neu3D']).to(device, dtype)
    vel_test = torch.from_numpy(testData['vel']).to(device, dtype)
    
    if normalize_data:
        # normalize neural data (new by JC)
        # neu_train has shape (num_trials, num_channels, num_time_hist)
        neu_mean = torch.mean(neu_train[:, :, 0], dim=0)  # has shape (num_channels,)
        neu_std = torch.std(neu_train[:, :, 0], dim=0)  # has shape (num_channels,)
        neu_mean = neu_mean.unsqueeze(0).unsqueeze(2)  # has shape (1, num_channels, 1)
        neu_std = neu_std.unsqueeze(0).unsqueeze(2)  # has shape (1, num_channels, 1)
        neu_train = (neu_train - neu_mean) / (neu_std + 1e-6)
        neu_test = (neu_test - neu_mean) / (neu_std + 1e-6)

        # normalize output velocities (new by JC)
        vel_mean = torch.mean(vel_train, dim=0)  # has shape (num_dof,)
        vel_std = torch.std(vel_train, dim=0)  # has shape (num_dof,)
        vel_mean = vel_mean.unsqueeze(0)  # has shape (1, num_dof)
        vel_std = vel_std.unsqueeze(0)  # has shape (1, num_dof)
        vel_train = (vel_train - vel_mean) / (vel_std + 1e-6)
        vel_test_norm = (vel_test - vel_mean) / (vel_std + 1e-6)

    train_ds = TensorDataset(neu_train, vel_train)
    #in this case val will be training data - getting training error
    val_ds = TensorDataset(neu_train, vel_train) 

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=len(val_ds),drop_last=True)

    num_models = 100
    models = {'TCN':[], 'TCN_noreg':[], 'TCN_nobn':[], 'TCN_nodp':[]}
    scalers = {'TCN':[], 'TCN_noreg':[], 'TCN_nobn':[], 'TCN_nodp':[]}
    train_histories = {'TCN':[], 'TCN_noreg':[], 'TCN_nobn':[], 'TCN_nodp':[]}
    val_histories = {'TCN':[], 'TCN_noreg':[], 'TCN_nobn':[], 'TCN_nodp':[]}

    in_size = neu_train.shape[1]
    num_states = 2

    if train_models:
        # models for main figure
        for i in range(num_models):
            # define keys and decoders to use
            keys = config.variance_models
            decoders = (nn_decoders.TCN, nn_decoders.TCNNoBN, nn_decoders.TCNNoDP, nn_decoders.TCNNoReg)

            # train two models each for the 4 types, one on normal data, one on normalized data
            for i, key in enumerate(keys):
                # initialize the decoder and optimizer
                model = offline_training.init_model(decoders[i], 'TCN', in_size, num_states)
                opt, scheduler = offline_training.init_opt(model, key)

                #train model
                vloss, tloss, loss_iters = offline_training.fit(model, key, opt, None, train_dl, val_dl, normalized=normalize_data)
                scaler = offline_training.generate_output_scaler(model, val_dl, num_outputs=num_states)

                # save model and training histories
                model = model.cpu()
                models[key].append(model)
                scalers[key].append(scaler)
                train_histories[key].append(tloss)
                val_histories[key].append(vloss)

        # save FNN decoders here
        if normalize_data:
            fpath = os.path.join(config.model_dir, 'variance_offline', f'tcFNNmodels_norm_{date}_{mk_name}.pkl')
        else:
            fpath = os.path.join(config.model_dir, 'variance_offline', f'tcFNNmodels_{date}_{mk_name}.pkl')
        
        with open(fpath, 'wb') as f:
            pickle.dump((models, scalers, train_histories, val_histories), f)
        print('All Decoders Saved')
    else:
        if calculate_results:
            if normalize_data:
                fpath = os.path.join(config.model_dir, 'variance_offline', f'tcFNNmodels_norm_{date}_{mk_name}.pkl')
            else:
                fpath = os.path.join(config.model_dir, 'variance_offline', f'tcFNNmodels_{date}_{mk_name}.pkl')
            
            with open(fpath, 'rb') as f:
                models, scalers, train_histories, val_histories = pickle.load(f)
            print('All Decoders Loaded')

    # Get predictions on the test set

    # iterate over all models and do some things - get predictions, metrics, and more
    sec = 1000
    vel_test = vel_test.cpu().detach().numpy()/config.binsize * sec

    keys = config.variance_models
    predictions = {}
    metrics = {'decoder':[], 'mse':[]}
    std_dev = {}
    hist = {}

    if normalize_data:
        result_filename = f'results_norm_{date}_{mk_name}.pkl'
    else:
        result_filename = f'results_{date}_{mk_name}.pkl'

    if calculate_results:
        for key in keys:           
            predictions[key] = np.zeros((vel_test.shape[0], vel_test.shape[1], num_models))

            # get test predictions, calculate MSE for each.
            for i in range(num_models):
                # get test predictions for models trained on non-normalized data
                model = models[key][i].to(device)
                model.eval()
                
                ## get prediction and scale to MR/sec
                pred = scalers[key][i].scale(model(neu_test)).cpu().detach().numpy()
                if normalize_data:
                    pred = (pred + vel_mean.cpu().detach().numpy()) * (vel_std.cpu().detach().numpy()+1e-6) # denormalize to put data on same scale as vel_test
                predictions[key][:,:,i] = pred/config.binsize * sec

                ## take model off gpu
                model = model.cpu()

                ## calculate MSE
                metrics['decoder'].append(key)
                metrics['mse'].append(offline_metrics.mse(vel_test.flatten(), predictions[key][:, :, i].flatten()))

            ## compute standard deviation across models at each time point:
            std_dev[key] = np.std(predictions[key], axis=2)

            # get mean and std val error history
            hist[key] = np.stack(val_histories[key])/config.binsize**2 * sec**2  # scaled to MR^2/sec^2

        metrics = pd.DataFrame(metrics)
        fpath = os.path.join(config.results_dir, 'variance_offline', result_filename)
        
        with open(fpath, 'wb') as f1:
            pickle.dump((predictions, metrics, std_dev, hist), f1)
    else:
        fpath = os.path.join(config.results_dir, 'variance_offline', result_filename)

        with open(fpath, 'rb') as f1:
            predictions, metrics, std_dev, hist = pickle.load(f1)

    # Start setting up figure
    varfig = None
    hist_ax = None
    mse_ax = None
    sd_ax = None

    if genfig:
        # set up the figure
        varfig = plt.figure(figsize=(12, 8))
        subfigs = varfig.subfigures(1, 2)
        trace_axes = subfigs[0].subplots(4,1, sharex=True, sharey=True)
        subfigs[0].suptitle('A. Example predictions for different models')
        analysis_axes = subfigs[1].subplots(3,1)
        mse_ax = analysis_axes[0]
        hist_ax = analysis_axes[1]
        sd_ax = analysis_axes[2]
        
        if mk_name == 'Joker':
            plotrange = np.arange(1499, 1562)
        else:
            plotrange = np.arange(599, 662)
        times = plotrange * config.binsize / sec
        traceargs = {'alpha': 0.2, 'lw': 1}

        for i,  key in enumerate(keys):
            ax = trace_axes[i]
            ax.plot(times, vel_test[plotrange, 0], 'k-', zorder=8, label='Hand Control', lw=3)

            ax.plot(times, predictions[key][plotrange, 0, :], **traceargs, c=config.offlineVariancePalette[key])
            ax.set(xlim=(times[0], times[-1]), xticks=(times[1],times[-2]),
                   ylabel='Velocity (AU/sec)' if i == 0 else None,
                   ylim=(-1.5, 2.5), yticks=[-1, 0, 1, 2], title=config.varianceLabels[i])

        trace_axes[3].set_xlabel('Experiment Time (sec)')

    return varfig, (hist_ax, mse_ax, sd_ax), metrics, hist, std_dev

def variance_offline_partII(mk_name, axs, metrics, history, std_dev, normalize_data=False):
    hist_ax, mse_ax, sd_ax = axs
    keys = config.variance_models
    lineargs = {'alpha': 1, 'lw': 1.5}
    average_std_dev = {'decoder':[],'average sd':[], 'sds':[], 'se_median':[]}
    sdd = []
    
    for i, key in enumerate(keys):
        hist = np.vstack([histi[key] for histi in history])
        hist_mean = np.mean(hist, axis=0)
        hist_std = np.std(hist, axis=0)

        hist_ax.plot(np.arange(len(hist_mean)), hist_mean, **lineargs, c=config.offlineVariancePalette[key])
        hist_ax.fill_between(np.arange(len(hist_mean)), hist_mean - hist_std, hist_mean + hist_std, alpha=0.3,
                             fc=config.offlineVariancePalette[key])


        sd_k = np.concatenate([sd[key] for sd in std_dev],axis=0)
        a_sd = np.median(sd_k)
        sdd.append(sd_k.flatten())

        average_std_dev['decoder'].append(key)
        average_std_dev['average sd'].append(a_sd)
        average_std_dev['se_median'].append(1.253*a_sd/np.sqrt(sd_k.size))
        average_std_dev['sds'].append(sd_k.flatten())

    if normalize_data:
        hist_ax.set(xlabel='Epochs (Log Scale)', title='C. Training error over time', xlim=(0, 15),
                    ylabel='MSE',xticks=(0,5,10,15))
    else:
        hist_ax.set(xlabel='Epochs (Log Scale)', title='C. Training error over time', xscale='log', xlim=(0, 100),
                    ylabel='MSE')

    sd_reformat_df = []
    for i, row in pd.DataFrame(average_std_dev).iterrows():
        decoder = row['decoder']
        for value in row['sds']:
            sd_reformat_df.append({'decoder': decoder, 'sd': value})
    sd_reformat_df = pd.DataFrame(sd_reformat_df)

    sns.barplot(ax=sd_ax, data=pd.DataFrame(average_std_dev), x='average sd', y='decoder', alpha=0.6,
                palette=config.offlineVariancePalette, hue_order=config.variance_models, order=keys)
    """
    sns.stripplot(data=sd_reformat_df, x='sd', y='decoder', hue_order=config.variance_models, order=keys,
                  palette=config.offlineVariancePalette, jitter=True, ax=sd_ax, alpha=0.7, size=4, zorder=0)
    """
    sd_ax.errorbar(x=average_std_dev['average sd'], y=average_std_dev['decoder'], color='black',
                   xerr=average_std_dev['se_median'], fmt='none', elinewidth=2, zorder=1)
    
    if mk_name == 'Joker':
        sd_ax.set(title='D. Median Prediction Deviations', xlabel='Median inter-model prediction SD across time',
                yticklabels=config.varianceTicks)#, xlim=(0,0.2))
    elif mk_name == 'Batman':
        if normalize_data:
            sd_ax.set(title='D. Median Prediction Deviations', xlabel='Median inter-model prediction SD across time',
                    yticklabels=config.varianceTicks)#, xlim=(0,0.3))
        else:
            sd_ax.set(title='D. Median Prediction Deviations', xlabel='Median inter-model prediction SD across time',
                    yticklabels=config.varianceTicks)#, xlim=(0,0.25))

    sns.barplot(ax=mse_ax, data=metrics, y='decoder', x='mse', estimator='mean', errorbar='se', alpha=0.6, errcolor='black',
                palette=config.offlineVariancePalette, hue_order=config.variance_models, order=keys)
    sns.stripplot(ax=mse_ax, data=metrics, y='decoder', x='mse', alpha=0.7, size=4,
                palette=config.offlineVariancePalette, hue_order=config.variance_models, order=keys, zorder=0)

    mse_ax.set(title='B. MSE on test set across days',xlabel='MSE',yticklabels=config.varianceTicks)#,xlim=(0,0.6))

    mse_summary = metrics.groupby('decoder').agg(('mean','std'))

    #do stats on mses, compare all to tcFNN
    tcfnn_mses = metrics.groupby('decoder').get_group('tcfnn').reset_index()[['mse']]
    msediffs = {'comparison':[],'diff':[],'pvalue':[], 'sd diff':[]}
    for label, group in metrics.groupby('decoder'):
        if label != 'tcfnn':
            group_mses = group.reset_index()['mse']
        else:
            continue
        msediffs['comparison'].append(f'tcfnn v {label}')
        msediffs['diff'].append((tcfnn_mses.mean() - group_mses.mean())/tcfnn_mses.mean())
        msediffs['pvalue'].append(stats.ttest_ind(tcfnn_mses, group_mses, alternative='less'))

        astd = pd.DataFrame(average_std_dev)
        tcfnnsd = astd.loc[astd['decoder'] == 'tcfnn', 'average sd'].to_numpy()[0]
        labelsd = astd.loc[astd['decoder'] == label, 'average sd'].to_numpy()[0]
        msediffs['sd diff'].append((tcfnnsd - labelsd)/tcfnnsd)

    msediffs = pd.DataFrame(msediffs)
    if normalize_data:
        msediffs.to_csv(os.path.join(config.results_dir, 'variance_offline', f'mse_diffs_norm_{mk_name}.csv'))
        mse_summary.to_csv(os.path.join(config.results_dir, 'variance_offline', f'mse_summary_norm_{mk_name}.csv'))
        pd.DataFrame(average_std_dev).to_csv(os.path.join(config.results_dir, 'variance_offline', f'avg_SD_norm_{mk_name}.csv'))
    else:
        msediffs.to_csv(os.path.join(config.results_dir, 'variance_offline', f'mse_diffs_{mk_name}.csv'))
        mse_summary.to_csv(os.path.join(config.results_dir, 'variance_offline', f'mse_summary_{mk_name}.csv'))
        pd.DataFrame(average_std_dev).to_csv(os.path.join(config.results_dir, 'variance_offline', f'avg_SD_{mk_name}.csv'))
