import config
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

import utils.online_metrics
from utils.ztools import ZStructTranslator, sliceMiddleTrials
from utils import online_metrics
import seaborn as sns

def variance_online(serverpath, mk_name, date, runs, decoderlabels,
                        trimlength = 5, preprocess=True):
    if preprocess: # preprocess here is pretty light, just loading/creating z structs and saving them.
        # run through each day, load the runs, get z structs as dataframes, append to list. then, concatenate into
        # one large dataframe and save it out.
        zlist = []
        for i in np.arange(len(runs)):
            run = 'Run-{}'.format(str(runs[i]).zfill(3))
            fpath = os.path.join(serverpath, mk_name, date, run)

            z = ZStructTranslator(fpath, numChans=96)
            z = z.asdataframe()
            if decoderlabels[i] != 'HC': # if not a hand control run, filter by only decoder on trials.
                z = z[z['ClosedLoop'] == True] #make sure decode is on as well
            z = z[trimlength:]
            z = z[z['BlankTrial'] == False] # remove blank trial
            z = z[z['TargetHoldTime'] != 750] # remove trials with 750 hold time

            # take middle 100 trials - all trials if less than 100
            z = sliceMiddleTrials(z, 100)

            z['Run'] = i
            z['Decoder'] = decoderlabels[i]
            zlist.append(z)

        z_all = pd.concat(zlist, axis=0) #concatenate list into one large dataframe
        z_all = z_all.reset_index()
        z_all.to_pickle(os.path.join(config.data_dir, 'variance_online', f'data_{date}.pkl'))
        print('data saved')

    else:
        ## Load in saved data
        z_all = pd.read_pickle(os.path.join(config.data_dir, 'variance_online', f'data_{date}.pkl'))
        print('data loaded')

    # Column 2 Bottom and Column 3: Online Performance Metrics
    (tt, at, ot) = online_metrics.calcTrialTimes(z_all)
    br = utils.online_metrics.calcBitRate(z_all)
    clMetrics = pd.DataFrame(data={'TrialTime': tt, 'AcquireTime': at, 'OrbitTime': ot, 'BitRate':br})
    clMetrics['Run'] = z_all['Run']
    clMetrics['Decoder'] = z_all['Decoder']
    clMetrics['TrialSuccess'] = z_all['TrialSuccess'].astype(bool)
    clMetrics.to_pickle(os.path.join(config.results_dir, 'variance_online', f'onlinevariancemetrics_{date}.pkl'))

    return clMetrics

def variance_online_partII(results):
    onlineVarFig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,5), layout='constrained')

    for i, (res, ax) in enumerate(zip(results,axs.T.flatten())):
        # create boxplots for each day showing the bitrate
        sns.boxplot(res, x='Run', y='BitRate', hue='Decoder', ax=ax, dodge=False)

        medians = res.groupby(['Run'])['BitRate'].median().values
        nobs = res['Run'].value_counts().values
        nobs = [str(x) for x in nobs.tolist()]
        nobs = ["n: " + i for i in nobs]

        pos = range(len(nobs))
        for tick, label in zip(pos, ax.get_xticklabels()):
            ax.text(pos[tick],
                    medians[tick] + .1,
                    nobs[tick],
                    ha='center',
                    color='w',
                    weight='regular')

        ax.set(yticks=[0,2,4,6], ylim=(-.5,6.5))
        ax.get_legend().remove()
        # prep trial by trial results for nested anova
        results[i] = results[i].loc[results[i].TrialSuccess == True, :] # remove unsuccessful trials
        results[i]['Day'] = f'Day{i + 1}'  # add a day label to each day
        results[i].loc[:, 'Run'] = results[i]['Run'] + 10 * i # make run labels unique across days

    axs[0, 0].set(title='Five tcFNNs trained on identical data', ylabel='Throughput (bits/s)', xticklabels=[], xlabel=None)
    axs[1, 0].set(ylabel='Throughput (bits/s)')
    axs[0, 1].set(title='One tcFNN tested in succession', ylabel=None, xticklabels=[], yticklabels=[], xlabel=None)
    axs[1, 1].set(ylabel=None, yticklabels=[])
    # save figure
    onlineVarFig.savefig(os.path.join(config.results_dir, 'variance_online', 'onlineVarFigure.pdf'))

    # prep trial by trial results for nested anova (Luis' analysis)
    results_df = pd.concat(results)
    results_df["Day"] = results_df['Day'].astype('category')
    results_df["Run"] = results_df['Run'].astype('category')
    results_df.to_csv(os.path.join(config.results_dir, 'variance_online', 'metrics_successful_full.csv'))

    # then, run nested_anova.R

    #save means
