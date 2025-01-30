import pdb
import pandas as pd
import config
from Analyses.fits_offline import fits_offline, fits_offline_partII
from Analyses.fits_online import fits_online, fits_online_partII
from Analyses.variance_offline import variance_offline, variance_offline_partII
from Analyses.context_offline import context_offline, context_offline_partII
from Analyses.variance_online import variance_online, variance_online_partII
import os
import pickle
import matplotlib.pyplot as plt

## Offline Fit of Velocity Distribution ################################################################################
run_section = False
if run_section:
    mk_name = 'Joker'
    dates = ['2021-02-16',
             '2021-04-12',
             '2022-06-16',
             '2022-09-06']
    runs = [3, 3, 2, 3]
    run_part1 = True
    finalfig = None
    fignum = 1
    if run_part1:
        results = []
        for i in range(len(dates)):
            date = dates[i]
            run = runs[i]

            genfig = i == fignum

            metrics, fitfig, mseax, klax = fits_offline(mk_name, date, run, preprocess=True, train_rr=True,
                                                        train_ds=True, train_nn=True, genfig=genfig)
            results.append(metrics)

            if genfig:
                finalfig = fitfig
                finalaxs = (mseax, klax)

        results = pd.concat(results, keys=dates, names = ['date','indayidx'], axis=0).set_index('fold', append=True)
        with open(os.path.join(config.results_dir, 'fits_offline', f'offlineFitResults.pkl'), 'wb') as f:
            pickle.dump((results, finalfig, finalaxs), f)
    else:
        with open(os.path.join(config.results_dir, 'fits_offline', f'offlineFitResults.pkl'), 'rb') as f:
            results, finalfig, finalaxs = pickle.load(f)

    fits_offline_partII(results, finalaxs[0], finalaxs[1])
    finalfig.savefig(os.path.join(config.results_dir, 'fits_offline', f'offlineFitFigure_{dates[fignum]}'))

# Online Velocity Distribution Comparisons #############################################################################
run_section = True
if run_section:
    mk_name = 'Joker'
    dates = ['2020-09-12',
             '2020-09-19',
             '2021-07-31'] #add another opt day
    runs = [[3, 10, 11, 12, 13],
            [4, 10, 11, 12, 13],
            [3, 7, 8, 9, 10]]
    decoderlabels = [['HC', 'RN', 'RK', 'RN', 'RK'],
                     ['HC', 'RK', 'RN', 'RK', 'RN'],
                     ['HC', 'RN', 'RK', 'RN', 'RK']]
    offby2 = [True, True, False]
    kldivs= []
    finalfig = None
    finalax = None
    
    results = []
    for i, (date, run, dclabs, off2) in enumerate(zip(dates, runs, decoderlabels, offby2)):
        genfig = i == 2

        kldiv, ax, distaxs, fig, metrics = fits_online(mk_name, date, run, dclabs, offby2=off2,
                                              preprocess=True)
        kldivs.append(kldiv)

        results.append(metrics)
        if genfig:
            finalfig = fig
            finalax = (ax, distaxs)
        else:
            plt.close(fig)
    results = pd.concat(results, keys=dates, names=['date', 'indayidx']).reset_index()
    kldivs = pd.concat(kldivs, keys=dates, names=['date'],axis=0).reset_index().drop('level_1',axis=1)

    fits_online_partII(kldivs, finalax, results)
    finalfig.savefig(os.path.join(config.resultsdir,'fits_online',f'onlineFitFigure_{dates[0]}.pdf'))

# Offline tcFNN Training Variance ######################################################################################
run_section = True

# take four days of offline data - same days as used for offline fit section
if run_section:
    dates = ['2021-02-16','2021-04-12', '2022-06-16', '2022-09-06']
    genfig = [False, True, False, False]
    fig = None
    axes = None
    results = []
    hists = []
    sds = []

    fig_n = None
    axes_n = None
    results_n = []
    hists_n = []
    sds_n = []

    # run the analysis for each day
    for date, gfig in zip(dates, genfig):
        #run variance offline analysis with standard data
        varfig, axs, metrics, hist, std_dev = variance_offline(date, gfig, train_models=False, calculate_results=False)
        if gfig:
            axes = axs
            fig = varfig
        results.append(metrics)
        hists.append(hist)
        sds.append(std_dev)

        #run  variance offline analysis with normalized data
        varfig, axs, metrics, hist, std_dev = variance_offline(date, gfig, train_models=False, calculate_results=False, normalize_data=True)
        if gfig:
            axes_n = axs
            fig_n = varfig
        results_n.append(metrics)
        hists_n.append(hist)
        sds_n.append(std_dev)

    #concatenate all the metrics for each model (MSE, VAF, Corr, etc) and save
    results = pd.concat(results, keys=dates, names=['date', 'indayidx'], axis=0)
    results_n = pd.concat(results_n, keys=dates, names=['date', 'indayidx'], axis=0)
    variance_offline_partII(axes, results, hists, sds, normalize_data=False)
    variance_offline_partII(axes_n, results_n, hists_n, sds_n, normalize_data=True)

    fig.savefig(os.path.join(config.resultsdir, 'variance_offline','offline_variance_figure.pdf'))
    fig_n.savefig(os.path.join(config.resultsdir, 'variance_offline','offline_variance_NORM_figure.pdf'))
    plt.show()

# Online tcFNN Training Variance #######################################################################################
run_section = False

if run_section:
    mk_name = 'Joker'

    dates = ['2022-02-02',
             '2023-01-31',
             '2023-02-07',
             '2023-02-14']
    runs = [[4,5,6,7,9],
            [8,9,10,11,12],
            [4,5,6,7,8],
            [12,13,14,15,16]]
    labels = [[1,2,3,4,5],
            [1,2,3,4,5],
            [1,1,1,1,1],
            [1,1,1,1,1]]
    results = []

    run_first = False
    for date, runs, labs in zip(dates, runs, labels):
        if run_first:
            results.append(variance_online(config.serverpath, mk_name, date, runs, labs,
                                           trimlength=5, preprocess=False))
        else:
            results.append(pd.read_pickle(os.path.join(config.resultsdir,
                                                       'variance_online',
                                                       f'onlinevariancemetrics_{date}.pkl')))

    variance_online_partII(results)

# Context Shifting Offline #############################################################################################
run_section = False

if run_section:
    firstpart = True
    if firstpart:
        results = []
        mk_name = 'Joker'
        dates = ['2022-05-31',
                 '2022-06-02',
                 '2023-01-17',
                 '2023-04-07',
                 '2023-04-11']
        runs = ((2, 5, 7, 9),
                (2, 4, 6, 8),
                (2, 4, 6, 8),
                (3, 5, 8, 10),
                (3, 5, 7, 9))
        labels = [['Normal', 'Wrist', 'SprWrst', 'Spring'],
                  ['Normal', 'Spring', 'SprWrst', 'Wrist'],
                  ['Normal', 'SprWrst', 'Spring', 'Wrist'],
                  ['Spring', 'SprWrst', 'Normal', 'Wrist'],
                  ['SprWrst', 'Wrist', 'Normal', 'Spring']]

        for date, run, label in zip(dates, runs, labels):
            metrics = context_offline(config.serverpath, mk_name, date, run, label,
                                                      preprocess=False, train_rr=False, train_nn=False)
            results.append(metrics)

        results = pd.concat(results, axis=0).reset_index()
        results.to_csv(os.path.join(config.resultsdir, 'context_offline','resultsAlldays.csv'))
        with open(os.path.join(config.resultsdir, 'context_offline', f'contextResults.pkl'), 'wb') as f:
            pickle.dump(results, f)
    else:
        with open(os.path.join(config.resultsdir, 'context_offline', f'contextResults.pkl'), 'rb') as f:
            results = pickle.load(f)
    context_offline_partII(results, '2022-06-02')

plt.show()