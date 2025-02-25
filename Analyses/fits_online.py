import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy import stats
import seaborn as sns
import pdb
import config
import utils.online_metrics
from utils.ztools import ZStructTranslator
from utils import offline_metrics
from utils import online_metrics


def fits_online(mk_name, date, runs, decoderlabels, trimlength = 5, offby2=False, preprocess=True):

    # load data from days - Including HC data. filter trials and separate runs by decoder
    if preprocess: # preprocess here is pretty light, just loading/creating z structs and saving them.
        # run through each day, load the runs, get z structs as dataframes, append to list. then, concatenate into
        # one large dataframe and save it out.
        zlist = []
        for i in np.arange(len(runs)):
            run = 'Run-{}'.format(str(runs[i]).zfill(3))
            fpath = os.path.join(config.raw_data_dir, mk_name, date, run)

            z = ZStructTranslator(fpath, os.path.join(config.data_dir, 'fits_online'), numChans=96, verbose=True)
            z = z.asdataframe()
            if decoderlabels[i] != 'HC': # if not a hand control run, filter by only decoder on trials.
                z = z[z['ClosedLoop'] == True] #make sure decode is on as well
            z = z[trimlength:]
            print(decoderlabels[i])
            print(np.sum(z['TrialSuccess'])/len(z['TrialSuccess']))
            z = z[z['TrialSuccess'] == True] # filter out unsuccessful trials
            z = z[z['BlankTrial'] == False] # remove blank trial

            z['Decoder'] = decoderlabels[i] # add decoder label to dataframe
            z['Run'] = run
            zlist.append(z)
        z_all = pd.concat(zlist, axis=0) #concatenate list into one large dataframe
        z_all = z_all.reset_index()
        z_all.to_pickle(os.path.join(config.data_dir, 'fits_online', f'data_{date}.pkl'))
        print('data saved')

    else:
        ## Load in saved data
        z_all = pd.read_pickle(os.path.join(config.data_dir, 'fits_online', f'data_{date}.pkl'))
        print('data loaded')

    # Figure Setup - Create figure and Subfigures
    onlinefitfig = plt.figure(figsize=(14,7))
    subfigs = onlinefitfig.subfigures(1,3, width_ratios=(3,2.5,2.5))
    subfigs[0].suptitle("A. Online positions")
    subfigs[1].suptitle('B. Online velocity distributions')
    subfigs[2].suptitle('C. Online performance metrics')

    # Create Axes for each subfigure
    posaxs = subfigs[0].subplots(3,1, sharex=True)

    distaxs = subfigs[1].subplots(4,1)
    dist_tops = distaxs[[0,2]]
    dist_bots = distaxs[[1,3]]

    metricaxs = subfigs[2].subplots(3,1)

    # Plot online position trace examples for both decoders and hand control
    z_RN = z_all[z_all['Decoder'] == 'RN']
    z_RK = z_all[z_all['Decoder'] == 'RK']
    z_HC = z_all[z_all['Decoder'] == 'HC']
    def plotOnlineTraces(z,posax,decoder):
        online_metrics.plotOnlinePositions(z, ax=posax)
        posax.set(ylabel='Extension (%)',xlabel=None, yticks=[0,25,50,75,100]) # check if flex or extend
        posax.set(title=decoder)
    plotOnlineTraces(z_RK[100:110],posaxs[1],"ReFIT KF (RK)")
    plotOnlineTraces(z_RN[102:112],posaxs[2],"Re-tcFNN (RN)")
    plotOnlineTraces(z_HC[100:110], posaxs[0], "Hand Control (HC)")

    # plot settings
    posaxs[1].legend()
    # posaxs[1].get_legend().legendHandles[0].set(edgecolor='k',facecolor=None)
    posaxs[2].set(xlabel='Time (sec)')

    for ax,color in zip(posaxs, [config.hcColor, config.kfColor, config.tcnColor]):
        [i.set_linewidth(2) for i in ax.spines.values()]
        [i.set_edgecolor(color) for i in ax.spines.values()]

    # Per day, plot velocity distributions with broken axes, and calculate the KL divergence
    porder = (z_RK, z_RN)
    labels = ('RK', 'RN')
    palette = (config.kfColor, config.tcnColor)
    kldivs = {'div':[], 'Decoder':[], 'counts':[], 'hc_counts':[]}
    for i, (zi, colors, decoder) in enumerate(zip(porder, palette, labels)):
        # top plot
        online_metrics.calcVelocityDistribution(zi, z_HC, plotResults=True, binrange=(-9, 9), numbins=100, binsize=50,
                                                      label=decoder, ax=dist_tops[i], color=colors)
        # bottom plot
        hist, binedges, hist_hc, binedges_hc, dist_fig = online_metrics.calcVelocityDistribution(zi, z_HC,
                                                                                                       plotResults=True,
                                                                                                       binrange=(-9, 9),
                                                                                                       numbins=100,
                                                                                                       binsize=50,
                                                                                                       label=decoder,
                                                                                                       ax=dist_bots[i],
                                                                                                       color=colors)

        # calculate kl divergence on each day
        f = hist_hc / np.sum(hist_hc) #scale histograms as pmfs
        g = hist / np.sum(hist)
        kldivs['div'].append(offline_metrics.kldiv(f,g))
        kldivs['Decoder'].append(decoder)
        kldivs['counts'].append(len(zi))
        kldivs['hc_counts'].append(len(z_HC))
        dist_tops[i].set(ylim=(0.5,3.5),xlim=(-4,4), xlabel=None, ylabel=None, title=None)
        dist_bots[i].set(ylim=(0, 0.2), xlim=(-4, 4), ylabel='Estimated Density', title=None,
                         yticks=[0,0.1,0.2])
        utils.online_metrics.drawBrokenAxes(dist_tops[i], dist_bots[i], d=0.015)

        dist_bots[i].get_legend().remove()
        dist_tops[i].get_legend().remove()

    # distribution plot settings
    dist_bots[0].set(xlabel=None, xticklabels=[])
    dist_tops[0].set(title='RK Velocity Distribution')
    dist_tops[1].set(title='RN Velocity Distribution')

    # Calculate performance metrics (time to first acquire, orbiting time)
    (tt, rt, ot) = online_metrics.calcTrialTimes(z_all, offBy2=offby2)
    clMetrics = pd.DataFrame(data={'TimeToTarget': rt, 'OrbitTime': ot})
    clMetrics['Decoder'] = z_all['Decoder']
    clMetrics.to_pickle(os.path.join(config.results_dir, 'fits_online', f'onlinefitmetrics_{date}.pkl'))

    kldivs = pd.DataFrame(data=kldivs)
    kldivs.to_pickle(os.path.join(config.results_dir, 'fits_online', f'onlinefitdivs_{date}.pkl'))

    return kldivs, metricaxs, dist_tops, onlinefitfig, clMetrics

def fits_online_partII(kldivs, ax, results):
    metricax = ax[0]
    distax = ax[1]

    # add orbited column for later difference of proportions test
    results['Orbited'] = results['OrbitTime'] != 0

    # get per day means, standard deviations, and orbit rates
    day_summaries = []
    for date, day_result in results.groupby('date'):
        # get mean tt, ot, and calc or for each day
        day_result = day_result[['TimeToTarget','OrbitTime','Orbited','Decoder']]

        tt = day_result.where(day_result['TimeToTarget'] != 0)
        tt = tt.groupby('Decoder')['TimeToTarget'].agg(('mean','std','count'))
        tt['HC diff'] = (tt.loc[:,'mean'] - tt.loc['HC','mean'])/tt.loc['HC','mean']

        ot = day_result.where(day_result['OrbitTime'] != 0)
        ot = ot.groupby('Decoder')['OrbitTime'].agg(('mean','std','count'))
        ot['HC diff'] = (ot.loc[:,'mean'] - ot.loc['HC','mean'])/ot.loc['HC','mean']

        orb = day_result.groupby('Decoder')['Orbited'].mean()
        orbcount = day_result.groupby('Decoder')['Orbited'].size()

        orbdiff = (orb - orb.loc['HC'])/orb.loc['HC']
        orb = pd.concat((orb, orbdiff,orbcount), keys=('rate', 'HC diff','count'), axis=1)
        
        day_summaries.append(pd.concat((tt,ot,orb),keys=('TT','OT','OR'),axis=1))

    
    kldivs.to_csv(os.path.join(config.results_dir, 'fits_online', 'kl_divs_w_counts.csv'))
    kldivs = kldivs.pivot(index='date',columns='Decoder',values='div').rename(columns={'RN':'div_RN','RK':'div_RK'})
    day_summaries = pd.concat(day_summaries, axis=0, keys=kldivs.index)

    # save results
    day_summaries.to_csv(os.path.join(config.results_dir, 'fits_online', 'online_fit_results.csv'))
    kldivs.to_csv(os.path.join(config.results_dir, 'fits_online', 'kl_divs.csv'))

    # do stats across days
    def runttest_ind(results, metric, althypo):
        results = results.where(results[metric] != 0)

        # get just the metric for rn and rk separately
        rn_metric = results.loc[results['Decoder'] == 'RN', :][metric]
        rk_metric = results.loc[results['Decoder'] == 'RK', :][metric]

        output = pd.DataFrame({'decoder':['RN','RK'],
                               'mean':[rn_metric.mean(), rk_metric.mean()], 
                               'counts':[rn_metric.size, rk_metric.size],
                               'std':[rn_metric.std(), rk_metric.std()]})
        
        return stats.ttest_ind(rn_metric, rk_metric, alternative=althypo), output

    tt_test, tt_output = runttest_ind(results, 'TimeToTarget', 'less')
    ot_test, ot_output = runttest_ind(results, 'OrbitTime', 'two-sided')

    # get total orbiting rate across days
    pop_sizes = results.groupby('Decoder').size()
    orbit_counts = results.groupby('Decoder').apply(lambda x: np.sum(x['Orbited']))
    orbit_props = orbit_counts / pop_sizes

    op_df = pd.DataFrame({'decoder':['RN','RK'],
                          'mean':[orbit_props['RN'],orbit_props['RK']],
                          'counts':[pop_sizes['RN'],pop_sizes['RK']],
                          'std':[0,0]})
    
    crossdayresults = pd.concat((tt_output, ot_output, op_df), keys=['tt','ot','or'])
    crossdayresults.to_csv(os.path.join(config.results_dir, 'fits_online', 'cross_day_metrics.csv'))
    
    # difference of proportions test
    # null hypothesis, 0 difference in proportion
    pdiff = orbit_props['RN'] - orbit_props['RK']
    phat = (orbit_counts['RN'] + orbit_counts['RK']) / (pop_sizes['RN'] + pop_sizes['RK'])
    sediff = np.sqrt(phat * (1 - phat) * (1 / pop_sizes['RN'] + 1 / pop_sizes['RK']))

    zscore = pdiff / sediff
    pval = 1 - stats.norm.cdf(np.abs(zscore))

    #consolidate and save results
    statsout = pd.Series(data={'RN TT < RK TT':tt_test.pvalue, 'RN OT != RK OT':ot_test.pvalue, 'RN OR > RK OR':pval})
    statsout.to_csv(os.path.join(config.results_dir,'fits_online','stats.csv'))

    # PLOTTING
    # add average kl-div across days to distribution plots
    klmeans = kldivs.mean(axis=0)
    klstds = kldivs.std(axis=0)

    textpos = (0.5, 0.65)
    distax[0].text(textpos[0], textpos[1], f" Mean KL-div from HC (over 3 days):\n {klmeans.iloc[0]:.2f} +/- {klstds.iloc[0]:.2f}",
                   transform=distax[0].transAxes, fontsize=mpl.rcParams['axes.labelsize'], ha='center')
    distax[1].text(textpos[0], textpos[1], f"Mean KL-div from HC:\n {klmeans.iloc[1]:.2f} +/- {klstds.iloc[1]:.2f}",
                   transform=distax[1].transAxes, fontsize=mpl.rcParams['axes.labelsize'], ha='center')

    # bar plots for time to target, orbiting rate, nonzero orbiting time
    
    sns.barplot(results[results['TimeToTarget'] != 0], x='TimeToTarget', y='date', hue='Decoder', errorbar='se', ax=metricax[0], 
                palette=config.online_palette[[0, 2, 1], :])
    sns.barplot(orbit_props.reset_index(), x=0, y='Decoder', ax=metricax[1], palette=config.online_palette[[0, 2, 1], :], order=['HC', 'RN', 'RK'])
    sns.barplot(results[results['OrbitTime'] != 0], x='OrbitTime', y='date', hue='Decoder', errorbar='se', ax=metricax[2], 
                palette=config.online_palette[[0, 2, 1], :])

    # plot settings
    metricax[0].set(title='Time-to-target', ylabel=None, xlabel='Time (s)', xlim=(0, 1500),
                  xticks=[0, 750, 1500], xticklabels=[0, .75, 1.5])
    metricax[1].set(title='Orbiting rate', ylabel=None, xlabel='Proportion',
               xticks=[0, .5, 1])
    metricax[2].set(title='Nonzero orbit time', ylabel=None, xlabel='Time (s)', xlim=(0, 1500),
                    xticks=[0, 750, 1500], xticklabels=[0, .75, 1.5])

