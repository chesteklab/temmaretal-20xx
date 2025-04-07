import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from scipy import stats
import seaborn as sns
import pdb
import csv
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
        succ_counts = {'HC':0, 'RN':0, 'RK':0, 'NN':0}
        num_trials = {'HC':0, 'RN':0, 'RK':0, 'NN':0}
        zlist = []
        for i in np.arange(len(runs)):
            run = 'Run-{}'.format(str(runs[i]).zfill(3))
            fpath = os.path.join(config.raw_data_dir, mk_name, date, run)

            z = ZStructTranslator(fpath, os.path.join(config.data_dir, 'fits_online'), numChans=96, verbose=True)
            z = z.asdataframe()
            if decoderlabels[i] != 'HC': # if not a hand control run, filter by only decoder on trials.
                z = z[z['ClosedLoop'] == True] #make sure decode is on as well
            z = z[trimlength:]
            z = z[z['BlankTrial'] == False] # remove blank trial
            num_trials[decoderlabels[i]] = num_trials[decoderlabels[i]] + len(z)
            
            z = z[z['TrialSuccess'] == True] # filter out unsuccessful trials
            succ_counts[decoderlabels[i]] = succ_counts[decoderlabels[i]] + len(z)

            z['Decoder'] = decoderlabels[i] # add decoder label to dataframe
            z['Run'] = run
            zlist.append(z)

        succ_df = pd.DataFrame.from_dict(succ_counts, orient='index', columns=['successes'])
        succ_df['trial_counts'] = succ_df.index.map(num_trials)
        succ_df.to_csv(os.path.join(config.results_dir, 'fits_online', f'success_{date}_{mk_name}.csv'))
            
        z_all = pd.concat(zlist, axis=0) #concatenate list into one large dataframe
        z_all = z_all.reset_index()
        z_all.to_pickle(os.path.join(config.data_dir, 'fits_online', f'data_{date}_{mk_name}.pkl'))
        print('data saved')
    else:
        ## Load in saved data
        z_all = pd.read_pickle(os.path.join(config.data_dir, 'fits_online', f'data_{date}_{mk_name}.pkl'))
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
    # posaxs[1].legend()
    # posaxs[1].get_legend().legend_handles[0].set(edgecolor='k',facecolor=None)
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
    clMetrics.to_pickle(os.path.join(config.results_dir, 'fits_online', f'onlinefitmetrics_{date}_{mk_name}.pkl'))

    kldivs = pd.DataFrame(data=kldivs)
    kldivs.to_pickle(os.path.join(config.results_dir, 'fits_online', f'onlinefitdivs_{date}_{mk_name}.pkl'))
    return kldivs, metricaxs, dist_tops, onlinefitfig, clMetrics

def fits_online_partII(mk_name, kldivs, ax, results):
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

    
    kldivs.to_csv(os.path.join(config.results_dir, 'fits_online', f'kl_divs_w_counts_{mk_name}.csv'))
    kldivs = kldivs.pivot(index='date',columns='Decoder',values='div').rename(columns={'RN':'div_RN','RK':'div_RK'})
    day_summaries = pd.concat(day_summaries, axis=0, keys=kldivs.index)

    # save results
    day_summaries.to_csv(os.path.join(config.results_dir, 'fits_online', f'online_fit_results_{mk_name}.csv'))
    kldivs.to_csv(os.path.join(config.results_dir, 'fits_online', f'kl_divs_{mk_name}.csv'))

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
    orbit_props = orbit_props.reindex(['HC', 'RN', 'RK'])

    op_df = pd.DataFrame({'decoder':['RN','RK'],
                          'mean':[orbit_props['RN'],orbit_props['RK']],
                          'counts':[pop_sizes['RN'],pop_sizes['RK']],
                          'std':[0,0]})
    
    crossdayresults = pd.concat((tt_output, ot_output, op_df), keys=['tt','ot','or'])
    crossdayresults.to_csv(os.path.join(config.results_dir, 'fits_online', f'cross_day_metrics_{mk_name}.csv'))
    
    # difference of proportions test
    # null hypothesis, 0 difference in proportion
    pdiff = orbit_props['RN'] - orbit_props['RK']
    phat = (orbit_counts['RN'] + orbit_counts['RK']) / (pop_sizes['RN'] + pop_sizes['RK'])
    sediff = np.sqrt(phat * (1 - phat) * (1 / pop_sizes['RN'] + 1 / pop_sizes['RK']))

    zscore = pdiff / sediff
    pval = 1 - stats.norm.cdf(np.abs(zscore))

    #consolidate and save results
    statsout = pd.Series(data={'RN TT < RK TT':tt_test.pvalue, 'RN OT != RK OT':ot_test.pvalue, 'RN OR > RK OR':pval})
    statsout.to_csv(os.path.join(config.results_dir,'fits_online',f'stats_{mk_name}.csv'))

    # PLOTTING
    # add average kl-div across days to distribution plots
    klmeans = kldivs.mean(axis=0)
    klstds = kldivs.std(axis=0)

    textpos = (0.5, 0.65)
    distax[0].text(textpos[0], textpos[1], f" Mean KL-div from HC (over 3 days):\n {klmeans.iloc[0]:.2f} +/- {klstds.iloc[0]:.2f}",
                transform=distax[0].transAxes, fontsize=mpl.rcParams['axes.labelsize'], ha='center')
    distax[1].text(textpos[0], textpos[1], f"Mean KL-div from HC:\n {klmeans.iloc[1]:.2f} +/- {klstds.iloc[1]:.2f}",
                transform=distax[1].transAxes, fontsize=mpl.rcParams['axes.labelsize'], ha='center')
    
    
    #Shows horizontal bars with day means
    # bar plots for time to target, orbiting rate, nonzero orbiting time
    sns.barplot(results[results['TimeToTarget'] != 0], x='TimeToTarget', y='Decoder', errorbar='se', ax=metricax[0], 
                palette=config.online_palette[[0, 2, 1], :], alpha=0.6, errcolor='black')
    sns.stripplot(results[results['TimeToTarget'] != 0], x='TimeToTarget', y='Decoder', 
                  palette=config.online_palette[[0, 2, 1], :], ax=metricax[0], alpha=0.7, zorder=1, size=4)
    """
    # Code for plotting day means
    day_means = results[results['TimeToTarget'] != 0].groupby(['date', 'Decoder'], as_index=False)['TimeToTarget'].mean()
    sns.scatterplot(data=day_means, x='TimeToTarget', y='Decoder', hue='Decoder', palette=config.online_palette[[0, 1, 2], :], 
                    ax=metricax[0], alpha = .7, edgecolor = 'none', zorder=2)
    """

    # Bar plot for Orbiting Rate
    sns.barplot(data=orbit_props.reset_index(), x=0, y='Decoder', 
                ax=metricax[1], palette=config.online_palette[[0, 2, 1], :], alpha=1)

    sns.barplot(results[results['OrbitTime'] > 0], x='OrbitTime', y='Decoder', errorbar='se', ax=metricax[2], errcolor='black',
                palette=config.online_palette[[0, 2, 1], :], alpha=0.6)
    sns.stripplot(results[results['OrbitTime'] != 0], x='OrbitTime', y='Decoder', 
                  palette=config.online_palette[[0, 2, 1], :], ax=metricax[2], alpha=0.7, size=4, zorder=1)

    # plot settings
    metricax[0].set(title='Time-to-target', ylabel='Decoder', xlabel='Time (s)', yticklabels = ['HC', 'RN', 'RK'])

    metricax[1].set(title='Orbiting rate', ylabel='Decoder', xlabel='Proportion', yticklabels = ['HC', 'RN', 'RK'])

    metricax[2].set(title='Nonzero orbit time', ylabel='Decoder', xlabel='Time (s)', yticklabels = ['HC', 'RN', 'RK'])
        
def fits_online_w(mk_name, date, runs, decoderlabels, trimlength = 5, offby2=False, preprocess=True, genfig = False):
    
    # load data from days - Including HC data. filter trials and separate runs by decoder
    if preprocess: # preprocess here is pretty light, just loading/creating z structs and saving them.
        # run through each day, load the runs, get z structs as dataframes, append to list. then, concatenate into
        # one large dataframe and save it out.
        succ_counts = {'HC':0, 'RN':0, 'RK':0, 'NN':0}
        num_trials = {'HC':0, 'RN':0, 'RK':0, 'NN':0}
        zlist = []
        for i in np.arange(len(runs)):
            run = 'Run-{}'.format(str(runs[i]).zfill(3))
            fpath = os.path.join(config.raw_data_dir, mk_name, date, run)

            z = ZStructTranslator(fpath, os.path.join(config.data_dir, 'fits_online'), numChans=96, verbose=True)
            z = z.asdataframe()
            if decoderlabels[i] != 'HC': # if not a hand control run, filter by only decoder on trials.
                z = z[z['ClosedLoop'] == True] #make sure decode is on as well
            z = z[trimlength:]
            z = z[z['BlankTrial'] == False] # remove blank trial

            if run == 'Run-009' and date == '2020-12-08':
                z = z[0:-84] # from willsey processing code and notes, gave up at this point

            num_trials[decoderlabels[i]] = num_trials[decoderlabels[i]] + len(z)
            
            z = z[z['TrialSuccess'] == True] # filter out unsuccessful trials
            succ_counts[decoderlabels[i]] = succ_counts[decoderlabels[i]] + len(z)

            z['Decoder'] = decoderlabels[i] # add decoder label to dataframe
            z['Run'] = run
            zlist.append(z)

        succ_df = pd.DataFrame.from_dict(succ_counts, orient='index', columns=['successes'])
        succ_df['trial_counts'] = succ_df.index.map(num_trials)
        succ_df.to_csv(os.path.join(config.results_dir, 'fits_online', f'success_{date}_{mk_name}.csv'))

        z_all = pd.concat(zlist, axis=0) #concatenate list into one large dataframe
        z_all = z_all.reset_index()
        z_all.to_pickle(os.path.join(config.data_dir, 'fits_online', f'data_{date}_{mk_name}.pkl'))
        print('data saved')
    else:
        ## Load in saved data
        z_all = pd.read_pickle(os.path.join(config.data_dir, 'fits_online', f'data_{date}_{mk_name}.pkl'))
        print('data loaded')

    # Figure Setup - Create figure and Subfigures
    onlinefitfig = plt.figure(figsize=(14,10))
    subfigs = onlinefitfig.subfigures(1,3, width_ratios=(3,2.5,2.5))
    subfigs[0].suptitle("A. Online positions")
    subfigs[1].suptitle('B. Online velocity distributions')
    subfigs[2].suptitle('C. Online performance metrics')

    # Create Axes for each subfigure
    posaxs = subfigs[0].subplots(4,1, sharex=True)

    distaxs = subfigs[1].subplots(6,1)
    dist_tops = distaxs[[0,2,4]]
    dist_bots = distaxs[[1,3,5]]

    metricaxs = subfigs[2].subplots(3,1)

    # Plot online position trace examples for both decoders and hand control
    z_RN = z_all[z_all['Decoder'] == 'RN']
    z_RK = z_all[z_all['Decoder'] == 'RK']
    z_HC = z_all[z_all['Decoder'] == 'HC']
    z_NN = z_all[z_all['Decoder'] == 'NN']
    def plotOnlineTraces(z,posax,decoder):
        online_metrics.plotOnlinePositions(z, ax=posax)
        posax.set(ylabel='Extension (%)',xlabel=None, yticks=[0,25,50,75,100]) # check if flex or extend
        posax.set(title=decoder)
    
    if genfig:
        z_RK_plot = ZStructTranslator(os.path.join(config.raw_data_dir, mk_name, '2020-12-08', 'Run-007'), 
                                      os.path.join(config.data_dir, 'fits_online'), numChans=96, verbose=True)
        z_RK_plot = z_RK_plot.asdataframe()
        z_RK_plot = z_RK_plot[trimlength:]
        z_RK_plot = z_RK_plot[z_RK_plot['TrialSuccess'] == True] # filter out unsuccessful trials
        z_RK_plot = z_RK_plot[z_RK_plot['BlankTrial'] == False] # remove blank trial

        z_RK_HC_plot = ZStructTranslator(os.path.join(config.raw_data_dir, mk_name, '2020-12-08', 'Run-003'),
                                         os.path.join(config.data_dir, 'fits_online'), numChans=96, verbose=True)
        z_RK_HC_plot = z_RK_HC_plot.asdataframe()
        z_RK_HC_plot = z_RK_HC_plot[trimlength:]
        z_RK_HC_plot = z_RK_HC_plot[z_RK_HC_plot['TrialSuccess'] == True] # filter out unsuccessful trials
        z_RK_HC_plot = z_RK_HC_plot[z_RK_HC_plot['BlankTrial'] == False] # remove blank trial

        plotOnlineTraces(z_HC[90:100], posaxs[0], "Hand Control (HC)")
        plotOnlineTraces(z_RK_plot[52:62],posaxs[1],"ReFIT KF (RK)")
        plotOnlineTraces(z_NN[32:42],posaxs[2],"tcFNN (NN)")
        plotOnlineTraces(z_RN[102:112],posaxs[3],"Re-tcFNN (RN)")

    # plot settings
    # posaxs[1].legend()
    # posaxs[1].get_legend().legend_handles[0].set(edgecolor='k',facecolor=None)
    posaxs[3].set(xlabel='Time (sec)')
    for ax,color in zip(posaxs, [config.hcColor, config.kfColor, config.tcnColorB, config.tcnColor]):
        [i.set_linewidth(2) for i in ax.spines.values()]
        [i.set_edgecolor(color) for i in ax.spines.values()]

    # Per day, plot velocity distributions with broken axes, and calculate the KL divergence
    porder = (z_RK, z_NN, z_RN)
    labels = ('RK', 'NN', 'RN')
    palette = (config.kfColor, config.tcnColorB, config.tcnColor)
    kldivs = {'div':[], 'Decoder':[], 'counts':[], 'hc_counts':[]}
    for i, (zi, colors, decoder) in enumerate(zip(porder, palette, labels)):
        if not zi.empty:
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
    if genfig:
        online_metrics.calcVelocityDistribution(z_RK_plot, z_RK_HC_plot, plotResults=True, binrange=(-9, 9), numbins=100, binsize=50,
                                                    label=decoder, ax=dist_tops[0], color=config.kfColor)
        # bottom plot
        hist, binedges, hist_hc, binedges_hc, dist_fig = online_metrics.calcVelocityDistribution(z_RK_plot, z_RK_HC_plot,
                                                                                                    plotResults=True,
                                                                                                    binrange=(-9, 9),
                                                                                                    numbins=100,
                                                                                                    binsize=50,
                                                                                                    label=decoder,
                                                                                                    ax=dist_bots[0],
                                                                                                    color=config.kfColor)

        dist_tops[0].set(ylim=(0.5,3.5),xlim=(-4,4), xlabel=None, ylabel=None, title=None)
        dist_bots[0].set(ylim=(0, 0.2), xlim=(-4, 4), ylabel='Estimated Density', title=None,
                        yticks=[0,0.1,0.2])
        utils.online_metrics.drawBrokenAxes(dist_tops[0], dist_bots[0], d=0.015)

        dist_bots[0].get_legend().remove()
        dist_tops[0].get_legend().remove()

    # distribution plot settings
    dist_bots[0].set(xlabel=None, xticklabels=[])
    dist_bots[1].set(xlabel=None, xticklabels=[])
    dist_tops[0].set(title='RK Velocity Distribution')
    dist_tops[1].set(title='NN Velocity Distribution')
    dist_tops[2].set(title='RN Velocity Distribution')

    # Calculate performance metrics (time to first acquire, orbiting time)
    (tt, rt, ot) = online_metrics.calcTrialTimes(z_all, offBy2=offby2)
    clMetrics = pd.DataFrame(data={'TimeToTarget': rt, 'OrbitTime': ot})
    clMetrics['Decoder'] = z_all['Decoder']
    clMetrics.to_pickle(os.path.join(config.results_dir, 'fits_online', f'onlinefitmetrics_{mk_name}.pkl'))

    kldivs = pd.DataFrame(data=kldivs)
    kldivs.to_pickle(os.path.join(config.results_dir, 'fits_online', f'onlinefitdivs_{mk_name}.pkl'))
    return kldivs, metricaxs, dist_tops, onlinefitfig, clMetrics

def fits_online_partII_w(mk_name, kldivs, ax, results):
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
    
    kldivs.to_csv(os.path.join(config.results_dir, 'fits_online', f'kl_divs_w_counts_{mk_name}.csv'))
    kldivs = kldivs.pivot(index='date',columns='Decoder',values='div').rename(columns={'RK':'div_RK','NN':'div_NN', 'RN':'div_RN'})
    day_summaries = pd.concat(day_summaries, axis=0, keys=kldivs.index)

    # save results
    day_summaries.to_csv(os.path.join(config.results_dir, 'fits_online', f'online_fit_results_{mk_name}.csv'))
    kldivs.to_csv(os.path.join(config.results_dir, 'fits_online', f'kl_divs_{mk_name}.csv'))

    # do stats across days
    def runttest_ind(results, metric, althypo):
        results = results.where(results[metric] != 0)

        # get just the metric for rn, nn, and rk separately
        rn_metric = results.loc[results['Decoder'] == 'RN', :][metric]
        nn_metric = results.loc[results['Decoder'] == 'NN', :][metric]
        rk_metric = results.loc[results['Decoder'] == 'RK', :][metric]

        output = pd.DataFrame({'decoder':['RN','NN','RK'],
                               'mean':[rn_metric.mean(), nn_metric.mean(), rk_metric.mean()], 
                               'counts':[rn_metric.size, nn_metric.size, rk_metric.size],
                               'std':[rn_metric.std(), nn_metric.std(), rk_metric.std()]})
        
        return stats.ttest_ind(rn_metric, rk_metric, alternative=althypo), stats.ttest_ind(nn_metric, rk_metric, alternative=althypo), output

    tt_rn_test, tt_nn_test, tt_output = runttest_ind(results, 'TimeToTarget', 'less')
    ot_rn_test, ot_nn_test, ot_output = runttest_ind(results, 'OrbitTime', 'two-sided')

    # get total orbiting rate across days
    pop_sizes = results.groupby('Decoder').size()
    orbit_counts = results.groupby('Decoder').apply(lambda x: np.sum(x['Orbited']))
    orbit_props = orbit_counts / pop_sizes

    op_df = pd.DataFrame({'decoder':['RN','NN','RK'],
                          'mean':[orbit_props['RN'],orbit_props['NN'],orbit_props['RK']],
                          'counts':[pop_sizes['RN'],pop_sizes['NN'],pop_sizes['RK']],
                          'std':[0,0,0]})
    
    crossdayresults = pd.concat((tt_output, ot_output, op_df), keys=['tt','ot','or'])
    crossdayresults.to_csv(os.path.join(config.results_dir, 'fits_online', f'cross_day_metrics_{mk_name}.csv'))
    
    # difference of proportions test
    # null hypothesis, 0 difference in proportion
    pdiff_rn = orbit_props['RN'] - orbit_props['RK']
    phat_rn = (orbit_counts['RN'] + orbit_counts['RK']) / (pop_sizes['RN'] + pop_sizes['RK'])
    sediff_rn = np.sqrt(phat_rn * (1 - phat_rn) * (1 / pop_sizes['RN'] + 1 / pop_sizes['RK']))

    zscore_rn = pdiff_rn / sediff_rn
    pval_rn = 1 - stats.norm.cdf(np.abs(zscore_rn))

    pdiff_nn = orbit_props['NN'] - orbit_props['RK']
    phat_nn = (orbit_counts['NN'] + orbit_counts['RK']) / (pop_sizes['NN'] + pop_sizes['RK'])
    sediff_nn = np.sqrt(phat_nn * (1 - phat_nn) * (1 / pop_sizes['NN'] + 1 / pop_sizes['RK']))

    zscore_nn = pdiff_nn / sediff_nn
    pval_nn = 1 - stats.norm.cdf(np.abs(zscore_nn))

    #consolidate and save results
    statsout = pd.Series(data={'RN TT < RK TT':tt_rn_test.pvalue, 'RN OT != RK OT':ot_rn_test.pvalue, 'RN OR > RK OR':pval_rn,
                               'NN TT < RK TT':tt_nn_test.pvalue, 'NN OT != RK OT':ot_nn_test.pvalue, 'NN OR > RK OR':pval_nn})
    statsout.to_csv(os.path.join(config.results_dir,'fits_online',f'stats_{mk_name}.csv'))

    # PLOTTING
    # add average kl-div across days to distribution plots
    klmeans = kldivs.mean(axis=0)
    klstds = kldivs.std(axis=0)

    textpos = (0.5, 0.65)
    distax[0].text(textpos[0], textpos[1], f"Mean KL-div from HC:\n {klmeans.iloc[1]:.2f}",
                    transform=distax[0].transAxes, fontsize=mpl.rcParams['axes.labelsize'], ha='center')
    distax[1].text(textpos[0], textpos[1], f"Mean KL-div from HC:\n {klmeans.iloc[0]:.2f} +/- {klstds.iloc[0]:.2f}",
                    transform=distax[1].transAxes, fontsize=mpl.rcParams['axes.labelsize'], ha='center')
    distax[2].text(textpos[0], textpos[1], f"Mean KL-div from HC:\n {klmeans.iloc[2]:.2f} +/- {klstds.iloc[2]:.2f}",
                    transform=distax[2].transAxes, fontsize=mpl.rcParams['axes.labelsize'], ha='center')
    
    """
    # horizontal bar plots for time to target, orbiting rate, nonzero orbiting time averaged across days
    sns.barplot(results.where(results['TimeToTarget'] != 0), x='TimeToTarget', y='Decoder', errorbar='se', ax=metricax[0],
                palette=config.online_palette_w[[0, 3, 2, 1], :], order=['HC', 'RN', 'NN', 'RK'])
    sns.barplot(orbit_props.reset_index(), x=0, y='Decoder', ax=metricax[1],
                palette=config.online_palette_w[[0, 3, 2, 1], :], order=['HC', 'RN', 'NN', 'RK'])
    sns.barplot(results.where(results['OrbitTime'] != 0), x='OrbitTime', y='Decoder', errorbar='se', ax=metricax[2],
                palette=config.online_palette_w[[0, 3, 2, 1], :], order=['HC', 'RN', 'NN', 'RK'])

    metricax[0].set(title='Time-to-target', ylabel=None, xlabel='Time (s)', xlim=(0, 3000),
                xticks=[0, 1500, 3000], xticklabels=[0, 1.5, 3])
    metricax[1].set(title='Orbiting rate', ylabel=None, xlabel='Proportion',
            xticks=[0, .5, 1])
    metricax[2].set(title='Nonzero orbit time', ylabel=None, xlabel='Time (s)', xlim=(0, 2000),
                    xticks=[0, 1000, 2000], xticklabels=[0, 1, 2])
    """
    
    decoder_order = ['HC', 'RN', 'NN', 'RK']
    """
    # Horizontal bar plots not averaged across days
    # Bar plot for Time-to-Target
    sns.barplot(data=results[results['TimeToTarget'] != 0], x='date', y='TimeToTarget', hue='Decoder', hue_order=decoder_order, errorbar='se',
                ax=metricax[0], palette=config.online_palette_w[[0, 3, 2, 1], :], alpha=0.6,
                order=sorted(results['date'].unique(), reverse=True), dodge=True)
    sns.stripplot(results[results['TimeToTarget'] != 0], x='date', y='TimeToTarget', hue='Decoder', hue_order=decoder_order,
                  palette=config.online_palette_w[[0, 3, 2, 1], :], dodge=True, ax=metricax[0], alpha=0.7, zorder=1)
    
    pop_sizes_dates = results.groupby(['Decoder','date']).size()
    orbit_counts_dates = results.groupby(['Decoder','date']).apply(lambda x: np.sum(x['Orbited']))
    orbit_props_dates = orbit_counts_dates / pop_sizes_dates

    # Bar plot for Orbiting Rate
    sns.barplot(data=orbit_props_dates.reset_index(), x='date', y=0, hue='Decoder', hue_order=decoder_order,
                ax=metricax[1], palette=config.online_palette_w[[0, 3, 2, 1], :], alpha=0.6, dodge=True)
    
    # Bar plot for Nonzero Orbit Time
    sns.barplot(data=results[results['OrbitTime'] != 0], x='date', y='OrbitTime', hue='Decoder', hue_order=decoder_order, errorbar='se',
                ax=metricax[2], alpha=0.6, palette=config.online_palette_w[[0, 3, 2, 1], :], 
                order=sorted(results['date'].unique(), reverse=True), dodge=True)
    sns.stripplot(results[results['OrbitTime'] > 0], x='date', y='OrbitTime', hue='Decoder', hue_order=decoder_order,
                  palette=config.online_palette_w[[0, 3, 2, 1], :], dodge=True, ax=metricax[2], alpha=0.7, zorder=1)
    """

    # Vertical bar plots for individual days
    sns.barplot(data=results[results['TimeToTarget'] != 0], x='date', y='TimeToTarget', hue='Decoder', hue_order=decoder_order, errorbar='se',
                ax=metricax[0], palette=config.online_palette_w[[0, 2, 3, 1], :], alpha=0.6, linewidth=2, errcolor='black', 
                order=sorted(results['date'].unique(), reverse=True), dodge=True)
    sns.stripplot(results[results['TimeToTarget'] != 0], x='date', y='TimeToTarget', hue='Decoder', hue_order=decoder_order, size=4, 
                  palette=config.online_palette_w[[0, 2, 3, 1], :], dodge=True, ax=metricax[0], alpha=0.7, zorder=1)
    
    pop_sizes_dates = results.groupby(['Decoder','date']).size()
    orbit_counts_dates = results.groupby(['Decoder','date']).apply(lambda x: np.sum(x['Orbited']))
    orbit_props_dates = orbit_counts_dates / pop_sizes_dates

    # Bar plot for Orbiting Rate
    sns.barplot(data=orbit_props_dates.reset_index(), x='date', y=0, hue='Decoder', hue_order=decoder_order,
                ax=metricax[1], palette=config.online_palette_w[[0, 2, 3, 1], :], alpha=1, dodge=True)
    
    sns.barplot(data=results[results['OrbitTime'] > 0], x='date', y='OrbitTime', hue='Decoder', hue_order=decoder_order, errorbar='se',
                ax=metricax[2], palette=config.online_palette_w[[0, 2, 3, 1], :], alpha=0.6, linewidth=2, errcolor='black',
                order=sorted(results['date'].unique(), reverse=True), dodge=True)
    sns.stripplot(results[results['OrbitTime'] > 0], x='date', y='OrbitTime', hue='Decoder', hue_order=decoder_order,
                  palette=config.online_palette_w[[0, 2, 3, 1], :], dodge=True, ax=metricax[2], alpha=0.7, size=4, zorder=0)
                  
    """
    # Code for horizontal lines as opposed to bar plots
    dates = sorted(results['date'].unique(), reverse=True)
    ax = metricax[2]
    # Loop through each category in the 'Decoder' column
    palette_plotting = config.online_palette_w[[0, 3, 2, 1], :]
    for i, decoder in enumerate(decoder_order):
        subset = results[(results['Decoder'] == decoder) & (results['OrbitTime'] != 0)]
        
        # Compute means and standard errors (assuming results contains these columns)
        means = subset.groupby('date')['OrbitTime'].mean()
        errors = subset.groupby('date')['OrbitTime'].sem()
        
        # Get x positions corresponding to `dodge=True` behavior
        x_positions = [dates.index(d) + (i - (len(decoder_order) - 1) / 2) * 0.2 for d in means.index]
        
        # Draw horizontal lines at the mean values
        ax.hlines(y=means, xmin=[x - 0.1 for x in x_positions], xmax=[x + 0.1 for x in x_positions], colors=palette_plotting[i, :], lw=3, alpha=1, zorder=1)

        # Optionally, add error bars
        ax.errorbar(x_positions, means, yerr=errors, fmt='none', color='black', capsize=0, elinewidth=2, zorder=2)
    """
    # Formatting for Time-to-Target plot
    metricax[0].set(title='Time-to-Target', xlabel='Day',  ylabel='Time (s)',xticklabels=['1','2','3'],
                    ylim=(0, 8000), yticks=[0, 4000, 8000], yticklabels=[0, 4, 8])
    metricax[0].legend_.remove()
    
    # Formatting for Orbiting Rate plot
    metricax[1].set(title='Orbiting Rate', ylabel='Proportion', xlabel='Day', xticklabels=['1','2','3'],
                    yticks=[0, 0.5, 1])
    metricax[1].legend_.remove()
    
    # Formatting for Nonzero Orbit Time plot
    metricax[2].set(title='Nonzero Orbit Time', xlabel='Day', ylabel='Time (s)', xticklabels=['1','2','3'],
                    ylim=(0, 8000), yticks=[0, 4000, 8000], yticklabels=[0, 4, 8])
    metricax[2].legend_.remove()
    