import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from utils import ztools
import config

def calcTrialTimes(z, fingers=None, offBy2 = False, singleout=False):
    '''
    Given a zStruct and depending on the number of outputs, this function returns either the total trial time or the
    separated target acquisition and orbiting times for each trial in z. Based on Sam Nason's MATLAB function, but with
    a minor difference, acquire time (timetarghit) when the fingers start in the target is 0 as opposed to 1 in the
    MATLAB function.
    Inputs:
        - z:                    The zStruct containing the trials we want the trial times of
        - fings (optional):     1d ndarray of fingers (0-4) to use. defaults to using only fingers used in first trial (that had targets)
        - offBy2 (optional):    A boolean to indicate whether the dataset was collected from rig code that had a hold
                                time 2ms longer than what was set in TargetHoldTime (True). Default (False) doesn't
                                include the 2 ms.
        - singleout(optional):  Default False. If true, returns a single ndarray with trial times minus hold time rather
                                than 2 ndarrays of time to first acquire and orbiting time.
    Outputs:
        - (zTimes, timetarghit, orbt):  If singleout == False, returns a tuple with 3 nx1 ndarrays containing the
                                trial time, time to first acquire and orbiting times of all n trials in order.
        - zTimes:               If singleout == True, returns a single nx1 ndarray contianing the total trial time
                                (hold time not included), basically acqt + orbt.
    '''
    if fingers is None:
        fingers = np.argwhere(z['TargetPos'].iloc[0] != -1)
        # Note: for online trials, we shift fingers by 5 to use the SBP decode

    cl = z['ClosedLoop'].astype(bool).to_numpy()
    try:
        decFeat = z['DecodeFeature'].astype(bool).to_numpy()
    except:
        decFeat = np.zeros((len(z),), dtype=bool)

    decFeat[~cl] = False

    # tells you if a run was hand contorl of closed loop (and which field to look at accordingly)
    fbfield = np.empty(len(z), dtype=object) #field containing finger feedback
    fbfield[cl] = 'Decode'
    fbfield[~cl] = 'FingerAnglesTIMRL'

    # get parameters per-trial
    zTimes = z['ExperimentTime'].str.len().to_numpy()
    widths = 0.0375 * (1+z['TargetScaling'].to_numpy()/100)
    timetarghit = np.zeros((len(z),))

    # iterates through each trial, getting the position of the target edges, and finding when all targets were first
    # hit simultaneously (if at all)
    for i in np.arange(len(z)):
        position = z[fbfield[i]].iloc[i][:, fingers + len(z['MoveMask'].iloc[i]) * decFeat[i].astype(int)]
        edge1 = position >= z['TargetPos'].iloc[i][fingers] - widths[i]
        edge2 = position <= z['TargetPos'].iloc[i][fingers] + widths[i]

        targhit = np.logical_and(edge1, edge2)
        timesHit = np.all(targhit > 0, axis=1).nonzero()[0]
        if np.size(timesHit) == 0:
            timetarghit[i] = -1
        else:
            timetarghit[i] = timesHit[0]

    # subtracts the time to reach and hold time from the trials to find the orbiting time. Also, accounts for offby2
    # error in some older models.
    oTime = -1 * np.ones_like(timetarghit)
    oTime[timetarghit != -1] = zTimes[timetarghit != -1] - z['TargetHoldTime'][timetarghit != -1] - 2*int(offBy2)  - \
                               timetarghit[timetarghit != -1]

    timetarghit[timetarghit == -1] = zTimes[timetarghit == -1]

    # note that when oTime = -1, the targets were never hit (timetarghit will be reported as the trial timeout).
    # if timetarghit is 0, the fingers started in the target. if otime is 0, there was no orbiting.
    if singleout:
        return zTimes
    else:
        return (zTimes, timetarghit, oTime)
    
def plotOnlinePositions(z, ax):
    fingers = np.argwhere(z['TargetPos'].iloc[0] != -1).squeeze()
    finger_label = ['Index', 'MRP']
    edgeColor = np.asarray([[.3,0,0],[0,.3,0]])

    #get the feedback that was shown on screen and flip to have extension at 100
    fb = 100 * (1-getVisFeedback(z, fingers=fingers))
    #get the target positions for each trial
    targs = np.stack(z['TargetPos'].to_numpy())[:, fingers] * 100
    #get widths of each target (from center)
    targwidths = .0375 * 100 * (1+z['TargetScaling'].to_numpy()/100)[:,np.newaxis]
    #get starting y position for patches.Rectangle
    targy = 100 - targs - targwidths
    #get times for each trial (as time edges of the box)
    etimes = z['ExperimentTime'].iloc[:].apply(lambda x: len(x)).to_numpy()
    #get starts for each trial (starting x position for patches.Rectangle)
    estarts = np.concatenate((np.zeros(1), np.cumsum(z['ExperimentTime'].iloc[0:-1].apply(lambda x: len(x)).to_numpy())))

    #plot fb and plot boxes
    rect_args = {'lw': 2, 'ls': '', 'alpha': 0.6}
    tr_args = {'lw': 2, 'ls':'-'}
    if ax is None:
        ax = plt.axes()
    for j in np.arange(len(fingers)):
        for i in np.arange(len(targs)):
            if (i == 0) and (j == 0):
                label = 'Target'
            else:
                label = None
            rect = mpl.patches.Rectangle((estarts[i]/1000,targy[i,j]), etimes[i]/1000,targwidths[i,0]*2,
                                         fc=config.onlineTracesPalette[j,:], label= label, **rect_args)
            ax.add_patch(rect)

        ax.plot(np.arange(0,len(fb))/1000, fb[:,j], color=config.onlineTracesPalette[j,:],
                label=finger_label[j], **tr_args)
    ax.set(xlim=(0, len(fb)/1000), ylim=(-10,110),xlabel='Time (s)', ylabel='Extension (%)')
    ax.set_ylim(-10,110)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Extension (%)')
    return ax


def drawBrokenAxes(ax1, ax2, d):
    ax1.spines['bottom'].set_visible(False)
    ax1.tick_params(labeltop=False,bottom=False,labelbottom=False)
    ax2.spines['top'].set_visible(False)
    ax2.xaxis.tick_bottom()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)  # bot-left diagonal
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # bot-right diagonal
    ax1.axhline(y=ax1.get_ylim()[0], linestyle='--', color='k')

    kwargs.update(transform=ax2.transAxes)  # switch to the top axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # top-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # top-right diagonal
    ax2.axhline(y=ax2.get_ylim()[1], linestyle='--', color='k')


def getVisFeedback(z, fingers=None, fillGaps=False):
    '''
    Takes in input z (dataframe) and returns the finger positions as given to the subject as visual feedback.
    Translated from Sam Nason's Utility Code Folder

    Inputs:
        - z (pd.dataframe):
            the z struct containing the visual feedback
        - fings (ndarray, optional):
            an array of fingers to return. default: whatever fingers have targets presented on first trial
        - fillGaps (bool, optional):
            interpolate the gaps in position in between trials. default: False
    '''

    #get CL/OL info to know which field contains the presented finger positions
    if fingers is None:
        fingers = np.argwhere(z['TargetPos'].iloc[0] != -1)

    cl = z['ClosedLoop'].astype(bool).to_numpy()
    try:
        decFeat = z['DecodeFeature'].astype(bool).to_numpy()
    except:
        decFeat = np.zeros((len(z),), dtype=bool)

    decFeat[~cl] = False

    fbfield = np.empty(len(z), dtype=object) #field containing finger feedback
    fbfield[cl] = 'Decode'
    fbfield[~cl] = 'FingerAnglesTIMRL'

    if fillGaps:
        timediffs = np.zeros((len(z),1))
        # this finds the quantity of samples missing between each trial, with a 0 at the end to prevent adding anything
        timediffs[:-1,0] = z['ExperimentTime'].iloc[1:].apply(lambda x: x[0]).to_numpy() - z['ExperimentTime'].iloc[0:-1].apply(lambda x: x[-1]).to_numpy() - 1

        nextpos = np.zeros((len(z), fingers.shape[0]))
        for i in np.arange(len(z)-1):
            #this contains the next position to interpolate
            nextpos[i, :] = z[fbfield[i+1]].iloc[i+1][0, fingers + len(z['MoveMask'].iloc[i+1]) * decFeat[i+1].astype(int)].flatten()
    else: #if we arent filling gaps, dont append anything
        timediffs = np.zeros((len(z),1))
        nextpos = np.zeros((len(z),len(fingers)))

    #sam had some fancy arrayfun ways of doing this in matlab, for now will not be doing that
    trials = []
    for i in np.arange(len(z)):
        #get visfeedback for each trial
        # trialfb = z[fbfield[i]].iloc[i][:,fingers+ len(z['MoveMask'].iloc[i]) * decFeat[i].astype(int)].squeeze()         # was giving an error
        trialfb = z[fbfield[i]].iloc[i][:, fingers + len(z['MoveMask'].iloc[i]) * decFeat[i].astype(int)].reshape(-1, len(fingers))

        # append the interpolated samples (if they are desired)
        interp = np.linspace(trialfb[-1,:], nextpos[i,:], timediffs[i,0].astype(int))

        full = np.concatenate((trialfb, interp), axis=0)

        #add to list
        trials.append(full)
    #stitch it all together
    fb = np.concatenate(trials, axis=0)
    # sams code also had a few lines which returned a variable 'changed' which amy indicate when things changed between
    # CL/OL. Might be TODO eventually
    return fb


def calcVelocityDistribution(z, z_hc=None, plotResults=True, binrange=(-1,1),numbins=100, binsize=50, label='Decoder',
                              color=(196/255, 64/255, 64/255), ax=None):
    '''
    Calculates and returns the bin counts for velocity distribution given a z struct. if provided, can also provide counts
    for z_hc, a z struct with hand control movements in it. histogram bincounts are returned as estimated densities
    of a pdf.

    inputs:
    z (pd.DataFrame):
        zstruct containing trial information to extract and analyze
    z_hc (pd.DataFrame, optional):
        zstruct containing hand control information to extract and analyze
    plotResults (bool, optional):
        whether or not to plot the results or just return the bincounts
    binrange (tuple, optional):
        range of velocities to calc hist over, in AU/sec. if not included, defaults to [-1,1]
    numbins (int, optional):
        number of bins to include. default 100.
    binsize (int, optional):
        perhaps confusingly named, this refers to the binsize of the zstruct to be used when getZfeats is run.
        default 50.
    decoder (string, optional):
        what label to use for the decoder. default 'Decoder'
    color (tuple, optional):
        len(3) tuple for the color, values should be [0,1]
    ax (plt.Axes, optional):
        axes to plot on. if not provided, creates new figure
    outputs:
    hist (np.ndarray):
        value of each bin
    binedges (np.ndarray)
        return the bin edges (len(bincount) + 1)
    hist_hc (np.ndarray, optional):
        values of each bin in the hand control histogram, optional
    bineadges_hc (np.ndarray, optional):
        bin edges of the hand control histogram
    dist_fig (plt.Figure, optional):
        if plotResults is true and no axes are provided, returns the figure
    '''

    #take only closed loop trials where decode is on
    z = z[z['ClosedLoop'] == True]

    feats = ztools.getZFeats(z, binsize, featList=['Decode'])

    #get velocities
    v_idx = np.argwhere(z['TargetPos'].iloc[0] != -1) + len(z['TargetPos'].iloc[0])
    #here, first half is tcfr, second is sbp positions - need to take diff.
    v = np.diff(feats['Decode'][:,v_idx],axis=0)

    #scale to AU/sec, not AU/bin
    sec = 1000
    v = v/binsize * sec
    # pdb.set_trace()
    #if z_hc is included, get the same
    if z_hc is not None:
        z_hc = z_hc[z_hc['ClosedLoop'] == False]
        feats = ztools.getZFeats(z_hc, binsize, featList=['FingerAnglesTIMRL'])
        v_idx = np.argwhere(z['TargetPos'].iloc[0] != -1) + len(z['TargetPos'].iloc[0])
        v_hc = feats['FingerAnglesTIMRL'][:, v_idx]
        v_hc = v_hc / binsize * sec

    # here things begin to differ base don plotting or not. plt.hist and np.histogram work the same way, but plt
    # will actually plot a figure
    if plotResults:
        if ax is None:
            dist_fig = plt.figure(figsize=(9,5.5))
            ax = dist_fig.add_axes([.1, .1, .8, .8])
        else:
            dist_fig = None
        if z_hc is not None:
            hist_hc, binedges_hc, _ = ax.hist(v_hc.flatten(), bins=numbins, range=binrange, density=True, label='Hand Control',
                                     color=(0.5, 0.5, 0.5), alpha=0.8)
        else:
            hist_hc = None
            binedges_hc = None

        hist, binedges, _ = ax.hist(v.flatten(), bins=numbins, range=binrange, density=True, histtype='step',
                                 label=label,color=color, linewidth = 4)
        ax.legend()
        ax.set_title('Velocity Distribution plot')
        ax.set_ylabel('Estimated Density')
        ax.set_xlabel('Velocity, (AU/sec)')

    else:
        hist, binedges = np.histogram(v.flatten(), bins=numbins, range=binrange, density=True)
        if z_hc is not None:
            hist_hc, binedges_hc = np.histogram(v_hc.flatten(), bins=numbins, range=binrange, density=True)
        else:
            hist_hc = None
            binedges_hc = None
        dist_fig = None

    return hist, binedges, hist_hc, binedges_hc, dist_fig


def calcBitRate(z, fingers_0idx=None, offby2=False):
    '''
    Given a zStruct and depending on the number of outputs, this function returns the trial bitrates. Based on
    Sam Nason's MATLAB function, and very similar to calcTrialTimes.

    Inputs:
        - z:                    The zStruct (dataframe) containing all trial info
        - fingers_0idx (optional):  [1,3]. A list of fingers (0-4) to use. defaults to using only fingers used in first
                                    trial (that had targets).
        - offBy2 (optional):    A boolean to indicate whether the dataset was collected from rig code that had a hold
                                time 2ms longer than what was set in TargetHoldTime (True). Default (False) doesn't
                                include the 2 ms.
    Outputs:
        - bitrates:             num_trials x 1 ndarray containing bitrates
    '''
    if fingers_0idx is None:
        fingers_0idx = np.argwhere(z['TargetPos'].iloc[0] != -1).reshape((-1,))

    # figure out which field to use (based on if it was online or offline)
    cl = z['ClosedLoop'].astype(bool).to_numpy()
    try:
        decFeat = z['DecodeFeature'].astype(bool).to_numpy()
    except:
        decFeat = np.zeros((len(z),), dtype=bool)
    decFeat[~cl] = False

    fbfield = np.empty(len(z), dtype=object)  # the field containing finger feedback
    fbfield[cl] = 'Decode'
    fbfield[~cl] = 'FingerAnglesTIMRL'

    # get trial info (start pos, target, success, targ width, trial time)
    succ = z['TrialSuccess'].astype(bool).to_numpy()
    targs = np.stack(z['TargetPos'].to_numpy())[:, fingers_0idx].reshape((-1, len(fingers_0idx)))
    starts = np.zeros_like(targs)

    for i in np.arange(len(z)):
        starts[i, :] = z[fbfield[i]].iloc[i][0, fingers_0idx + len(z['MoveMask'].iloc[i]) * decFeat[i].astype(int)]
    zTimes = z['ExperimentTime'].str.len().to_numpy() - z['TargetHoldTime'].to_numpy() - offby2 * 2
    zTimes_sec = zTimes / 1000
    zTimes_sec[zTimes_sec < 1e-6] = 0.001                       # TODO why are some trials less than hold time??
    widths = 0.0375 * (1 + z['TargetScaling'].to_numpy() / 100)

    # now calc bitrate (as the sum of bitrates for each finger)
    dists = np.abs(targs - starts)
    widths_each_fing = np.tile(widths.reshape((-1, 1)), (1, dists.shape[1]))
    dists -= widths_each_fing
    dists[dists < 0] = 0
    difficulties = np.log2(1 + dists / (2 * widths_each_fing))
    bitrates = np.sum(difficulties, axis=1) / zTimes_sec
    bitrates[~succ] = 0

    return bitrates