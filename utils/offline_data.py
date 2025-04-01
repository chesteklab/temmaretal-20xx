# Filename:
# Title:
# Date:
# Version:
# Description:
import numpy as np

import config


def adjust_feats(X, Y, lag=0, hist=0, fillb4=None, out2d=False):
    '''
    This function takes in neural data X and behavior Y and returns two "adjusted" neural data and behavior matrices
    based on the optional params. Specifically the amount of lag between neural data and behavior can be set in units
    and the number of historical bins of neural data can be set. (BASED ON CODE BY SAM NASON)
    Inputs:
        - X (ndarray):
            The neural data, which should be [t, neu] in size, where t is the numebr of smaples and neu is the number
            of neurons.
        - y (ndarray):
            The behavioral data, which should be [t, dim] in size, where t is the number of samples and dim is the
            number of states.
        - lag (int, optional):
            Defaults to 0. The number of bins to lag the neural data relative to the behavioral data. For example,
            adjust_feats(X,Y, lag=1) will return X[0:-1] for adjX and Y[1:] for adjY.
        - hist (int, optional):
            Default 0. The number of bins to append to each sample of neural data from the previous 'hist' bins.
        - fillb4 (ndarray or scalar, optional):
            Default None, disabled. This fills previous neural data with values before the experiment began. A single
            scalar wil fill all previous neural data with that value. Otherwise, a [1,neu] ndarray equal to the first
            dimension of X (# of neurons) should represent the value to fill for each channel.
        - out2d (bool, optional):
            if history is added, will return the adjusted matrices either in 2d or 3d form (2d has the history appended
            as extra columns, 3d has history as a third dimension. For example, out2d true returns a sample as:
            [1, neu*hist+1] whereas out2d false returns: [1, neu, hist+1]. Default False
    Outputs:
        - adjX (ndarray):
            The adjusted neural data
        - adjY (ndarray):
            The adjusted behavioral data
    '''
    nNeu = X.shape[1]
    if fillb4 is not None:
        if isinstance(fillb4, np.ndarray):
            Xadd = np.tile(fillb4, hist)
            Yadd = np.zeros((hist, Y.shape[1]))
        else:
            Xadd = np.ones((hist, nNeu))*fillb4
            Yadd = np.zeros((hist, Y.shape[1]))
        X = np.concatenate((Xadd, X))
        Y = np.concatenate((Yadd, Y))

    #reshape data to include historical bins
    adjX = np.zeros((X.shape[0]-hist, nNeu, hist+1))
    for h in range(hist+1):
        adjX[:,:,h] = X[h:X.shape[0]-hist+h,:]
    adjY = Y[hist:,:]

    if lag != 0:
        adjX = adjX[0:-lag,:,:]
        adjY = adjY[lag:,:]

    if out2d:
        #NOTE: History will be succesive to each column (ie with history 5, columns 0-5 will be channel 1, 6-10
        # channel 2, etc..
        adjX = adjX.reshape(adjX.shape[0],-1)

    return adjX, adjY

def data_cleanup(data):
    vel = data['FingerAnglesTIMRL'][:, [6, 8]]
    neu = data['NeuralFeature']
    # add time history
    neu, vel = adjust_feats(neu, vel, hist=2, out2d=True)
    neu2D = np.concatenate((neu, np.ones((len(neu), 1))), axis=1) # add a column of ones for RR
    neu3D = neu.reshape(len(neu), config.numChans, -1)
    datadict = {'vel':vel,'neu2D':neu2D,'neu3D':neu3D}
    return datadict

def truncate_to_shortest(data, length):
    # this probably could have been done in a better way
    for context in data.keys():
        for key in data[context].keys():
            data[context][key] = data[context][key][:length,:]
    return data

def split_offline_data(data, numFolds):
    '''

    :param data: dict with keys 'vel','neu','neu3D' produced by datacleanup()
    :param numFolds: number of folds to split into
    :return:
    '''
    n = len(data['vel'])
    rng = np.random.default_rng()
    # get all the indices and shuffle them around.
    shuffled_idx = np.arange(n)
    rng.shuffle(shuffled_idx)

    #split them into numFolds arrays roughly equal in size.
    shuffled_idx = np.array(np.array_split(shuffled_idx, numFolds),dtype=object)
    # now we have indices that are shuffled and split into groups

    inIDXList = []
    outIDXList = []

    newdata = {'vel': [], 'neu2D': [], 'neu3D': []}
    for k in np.arange(numFolds): #for each fold

        # get the indices of the leave in group together
        inidx = np.concatenate(shuffled_idx[np.arange(numFolds) != k])

        # get the indices of the leave out group
        outidx = shuffled_idx[k]

        #save those so we have them if needed
        inIDXList.append(inidx)
        outIDXList.append(outidx)

        # go through the keys
        for feat in data.keys():
            fulldat = data[feat]
            #for each context and each feat, add the leave in group for this fold.
            newdata[feat].append(fulldat[inidx.astype(int), ...])

    return newdata, inIDXList, outIDXList

def split_context_data(data, numFolds):
    '''
    :param data: dict with contexts as keys - each key contains a dict with keys 'vel','neu','neu3d' produced by
    datacleanup().
    :param numFolds: number of folds to split into
    :return:
    '''
    #given we truncated all the data, it should be all the same length.
    n = len(data[config.context_order[0]]['vel'])
    rng = np.random.default_rng()
    # get all the indices and shuffle them around.
    shuffled_idx = np.arange(n)
    rng.shuffle(shuffled_idx)

    #split them into numFolds arrays roughly equal in size.
    shuffled_idx = np.array(np.array_split(shuffled_idx, numFolds),dtype=object)
    # now we have indices that are shuffled and split into groups

    #These lists will be for us to keep track of which indices are going where if we need that.
    inIDXList = []
    outIDXList = []
    mixIDXList = []

    #this will be a hierarchical dict organized like so: newdata[context][feature][fold]
    newdata = {}
    newdata['Mixed'] = {'vel': [], 'neu2D': [], 'neu3D': []}  # create the mixed context
    newdata['Mixed_Full'] = {'vel': [], 'neu2D': [], 'neu3D': []}  # create the full mixed context
    for k in np.arange(numFolds): # do this for each fold
        #get the indices of the leave in group and the leave out fold
        #we'll save what indices were used for record-keeping
        inidx = np.concatenate(shuffled_idx[np.arange(numFolds) != k])
        outidx = shuffled_idx[k]
        inIDXList.append(inidx)
        outIDXList.append(outidx)

        # setting up the short mixed datasets
        # take a roughly equal sized piece from each context and concatenate into the short mixed dataset
        # we'll add these indices to a list for record-keeping
        mixed_splits = np.array_split(inidx, len(data.keys()))
        mixIDXList.append(mixed_splits)
        shortmixedfold = {}

        #setting up the long mixed dataset for this fold
        longmixedfold = {}

        # now, go through each context and a) add its leave in group for this fold to newdata, and b) create the mixed
        # datasets
        for i,context in enumerate(data.keys()):
            if k == 0:
                # if we're on the first fold, we haven't made a feature subdict for this context yet.
                newdata[context] = {}

            # go through neural and behavioral data and do the same splits
            for feat in data[context].keys():
                fulldat = data[context][feat]

                if k == 0:
                    # if we're on the first fold, we haven't made a list for this feat yet.
                    newdata[context][feat] = []

                if i == 0:
                    # if we're on the first context, the temporary mixed dicts need a list for each feature
                    shortmixedfold[feat] = []
                    longmixedfold[feat] = []

                # add the leave in group to the list for this feat and context.
                newdata[context][feat].append(fulldat[inidx,...])

                # now, append the whole thing to the full mixed dataset for this fold, and the portion from mixed splits
                # for the short mixed fold.
                longmixedfold[feat].append(fulldat[inidx,...])
                shortmixedfold[feat].append(fulldat[mixed_splits[i],...])

        # after running through the context, the long and short mixed fold dicts will contain lists with all the data
        # they need which can be concatenated and places in the newdata dict.
        for feat in newdata['Mixed'].keys():
            newdata['Mixed'][feat].append(np.concatenate(shortmixedfold[feat]))
            newdata['Mixed_Full'][feat].append(np.concatenate(longmixedfold[feat]))

    #at the end of it all, we should have a newdata dict of dicts of lists, organized like:
    # newdata[context][feature][fold], which now has an additional mixed set for each fold.
    # we should also have 3 lists of length numfolds. the first is an array per fold with the indices for the
    # leave in groups, the second is the same but for the leave out groups, and the third is a list of lists per fold
    # of length m (where m is the number of contexts) which contain arrays indicating the indices for each context
    # that should go in the mixed set for that fold.

    # just so it's on the record I know this is probably a horribly inefficient way to do this, but 'it is what it is'
    return newdata, inIDXList, outIDXList, mixIDXList