import numpy as np
import pdb
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils import offline_metrics, nn_decoders
import config
import yaml

def rrTrain(X, y, lbda=0.0):
    '''
    Translated from Sam Nason's Utility folder on SVN. Trains a ridge regression and puts the model in p. It assumes
    that the first dimension of X and y are time, the second dimension of X is neuron, and the second dimension of y is
    in the order of [pos, vel, acc, 1]. Here, pos, vel, and acc should be [n, m] matrices where n is the number of
    samples and m is the number of movement dimensions.
    Inputs:
        - X (ndarray):
                Input neural data, which is size [n, neu], where n is the number of samples and neu is the
                number of neurons. X should already include a column of ones, if desired.
        - y (ndarray):
                Input behavior which is size [n, d], where n is the numebr of samples and m is the number of
                state dimensions.
        - lbda (scalar, optional):
                Customized value for lambda, defaults to 0.

    Outputs:
        - p (dict):
                p['Theta'] contains the trained theta matrix
                p['Lambda'] contains the value used for lambda
    '''
    p = {}

    temp = np.linalg.lstsq(np.matmul(X.T, X) + lbda * np.eye(X.shape[1]), np.matmul(X.T, y))
    p['Theta'] = temp[0]
    p['Lambda'] = lbda
    return p

def rrPredict(X, p):
    '''
    Translated from Sam Nason's Utility folder on SVN. This function makes ridge regression predictions from the neural
    data in X based on the params in p.
    Inputs:
        - X (ndarray):
                Input neural data, which is size [n, neu], wherre n is the number of samples and neu is the number of
                neurons. X should already include a column of ones, if desired.
        - p (dict):
                Input RR parameters. Should be a dict with at least the following field: theta. Run rrTrain first on
                training data to get these parameters made in the proper format
    Outputs:
        - yhat (ndarray):
                A size [n, d] matrix of predctions, where n is the number of samples and d is the number of state
                dimensions (dim 1 of p theta).
    '''
    yhat = None
    if 'Theta' in p:
        yhat = np.matmul(X, p['Theta'])
    else:
        ValueError('P does not contain a field Theta')
    return yhat

def dsTrain(X, y, vel_idx=None, post_prob=0.5, alpha=0.01, lbda=4, initK=None, initMoveProb=None):
    '''
    Trains a dual-state decoder based on work by Sachs et al. 2016. Separates movements into movement and posture states
    by sorting velocities, trains separate RR models on each, and trains the LDA weight matrix used for later prediction
    Inputs:
        - X (ndarray):
                Input neural data, which is size [n, neu], wherre n is the number of samples and neu is the number of
                neurons. X should already include a column of ones (REQUIRED).
        - y (ndarray):
                Input behavior which is size [n, d], where n is the numebr of samples and m is the number of state
                dimensions.
        - vel_idx (int list, optional):
                Indices of velocity states. Default None, assumes all dimensions are velocity
        - post_prop (float, optional):
                Optionally set the desires proportion of separation for movement vs posture states. Otherwise, defaults
                to 50/50 split.
        - alpha (float, optional):
                Update rate for moving threshold
        - lbda (float, optional):
                'Steepness' of logistic function, default 4
        - initK (float, optional):
                What initial k to start with in decoding. Default None, in which it's estimated from training data
        - initMoveProb (float, optional):
                Initial probability of movement to use in the first timestep.
                Default None, which which just assume standard mean movement probability.
    Outputs:
        - p (dict):
                p['move_theta'] contains the movement model. p['post_theta'] contains the posture model. p['W'] contains
                the LDA weights. p['postprob'], p['alpha'], p['lbda'], p['initK'], p['initMoveProb'] contain the
                respective parameters.
    '''

    move_prob = 1 - post_prob
    if vel_idx is None:
        vel_idx = np.arange(y.shape[1]).astype(int)
    if initMoveProb is None:
        initMoveProb = move_prob

    # separate states
    v_mag = np.sqrt(np.sum(y[:, vel_idx] ** 2, axis=1))
    mag_idx = np.argsort(v_mag)

    split_boundary = np.round(mag_idx.shape[0] * post_prob).astype(int)
    post_idx = mag_idx[:split_boundary]
    move_idx = mag_idx[split_boundary:]

    ypost = y[post_idx, :]
    ymove = y[move_idx, :]
    Xpost = X[post_idx, :]
    Xmove = X[move_idx, :]

    # Train Models for Movement and Posture
    lbda_best = 0.001

    post_model = rrTrain(Xpost, ypost, lbda=lbda_best)
    move_model = rrTrain(Xmove, ymove, lbda=lbda_best)

    # Train LDA
    S = np.cov(X[:, 0:-1].T)
    meandiff = np.mean(Xmove[:, :-1], axis=0) - np.mean(Xpost[:, :-1], axis=0)
    W = np.linalg.lstsq(S, meandiff)[0].T

    if initK is None:
        initK = np.dot(W, move_prob * (np.mean(Xmove[:, -1], axis=0) + np.mean(Xpost[:, :-1], axis=0)).T)

    p = {'move_model': move_model, 'post_model': post_model, 'W': W, 'alpha': alpha, 'lbda': lbda, 'initK': initK,
         'initMoveProb': initMoveProb, 'postprob': post_prob}
    return p

def dsPredict(X, p):
    '''
    Generates dual-state decoder predictions using pretrained matrices.
    Inputs:
        - X (ndarray):
                Input neural data, which is size [n, neu], wherre n is the number of samples and neu is the number of
                neurons. X should already include a column of ones, if desired.
        - p (dict):
                p['move_theta'] contains the movement model. p['post_theta'] contains the posture model. p['W'] contains
                the LDA weights. p['postprob'], p['alpha'], p['lbda'], p['initK'], p['initMoveProb'] contain the
                respective parameters.
    Outputs:
        - yhat(ndarray):
                A size [n,d] matrix of predictions, where n is the number of samples and d is the number of state
                dimensions (dim 1 of move and post models).
    '''
    alpha = p['alpha']
    kprev = p['initK']
    avgmoveprob = p['initMoveProb']
    lbda = p['lbda']
    meanWindowSize = 200
    postprob = p['postprob']
    moveprob = 1 - postprob
    move_model = p['move_model']
    post_model = p['post_model']
    W = p['W']

    k = kprev
    pmh = np.zeros((X.shape[0], 1))

    for i in np.arange(X.shape[0]):
        k = kprev + alpha * (avgmoveprob - moveprob)
        pm = 1 / (1 + np.exp(-lbda * (np.dot(W, X[i, :-1]) - k)))
        pmh[i] = pm
        avgmoveprob = np.mean(pmh[np.maximum(0, i + 1 - meanWindowSize):i + 1])
        kprev = k
    # pdb.set_trace()
    movepred = rrPredict(X, move_model)
    postpred = rrPredict(X, post_model)

    yhat = pmh * movepred + (1 - pmh) * postpred
    return yhat, pmh

# initialize an NN model based on architecture and config
def init_model(model_class, model_type, in_size, out_size):
    with open(config.architectures_path) as f:
        model_params = yaml.load(f, Loader=yaml.FullLoader)[model_type]
    
    if model_type == 'TCN':
        model = model_class(in_size,
                            model_params['layer_size'],
                            model_params['conv_size'],
                            model_params['conv_size_out'],
                            out_size).to(config.device)
    elif model_type == 'LSTM_xnorm_ynorm':
        model = model_class(in_size,
                            model_params['hidden_size'],
                            out_size,
                            model_params['num_layers'],
                            rnn_type=model_params['rnn_type'],
                            drop_prob = model_params['drop_prob'],
                            dropout_input=0).to(config.device)
    return model

# initialize an optimizer and potentially a learning rate scheduler
def init_opt(model, model_type):
    with open(config.training_params_path) as f:
        training_params = yaml.load(f, Loader=yaml.FullLoader)[model_type]

    opt = torch.optim.Adam(model.parameters(), 
                           lr=float(training_params['learning_rate']),
                           weight_decay=float(training_params['weight_decay']))
    
    if training_params['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5,
                                                               patience=training_params['scheduler_patience'])
    else:
        scheduler = None
    
    return opt, scheduler
    
    # TODO deal with training parameters

def train_nn(train_neu, train_vel, model_class, model_type, normalize=True):
    #turn training data into tensors and move to gpu if needed
    neu = torch.from_numpy(train_neu).to(config.device, config.dtype)
    vel = torch.from_numpy(train_vel).to(config.device, config.dtype)

    if normalize:
        # normalize neural data (new by JC)
        # neu_train has shape (num_trials, num_channels, num_time_hist)
        neu_mean = torch.mean(neu[:, :, 0], dim=0)  # has shape (num_channels,)
        neu_std = torch.std(neu[:, :, 0], dim=0)  # has shape (num_channels,)
        neu_mean = neu_mean.unsqueeze(0).unsqueeze(2)  # has shape (1, num_channels, 1)
        neu_std = neu_std.unsqueeze(0).unsqueeze(2)  # has shape (1, num_channels, 1)
        neu = (neu - neu_mean) / (neu_std + 1e-6)

        # normalize output velocities (new by JC)
        vel_mean = torch.mean(vel, dim=0)  # has shape (num_dof,)
        vel_std = torch.std(vel, dim=0)  # has shape (num_dof,)
        vel_mean = vel_mean.unsqueeze(0)  # has shape (1, num_dof)
        vel_std = vel_std.unsqueeze(0)  # has shape (1, num_dof)
        vel = (vel - vel_mean) / (vel_std + 1e-6)
    else:
        neu_mean = None
        neu_std = None
        vel_mean = None
        vel_std = None
    
    #create pytorch datasets and then dataloaders
    ds = TensorDataset(neu, vel)

    #since we know how long we're training, val dataset can just be the same as training
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    dl2 = DataLoader(ds, batch_size=len(ds), shuffle=False)

    model = init_model(model_class, model_type, neu.shape[1], vel.shape[1])
    opt, scheduler = init_opt(model, model_type)

    #run fit function (model is trained here)
    val_losses, train_losses, loss_iters = fit(model, 'TCN', opt, scheduler, dl, dl2)

    #train scaler
    scaler = generate_output_scaler(model, dl2, num_outputs=vel.shape[1])
    model = model.cpu()
    
    return model, scaler, (val_losses, train_losses, loss_iters), (neu_mean, neu_std, vel_mean, vel_std)

# Basic Fit Function for forward/backprop dependent models
def fit(model, model_type, opt, scheduler, dl, val_dl, scaler_used=True, normalized=False):
    loss_fn = torch.nn.MSELoss()
    train_losses = []
    val_losses = []
    loss_iters = []
    
    with open(config.training_params_path) as f:
        training_params = yaml.load(f, Loader=yaml.FullLoader)[model_type]
    
    if training_params['use_scheduler'] == True:
        scheduler_update = training_params['scheduler_update']
        min_lr = float(training_params['learning_rate'])/training_params['num_lr_steps']

    if normalized:
        max_epoch = training_params['max_epoch_norm']
        print(max_epoch)
    else:
        max_epoch = training_params['max_epoch']

    iter = 0
    for epoch in range(max_epoch):
        epoch_train_losses = []
        for x, y in dl:
            model.train()

            # add noise if needed
            if training_params['noise_std'] or training_params['bias_std']:
                x = add_training_noise(x, training_params['noise_std'], training_params['bias_std'])

            # reset gradients
            opt.zero_grad()

            # 1. Generate your predictions
            yh = model(x)

            if isinstance(yh, tuple):
                # RNNs return y, h
                yh = yh[0]
            
            # 2. Find Loss
            loss = loss_fn(yh, y)
            epoch_train_losses.append(loss.item())
            # 3. Calculate gradients with respect to weights/biases
            loss.backward()

            # 4. Adjust your weights
            opt.step()

            # scheduler is based on 100 iter increments
            if training_params['use_scheduler'] and ((iter % scheduler_update == 0)):
                val_loss = evaluate_model_acc(model, 
                                              loss_fn, 
                                              val_dl, 
                                              epoch,
                                              iter, 
                                              max_epoch,
                                              scaler_used=scaler_used,
                                              verbose=False)

                if scheduler is not None:
                    scheduler.step(val_loss)
                    print('steppin')
                
                print(opt.param_groups[0]['lr'])
                if opt.param_groups[0]['lr'] < float(min_lr):
                    print(f'scheduler stop, final result:')
                    print('Epoch [{}/{}], iter {} Validation Loss: {:.4f}'.format(epoch, 
                                                                                  max_epoch - 1, 
                                                                                  iter,
                                                                                  val_loss.item()))
                    val_losses.append(val_loss.item())
                    train_loss = np.mean(np.asarray(epoch_train_losses))
                    train_losses.append(train_loss)
                    loss_iters.append(iter)
                    return val_losses, train_losses, loss_iters

            iter = iter + 1

        # Per-epoch eval
        val_loss = evaluate_model_acc(model, loss_fn, val_dl, epoch, iter - 1, training_params['max_epoch'], scaler_used=scaler_used, verbose=True)
        train_loss = np.mean(np.asarray(epoch_train_losses))
        train_losses.append(train_loss)
        val_losses.append(val_loss.item())
        loss_iters.append(iter - 1)
    
    return val_losses, train_losses, loss_iters

def evaluate_model_acc(model, loss_fn, val_dl, epoch, iter, max_epoch, scaler_used=True, verbose=True):
    for x2, y2 in val_dl:
        with torch.no_grad():
            model.eval()

            yh = model(x2)
            if isinstance(yh, tuple):
                # RNNs return y, h
                yh = yh[0]
            
            if scaler_used:
                scaler = generate_output_scaler(model, val_dl, num_outputs=y2.shape[1], verbose=False)
                yh = scaler.scale(yh)

            val_loss = loss_fn(yh.cpu(), y2.cpu())
                
        if verbose:
            print('Epoch [{}/{}], iter {} Validation Loss: {:.4f}'.format(epoch, 
                                                                          max_epoch - 1, 
                                                                          iter,
                                                                          val_loss.item()))
    return val_loss

def generate_output_scaler(model, loader, num_outputs=2, verbose=True):
    """Returns a scaler object that scales the output of a decoder

    Args:
        model:      model
        loader:     dataloader
        num_outputs:  how many outputs (2)

    Returns:
        scaler: An OutputScaler object that takes returns scaled version of input data. If refit, this is the
                composition of the original and new scalers.
    """
    # fit constants using regression
    model.eval()  # set model to evaluation mode
    batches = len(list(loader))
    with torch.no_grad():
        for x, y in loader:

            x = x.to(device=config.device, dtype=config.dtype)  # move to device, e.g. GPU
            y = y.to(device=config.device, dtype=config.dtype)
            yhat = model(x)

            if isinstance(yhat, tuple):
                # RNNs return y, h
                yhat = yhat[0]

            num_samps = yhat.shape[0]
            num_outputs = yhat.shape[1]
            yh_temp = torch.cat((yhat, torch.ones([num_samps, 1]).to(config.device)), dim=1)

            # train ~special~ theta
            # (scaled velocities are indpendent of each other - this is the typical method)
            # Theta has the following form: [[w_x,   0]
            #                                [0,   w_y]
            #                                [b_x, b_y]]
            theta = torch.zeros((num_outputs + 1, num_outputs)).to(device=config.device, dtype=config.dtype)
            for i in range(num_outputs):
                yhi = yh_temp[:, (i, -1)]
                thetai = torch.matmul(torch.mm(torch.pinverse(torch.mm(torch.t(yhi), yhi)), torch.t(yhi)), y[:, i])
                theta[i, i] = thetai[0]  # gain
                theta[-1, i] = thetai[1]  # bias
                if verbose:
                    print("Finger %d RR Calculated Gain, Offset: %.6f, %.6f" % (i, thetai[0], thetai[1]))

    gains = np.zeros((1, num_outputs))
    biases = np.zeros((1, num_outputs))
    for i in range(num_outputs):
        gains[0, i] = theta[i, i]
        biases[0, i] = theta[num_outputs, i]

    return OutputScaler(gains, biases)

def add_training_noise(x, bias_std, noise_std):
    if bias_std:
        # bias is constant across time (i.e. the 3 conv inputs), but different for each channel & batch
        # biases = torch.normal(0, bias_std, x.shape[:2]).unsqueeze(2).repeat(1, 1, x.shape[2])
        biases = torch.normal(torch.zeros(x.shape[:2]), bias_std).unsqueeze(2).repeat(1, 1, x.shape[2])
        x = x + biases.to(device=config.device)

    if noise_std:
        # adds white noise to each channel and timepoint (independent)
        # noise = torch.normal(0, noise_std, x.shape)
        noise = torch.normal(torch.zeros_like(x), noise_std)
        x = x + noise.to(device=config.device)
    
    return x

class OutputScaler:
    def __init__(self, gains, biases):
        """An object to linearly scale data, like the output of a neural network

        Args:
            gains (1d np array):  [1,NumOutputs] array of gains
            biases (1d np array):           [1,NumOutputs] array of biases
        """
        self.gains = gains
        self.biases = biases

    def scale(self, data):
        """
        data should be an numpy array/tensor of shape [N, NumOutputs]
        :param data:    np.ndarray or torch.Tensor, data to scale
        :return scaled_data np.ndarray or torch.Tensor, returns either according to what was input
        """

        # check if input is tensor or numpy
        isTensor = False
        if type(data) is torch.Tensor:
            isTensor = True
            data = data.cpu().detach().numpy()
        N = data.shape[0]

        # scale data
        scaled_data = np.tile(self.gains, (N, 1)) * data + np.tile(self.biases, (N, 1))

        # convert back to tensor if needed
        if isTensor:
            scaled_data = torch.from_numpy(scaled_data)

        return scaled_data

    def unscale(self, data):
        """Data should be an numpy array/tensor of shape [N, NumOutputs].
            Performs the inverse of the scale function (used in Refit)"""
        N = data.shape[0]
        # unscaled_data = (data / np.tile(self.gains, (N, 1))) - np.tile(self.biases, (N, 1))
        unscaled_data = (data - np.tile(self.biases, (N, 1))) / np.tile(self.gains, (N, 1))
        return unscaled_data
