import torch
import torchvision
import numpy as np
import os
import time
import s3gd_cuda
import matplotlib.pyplot as plt
from shutil import copyfile
import pickle
from scipy import stats
import tables
import json

torch.set_num_threads(1)
from config import CONFIGS
from plot_utils import plot_error

surr_grad_spike_cuda = s3gd_cuda.surr_grad_spike
s3gd_backward_cuda = s3gd_cuda.s3gd_w_backward
s3gd_s_backward_master = s3gd_cuda.s3gd_s_backward_master



############################################ DATASET LOADING ############################################
def open_file(hdf5_file_path):
    fileh = tables.open_file(hdf5_file_path, mode='r')
    units = fileh.root.spikes.units
    times = fileh.root.spikes.times
    labels = fileh.root.labels
    return fileh, units, times, labels

def load_dataset(prs):

    fileh_train = None
    fileh_test = None
    if prs['dataset_id'] == 'fmnist':
        # Here we load the Dataset
        root = os.path.expanduser(prs.get('dataset_path', CONFIG['dataset_path']))
        train_dataset = torchvision.datasets.FashionMNIST(root, train=True, transform=None, target_transform=None,
                                                          download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root, train=False, transform=None, target_transform=None,
                                                         download=True)
        # Standardize data
        x_train = np.array(train_dataset.data, dtype=np.float)
        x_train = x_train.reshape(x_train.shape[0], -1) / 255
        x_test = np.array(test_dataset.data, dtype=np.float)
        x_test = x_test.reshape(x_test.shape[0], -1) / 255
        y_train = np.array(train_dataset.targets, dtype=np.int)
        y_test = np.array(test_dataset.targets, dtype=np.int)
        print("Opening dataset in: {}".format(root))
    elif prs['dataset_id'] == 'nmnist':
        root = os.path.expanduser(prs.get('dataset_path', CONFIG['dataset_path']))
        fileh_train, units_train, times_train, y_train = open_file(os.path.join(root, 'train.h5'))
        fileh_test, units_test, times_test, y_test = open_file(os.path.join(root, 'test.h5'))
        x_train = {'times': times_train, 'units': units_train}
        x_test = {'times': times_test, 'units': units_test}
        print("Opening dataset in: {}".format(os.path.join(root, 'train.h5')))
    elif prs['dataset_id'] == 'SHD':
        root = os.path.expanduser(prs.get('dataset_path', CONFIG['dataset_path']))
        fileh_train, units_train, times_train, y_train = open_file(os.path.join(root, 'train_shd.h5'))
        fileh_test, units_test, times_test, y_test = open_file(os.path.join(root, 'test_shd.h5'))
        x_train = {'times': times_train, 'units': units_train}
        x_test = {'times': times_test, 'units': units_test}
        print("Opening dataset in: {}".format(os.path.join(root, 'train.h5')))
    else:
        raise ValueError('Dataset must be either fmnist, nmnist, or SHD')

    return fileh_train, fileh_test, x_train, x_test, y_train, y_test


############################################ DATA GENERATION ############################################
def current2firing_time(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    idx = x < thr
    x = np.clip(x, thr + epsilon, 1e9)
    T = tau * np.log(x / (x - thr))
    T[idx] = tmax
    return T


def sparse_data_generator_torchvision(X, y, prs, shuffle=True):
    """ This generator takes datasets in analog format and generates spiking network input as sparse tensors.

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """

    batch_size = prs['batch_size']
    time_step = prs['time_step']
    nb_steps = prs['nb_steps']
    nb_inputs = prs['nb_inputs']
    device = prs['device']

    labels_ = np.array(y, dtype=np.int)
    number_of_batches = len(X) // batch_size
    sample_index = np.arange(len(X))

    # compute discrete firing times
    tau_eff = prs['tau_eff'] if prs['tau_eff'] is not None else 20e-3
    tau_eff /= time_step
    firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=np.int)
    unit_numbers = np.arange(nb_inputs)

    if shuffle:
        if prs.get('set_random_state', False):
            r = np.random.RandomState(prs['seed'])
            r.shuffle(sample_index)
        else:
            np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]

        coo = [[] for i in range(3)]
        for bc, idx in enumerate(batch_index):
            c = firing_times[idx] < nb_steps
            times, inputs = firing_times[idx][c], unit_numbers[c]

            batch = [bc for _ in range(len(times))]
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(inputs)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_inputs])).to(device)
        y_batch = torch.tensor(labels_[batch_index], device=device).long()

        yield X_batch.to(device=device), y_batch.to(device=device)

        counter += 1


def sparse_data_generator_h5(X, y, prs, shuffle=True):

    batch_size = prs['batch_size']
    nb_steps = prs['nb_steps']
    nb_units = prs['nb_inputs']
    time_step = prs['time_step']
    device = prs['device']

    units, times = X['units'], X['times']

    labels = np.array(y, dtype=np.int)
    num_samples = len(labels)  # Number of samples in data
    sample_index = np.arange(num_samples)
    number_of_batches = num_samples // batch_size

    if shuffle:
        if prs.get('set_random_state', False):
            r = np.random.RandomState(prs['seed'])
            r.shuffle(sample_index)
        else:
            np.random.shuffle(sample_index)

    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter:min(num_samples, batch_size * (counter + 1))]
        batch_size = len(batch_index)

        coo = [[] for i in range(3)]
        for bc, idx in enumerate(batch_index):
            ts = (np.round(times[idx]*1./time_step).astype(np.int))
            us = units[idx]
            if prs['dataset_id']=='nmnist':
                us = us % (34*34)

            # Constrain spike length
            idxs = (ts < nb_steps)
            ts = ts[idxs]
            us = us[idxs]

            batch = [bc for _ in range(ts.size)]
            coo[0].extend(batch)
            coo[1].extend(ts.tolist())
            coo[2].extend(us.tolist())

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to(device)
        y_batch = torch.tensor(labels[batch_index], device=device).long()

        # X_batch[X_batch[:] > 1.] = 1.

        yield X_batch.to(device=device), y_batch.to(device=device)

        counter += 1


############################################ REGULARISATION LOSSES ############################################4
def loss_upper(spikes, prs, nb_hidden):
    tmp = (1./nb_hidden)*spikes.sum(1).sum(1) - prs['f_upper']  # Average number of spikes per neuron
    loss_upper = prs['lambda_upper'] * ((torch.clamp(tmp, min=0.))**prs['p_up'])
    return loss_upper

def loss_lower(spikes, prs, nb_hidden):
    tmp = prs['f_lower'] - spikes.sum(1)
    tmp = (torch.clamp(tmp, min=0.))**2
    loss_lower = prs['lambda_lower'] * (tmp.sum(1))
    return loss_lower / nb_hidden

def loss_reg(spikes, prs, hidden_level=0):
    if hidden_level==0:
        nb_hidden = prs['nb_hidden']
    else:
        nb_hidden = prs['nb_hidden2']
    l_up = loss_upper(spikes, prs, nb_hidden)
    l_low = loss_lower(spikes, prs, nb_hidden)
    l_reg = (l_up + l_low).sum()
    return (1./prs['batch_size'])*l_reg


############################################ NETWORK INIT ############################################
def init(prs):
    prs['dataset_type'] = prs.get('dataset_type', CONFIG['dataset_type'])
    # Params
    prs['nb_inputs'] = prs.get('nb_inputs', CONFIG['nb_inputs'])
    prs['nb_hidden'] = prs.get('nb_hidden', CONFIG['nb_hidden'])
    prs['nb_hidden2'] = prs.get('nb_hidden2', CONFIG['nb_hidden2'])
    prs['nb_outputs'] = prs.get('nb_outputs', CONFIG['nb_outputs'])
    prs['time_step'] = prs.get('time_step', CONFIG['time_step'])
    prs['nb_steps'] = prs.get('nb_steps', CONFIG['nb_steps'])
    prs['batch_size'] = prs.get('batch_size', CONFIG['batch_size'])
    prs['lr'] = prs.get('lr', CONFIG['lr'])
    prs['betas'] = prs.get('betas', CONFIG['betas'])
    prs['nb_epochs'] = prs.get('nb_epochs', CONFIG['nb_epochs'])
    prs['dtype'] = prs.get('dtype', CONFIG['dtype'])

    prs['th'] = prs.get('th', CONFIG['th'])
    prs['b_th'] = prs.get('b_th', CONFIG['b_th'])
    prs['weight_multiplier'] = prs.get('weight_multiplier', CONFIG['weight_multiplier'])
    prs['tau_mem'] = prs.get('tau_mem', CONFIG['tau_mem'])
    prs['tau_readout'] = prs.get('tau_readout', CONFIG['tau_readout'])
    prs['beta'] = float(np.exp(-prs['time_step'] / prs['tau_mem']))
    prs['beta_readout'] = float(np.exp(-prs['time_step'] / prs['tau_readout']))
    prs['dataset_id'] = prs.get('dataset_id', CONFIG['dataset_id'])
    prs['dataset_path'] = prs.get('dataset_path', CONFIG['dataset_path'])
    prs['tau_eff'] = prs.get('tau_eff', CONFIG['tau_eff'])

    # Regularisation
    prs['lambda_upper'] = prs.get('lambda_upper', CONFIG['lambda_upper'])
    prs['lambda_lower'] = prs.get('lambda_lower', CONFIG['lambda_lower'])
    prs['f_upper'] = prs.get('f_upper', CONFIG['f_upper'])
    prs['f_lower'] = prs.get('f_lower', CONFIG['f_lower'])
    prs['p_up'] = prs.get('p_up', CONFIG['p_up'])

    if torch.cuda.is_available():
        prs['device'] = 'cuda'
    else:
        raise ('Sparse spike backpropagation requires CUDA available!')

    # Dataset
    fileh_train, fileh_test, x_train, x_test, y_train, y_test = load_dataset(prs)

    # Weights
    prs['seed'] = prs.get('seed', CONFIG['seed'])
    np.random.seed(prs['seed'])
    torch.random.manual_seed(prs['seed'])
    torch.manual_seed(prs['seed'])
    w1 = torch.empty((prs['nb_inputs'], prs['nb_hidden']), device=prs['device'], dtype=prs['dtype'], requires_grad=True)
    w2 = torch.empty((prs['nb_hidden'], prs['nb_hidden2']), device=prs['device'], dtype=prs['dtype'],
                     requires_grad=True)
    w3 = torch.empty((prs['nb_hidden2'], prs['nb_outputs']), device=prs['device'], dtype=prs['dtype'],
                     requires_grad=True)

    if prs['dataset_id'] == 'fmnist':
        prs['weight_scale'] = prs['weight_multiplier'] * (1.0 - prs['beta'])
        torch.nn.init.normal_(w1, mean=0.0, std=prs['weight_scale'] / np.sqrt(prs['nb_inputs']))
        torch.nn.init.normal_(w2, mean=0.0, std=prs['weight_scale'] / np.sqrt(prs['nb_hidden']))
        torch.nn.init.normal_(w3, mean=0.0, std=prs['weight_scale'] / np.sqrt(prs['nb_hidden2']))
    elif prs['dataset_id'] == 'nmnist':
        bound = 1. / np.sqrt(prs['nb_inputs'])
        torch.nn.init.uniform_(w1, -bound, bound)
        bound = 1. / np.sqrt(prs['nb_hidden'])
        torch.nn.init.uniform_(w2, -bound, bound)
        bound = 1. / np.sqrt(prs['nb_hidden2'])
        torch.nn.init.uniform_(w3, -bound, bound)
    elif prs['dataset_id'] == 'SHD':
        bound = 1. / np.sqrt(prs['nb_inputs'])
        torch.nn.init.uniform_(w1, -bound, bound)
        bound = 1. / np.sqrt(prs['nb_hidden'])
        torch.nn.init.uniform_(w2, -bound, bound)
        bound = 1. / np.sqrt(prs['nb_hidden2'])
        torch.nn.init.uniform_(w3, -bound, bound)

    params = [w1, w2, w3]

    print("Init finished")

    return fileh_train, fileh_test, x_train, x_test, y_train, y_test, prs, params

############################################ LAYERS ############################################
def SNNReadout(spk_rec, weight, prs):
    batch_size = prs['batch_size']
    nb_steps = prs['nb_steps']
    nb_outputs = prs['nb_outputs']
    device = prs['device']
    dtype = prs['dtype']

    h2 = torch.einsum("abc,cd->abd", (spk_rec, weight))
    out = torch.zeros((batch_size, nb_outputs), device=device, dtype=dtype)
    out_rec = [out]
    for t in range(nb_steps - 1):
        new_out = prs['beta_readout'] * out + (1-prs['beta_readout'])*h2[:, t, :]
        out = new_out
        out_rec.append(out)
    out_rec = torch.stack(out_rec, dim=1)
    return out_rec


def SNNLayerOrig(inputs, weight, dummy, prs):
    batch_size = prs['batch_size']
    nb_inputs = inputs.shape[2]  # Batch, Time, Units
    nb_steps = prs['nb_steps']
    nb_hidden = prs['nb_hidden'] if dummy is None else prs['nb_hidden2']
    beta = prs['beta']
    th = prs['th']
    device = prs['device']
    dtype = prs['dtype']
    spike_fn = prs['spike_fn']

    # Forward
    h1 = torch.einsum("abc,cd->abd", (inputs, weight))  # Batch,Time,Input x Input,Output -> Batch,Time,Output
    mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    spk = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
    rst = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

    mem_rec = [mem]
    spk_rec = [spk]

    # Compute hidden layer activity
    for t in range(nb_steps - 1):
        new_mem = beta * mem + h1[:, t, :] - rst
        mem = new_mem

        mthr = mem - th
        out = spike_fn(mthr)
        rst = torch.zeros_like(mem)
        c = (mthr > 0)
        rst[c] = torch.ones_like(mem)[c]

        mem_rec.append(mem)
        spk_rec.append(out)

    mem_rec = torch.stack(mem_rec, dim=1)
    spk_rec = torch.stack(spk_rec, dim=1)

    b_th = prs['b_th']
    aout_idxs = torch.nonzero(torch.logical_and(mem_rec > b_th,
                                                mem_rec < 2. * th - b_th))  # Indices of mem_rec above threshold mask [b, t, j]
    return spk_rec, aout_idxs


class SparseSNNLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, weight, ain_idxs, prs):
        batch_size = prs['batch_size']
        nb_inputs = inputs.shape[2]  # Batch, Time, Units
        nb_steps = prs['nb_steps']
        nb_hidden = prs['nb_hidden'] if ain_idxs is None else prs['nb_hidden2']
        beta = prs['beta']
        th = prs['th']
        b_th = prs['b_th']
        device = prs['device']
        dtype = prs['dtype']

        # Forward
        with torch.no_grad():
            h1 = torch.einsum("abc,cd->abd", (inputs, weight))  # Batch,Time,Input x Input,Output -> Batch,Time,Output
            mem = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
            spk = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
            rst = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)

            trace = torch.zeros((batch_size, nb_inputs), device=device, dtype=torch.float)

            mem_rec = [mem]
            spk_rec = [spk]
            spk_trace = [trace]

            # Compute first hidden layer activity
            for t in range(nb_steps - 1):

                new_mem = beta * mem + h1[:, t, :] - rst
                mem = new_mem

                mthr = mem - th
                out = torch.zeros_like(mthr)
                out[mthr > 0] = 1.0
                rst = torch.zeros_like(mem)
                c = (mthr > 0)
                rst[c] = torch.ones_like(mem)[c]

                # ============================================ NEW STUFF ============================================= #
                trace = beta * trace + inputs[:, t, :]
                spk_trace.append(trace)
                # ==================================================================================================== #

                mem_rec.append(mem)
                spk_rec.append(out)

            mem_rec = torch.stack(mem_rec, dim=1)
            spk_rec = torch.stack(spk_rec, dim=1)
            spk_trace = torch.stack(spk_trace, dim=1)

            # ============================================ NEW STUFF ============================================= #
            aout_idxs = torch.nonzero(torch.logical_and(mem_rec > b_th,
                                                     mem_rec < 2. * th - b_th))  # Indices of mem_rec above threshold mask [b, t, j]

            # Membrane - th
            aout_mem = (mem_rec[aout_idxs[:, 0], aout_idxs[:, 1], aout_idxs[:, 2]] - th)
            # Indices out
            aout_b = aout_idxs[:, 0]
            aout_t = aout_idxs[:, 1]
            aout_i = aout_idxs[:, 2]
            # Indices in
            ain_b = ain_idxs[:, 0] if ain_idxs is not None else None
            ain_t = ain_idxs[:, 1] if ain_idxs is not None else None
            ain_i = ain_idxs[:, 2] if ain_idxs is not None else None
            # ==================================================================================================== #

        # Save for backward
        alphas = (torch.full((nb_steps,), beta, device=device) ** torch.arange(nb_steps, device=device))
        ctx.save_for_backward(spk_trace, aout_b, aout_t, aout_i, aout_mem, ain_b, ain_t, ain_i, alphas, weight)
        ctx.batch_size = batch_size
        ctx.nb_steps = nb_steps
        ctx.nb_hidden = nb_hidden
        ctx.nb_inputs = nb_inputs
        spk_rec.requires_grad = True
        aout_idxs.requires_grad = False
        return spk_rec, aout_idxs

    @staticmethod
    def backward(ctx, grad_output, grad_dummy1):
        spk_trace, aout_b, aout_t, aout_i, aout_mem, ain_b, ain_t, ain_i, alphas, weight = ctx.saved_tensors
        batch_size, nb_steps, nb_inputs, nb_hidden = ctx.batch_size, ctx.nb_steps, ctx.nb_inputs, ctx.nb_hidden
        grad_input, grad_weights = None, None

        # Active output values gradient
        grad_output = grad_output[aout_b, aout_t, aout_i]
        grad_output_idxs = torch.nonzero(grad_output, as_tuple=True)[0]
        grad_output = grad_output[grad_output_idxs]
        aout_b = aout_b[grad_output_idxs]
        aout_t = aout_t[grad_output_idxs]
        aout_i = aout_i[grad_output_idxs]
        aout_mem = aout_mem[grad_output_idxs]
        if grad_output.numel()==0:
            grad_weights = torch.zeros_like(weight)
            grad_input = torch.zeros((batch_size, nb_steps, nb_inputs), device=grad_output.device)
            return grad_input, grad_weights, None, None
        ds_out = surr_grad_spike_cuda(aout_mem, grad_output)

        # Weight gradient
        if ctx.needs_input_grad[1]:
            grad_weights = s3gd_backward_cuda(spk_trace, aout_b, aout_t, aout_i, ds_out,
                                              nb_inputs, nb_hidden)

        # Input spikes gradient
        if ctx.needs_input_grad[0]:  # This is false for first layer

            # Reorder active output indices to (b,j,t) format
            bjt = aout_b * nb_hidden * nb_steps + aout_i * nb_steps + aout_t
            bjt, bjt_idx = torch.sort(bjt)
            ds_out = ds_out[bjt_idx]

            # Find indices to start from and frequencies (output)
            b = bjt // (nb_hidden * nb_steps)
            j = (bjt - b * nb_hidden * nb_steps) // nb_steps
            ts_out = bjt - b * nb_hidden * nb_steps - j * nb_steps
            bj = b * nb_hidden + j
            aout_bj_freqs = torch.bincount(bj, minlength=batch_size*nb_hidden)  # How many t for a given b and j
            idxs = torch.nonzero(aout_bj_freqs)  # Where to put the results
            ends = torch.cumsum(aout_bj_freqs[idxs], dim=0) - 1  # Indices
            bj_ends = torch.full((batch_size*nb_hidden,), -1, device="cuda")
            bj_ends[idxs] = ends

            # Find indices to start from and frequencies (input)
            bt_input_full = ain_b * nb_steps + ain_t
            bt_in_unique = torch.unique(bt_input_full)
            b = bt_in_unique // nb_steps
            ts_in = bt_in_unique - b * nb_steps
            ain_b_freqs = torch.bincount(b, minlength=batch_size)  # Get frequencies of input batch
            idxs = torch.nonzero(ain_b_freqs)  # Where to put the results
            ends = torch.cumsum(ain_b_freqs[idxs], dim=0) - 1  # Indices
            b_ends = torch.full((batch_size,), -1, device="cuda")
            b_ends[idxs] = ends

            #########################
            # Get bj_out_unique
            bj_out_unique = torch.unique(bj)
            total_num_threads = bj_out_unique.numel()

            # Get counts
            b_out_unique = bj_out_unique // (nb_hidden)
            b_in_unique = bt_in_unique // (nb_steps)
            b_out_unique_freqs = torch.bincount(b_out_unique, minlength=batch_size)  # Get frequencies of input batch
            b_in_unique_freqs = torch.bincount(b_in_unique, minlength=batch_size)  # Get frequencies of input batch

            # Sparse matrix size
            bM_freqs = torch.cumsum(b_out_unique_freqs * b_in_unique_freqs, 0)
            M = bM_freqs[-1]
            bM_freqs = (bM_freqs).roll(1)
            bM_freqs[0] = 0

            # Get starts for each bj
            bM_starts = torch.roll(bM_freqs, 1)
            bM_starts[0] = 0  # Start for batch
            bj_out_unique_freqs = torch.bincount(bj_out_unique, minlength=batch_size * nb_hidden)  # Get frequencies of each unique bj

            bj_out_unique_freqs = bj_out_unique_freqs.reshape(batch_size, -1).cumsum(1).roll(1)
            bj_out_unique_freqs[:, 0] = 0
            bj_out_unique_freqs = bj_out_unique_freqs.reshape(-1)  # Flatten
            ###############################################################


            # Call kernel to compute ds gradient
            ain_bt_freqs = torch.bincount(bt_input_full, minlength=batch_size*nb_steps)  # Get frequencies of input batch
            ain_bt_starts = torch.cumsum(ain_bt_freqs, 0).roll(1)
            ain_bt_starts[0] = 0
            grad_input = s3gd_s_backward_master(
                                                 # Computing
                                                 ds_out,  # Values to compute deltas (\red{dS[bjt]})
                                                 ts_out,  # Times active output values to compute deltas
                                                 bj_ends,  # Index fot ts_out with last time to read. Points to last time for each bj
                                                 aout_bj_freqs,  # How many t we need to compute for a given b&j
                                                 # Recording
                                                 ts_in,  # Times to record
                                                 b_ends,  # Index  for ts_in with last time to record each batch
                                                 ain_b_freqs,  # How many t we need to record for a given batch (for tensor creation)
                                                 # Other
                                                 alphas,  # Powers of alpha
                                                 weight,
                                                 # Writing deltas
                                                 bM_freqs,
                                                 b_in_unique_freqs,
                                                 bj_out_unique_freqs,
                                                 bj_out_unique,
                                                 # Computing gradient
                                                 ain_bt_freqs,
                                                 ain_bt_starts,
                                                 ain_i,
                                                 # Constants
                                                 M, total_num_threads, batch_size, nb_steps, nb_inputs, nb_hidden
                                                 )
        return grad_input, grad_weights, None, None


my_SparseSNNLayer = SparseSNNLayer.apply


############################################ TRAIN AND TEST ACC. ############################################
def run_snn(input, params, prs):
    SNN_function = prs['SNN_function']

    # Parameters
    w1, w2, w3 = params
    ### FIRST LAYER FORWARD ###
    spk_rec1, aout_idxs1 = SNN_function(input, w1, None, prs)
    # ### SECOND LAYER FORWARD ###
    spk_rec2, _ = SNN_function(spk_rec1, w2, aout_idxs1, prs)
    ### READOUT LAYER FORWARD ###
    out_rec = SNNReadout(spk_rec2, w3, prs)
    return out_rec, None


def compute_classification_accuracy(x_data, y_data, params, prs):
    """ Computes classification accuracy on supplied data in batches. """
    if prs['dataset_type'] == 'torchvision':
        sparse_data_generator = sparse_data_generator_torchvision
    elif prs['dataset_type'] == 'h5':
        sparse_data_generator = sparse_data_generator_h5
    accs = []
    for x_local, y_local in sparse_data_generator(x_data, y_data, prs, shuffle=False):
        if len(accs) % 25 == 0 and len(accs) > 0:
            print("Progress:{:d}/{:d}".format(len(accs), len(x_data) // prs['batch_size']))
        output, _ = run_snn(x_local.to_dense(), params, prs)
        m, _ = torch.max(output, 1)  # max over time
        _, am = torch.max(m, 1)  # argmax over output units
        tmp = np.mean((y_local == am).detach().cpu().numpy())  # compare to labels
        accs.append(tmp)
    return np.mean(accs)


############################################ TRAIN AND TIME ############################################
def time_snn(x_local, y_local, params, prs, warmup=None):
    """ We detached output of first layer to run the second (readout) so that it we do not backpropagate
    to the first layer when calling loss_val.backward() """

    SNN_function = prs['SNN_function']
    log_softmax_fn = torch.nn.LogSoftmax(dim=1)
    loss_fn = torch.nn.NLLLoss()
    # Parameters
    w1, w2, w3 = params

    # ################### Warmup ################### #
    for n in range(warmup):
        ### FIRST LAYER FORWARD ###
        spk_rec11_wu, aout_idxs11 = SNN_function(x_local, w1, None, prs)
        ### SECOND LAYER FORWARD ###
        # Copy spikes from first layer (detached from first layer)
        spk_rec12_wu = spk_rec11_wu.detach().clone()
        spk_rec12_wu.requires_grad = True
        spk_rec12_wu.retain_grad()
        spk_rec21_wu, _ = SNN_function(spk_rec12_wu, w2, aout_idxs11, prs)
        ### READOUT LAYER FORWARD ###
        # Copy spikes from second layer (detached from second layer)
        spk_rec22_wu = spk_rec21_wu.detach().clone()
        spk_rec22_wu.requires_grad = True
        spk_rec22_wu.retain_grad()
        # Run readout and loss
        out_rec_wu = SNNReadout(spk_rec22_wu, w3, prs)
        m, _ = torch.max(out_rec_wu, 1)
        log_p_y = log_softmax_fn(m)
        loss_val_wu = loss_fn(log_p_y, y_local)
        ### READOUT LAYER BACKWARD ###
        spk_rec22_wu.grad = None
        w3.grad = None
        loss_val_wu.backward()
        layer2_spikes_grad = spk_rec22_wu.grad.detach().clone()  # These are inputs to the second layer backward
        ### SECOND LAYER BACKWARD ###
        spk_rec12_wu.grad = None
        w2.grad = None
        spk_rec21_wu.backward(layer2_spikes_grad)
        layer1_spikes_grad = spk_rec12_wu.grad.detach().clone()  # These are inputs to the first layer backward
        ### FIRST LAYER BACKWARD ###
        w1.grad = None
        spk_rec11_wu.backward(layer1_spikes_grad)
    # ############################################## #

    # ################### TIME ################### #
    ### FIRST LAYER FORWARD ###
    spk_rec11, aout_idxs11 = SNN_function(x_local, w1, None, prs)
    ### SECOND LAYER FORWARD ###
    # Copy spikes from first layer (detached from first layer)
    spk_rec12 = spk_rec11.detach().clone()
    spks1 = torch.count_nonzero(spk_rec12).item()
    act1 = aout_idxs11.shape[0]
    spk_rec12.requires_grad = True
    spk_rec12.retain_grad()
    # Run second layer
    torch.cuda.synchronize()
    curr_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    start = time.time()
    spk_rec21, aout_idxs21 = SNN_function(spk_rec12, w2, aout_idxs11, prs)
    torch.cuda.synchronize()
    fwd_time = time.time() - start

    peak_alloc = torch.cuda.max_memory_allocated()
    fwd_alloc = peak_alloc-curr_alloc
    ### READOUT LAYER FORWARD ###
    # Copy spikes from first layer (dettached from first layer)
    spk_rec22 = spk_rec21.detach().clone()
    spks2 = torch.count_nonzero(spk_rec22).item()
    act2 = aout_idxs21.shape[0]
    spk_rec22.requires_grad = True
    spk_rec22.retain_grad()
    # Run readout and loss
    out_rec = SNNReadout(spk_rec22, w3, prs)
    m, _ = torch.max(out_rec, 1)
    log_p_y = log_softmax_fn(m)
    loss_val = loss_fn(log_p_y, y_local)

    ### REGULARIZATION LOSS ###
    spk_rec13 = spk_rec12.detach().clone()
    spk_rec13.requires_grad = True
    spk_rec13.retain_grad()
    spk_rec23 = spk_rec22.detach().clone()
    spk_rec23.requires_grad = True
    spk_rec23.retain_grad()
    torch.cuda.synchronize()
    start = time.time()
    l_reg1 = loss_reg(spk_rec13, prs, hidden_level=0)
    l_reg2 = loss_reg(spk_rec23, prs, hidden_level=1)
    torch.cuda.synchronize()
    fwd_time += time.time() - start

    loss = loss_val + l_reg1 + l_reg2

    ### READOUT LAYER BACKWARD ###
    # Backward the readout
    # loss backward modifies spk_rec22.grad, spk_rec12.grad and w3.grad
    # second layer backward modifies spk_rec12.grad and w2.grad
    # first layer backward modifies w1.grad
    w1.grad = None
    spk_rec12.grad = None
    spk_rec13.grad = None
    spk_rec22.grad = None
    spk_rec23.grad = None
    w2.grad = None
    w3.grad = None
    loss.backward()
    layer2_spikes_grad = spk_rec22.grad.detach().clone() + spk_rec23.grad.detach().clone()  # Regular loss + Regularization loss
    ### SECOND LAYER BACKWARD ###
    torch.cuda.synchronize()
    curr_alloc = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    start = time.time()
    spk_rec21.backward(layer2_spikes_grad)
    torch.cuda.synchronize()
    bwd_time = time.time() - start

    peak_alloc = torch.cuda.max_memory_allocated()
    bwd_alloc = peak_alloc-curr_alloc

    layer1_spikes_grad = spk_rec12.grad.detach().clone() + spk_rec13.grad.detach().clone()  # Regular loss + Regularization loss
    ### FIRST LAYER BACKWARD ###
    spk_rec11.backward(layer1_spikes_grad)
    # ############################################## #
    return loss_val, fwd_time, bwd_time, fwd_alloc, bwd_alloc, spks1, spks2, act1, act2


def train_and_time(x_data, y_data, params, prs):
    nb_epochs = prs['nb_epochs']
    lr = prs['lr']
    betas = prs['betas']
    batch_size = prs['batch_size']

    optimizer = torch.optim.Adam(params, lr=lr, betas=betas)

    if prs['dataset_type'] == 'torchvision':
        sparse_data_generator = sparse_data_generator_torchvision
        nb_samples = len(x_data)
    elif prs['dataset_type'] == 'h5':
        sparse_data_generator = sparse_data_generator_h5
        nb_samples = len(x_data['units'])

    loss_hist = []
    fwd_times = []
    bwd_times = []
    fwd_mems = []
    bwd_mems = []
    spike_counts1 = []
    spike_counts2 = []
    active_counts1 = []
    active_counts2 = []
    for e in range(nb_epochs):
        start = time.time()
        for x_local, y_local in sparse_data_generator(x_data, y_data, prs):
            if len(loss_hist) % 25 == 0 and len(loss_hist)>0:
                log_entry = "EPOCH {:d}/{:d}".format(e+1, nb_epochs) + \
                            " Progress:{:d}/{:d}".format(len(loss_hist) % (nb_samples // batch_size), nb_samples // batch_size) + \
                            " LOSS: {:.4f}".format(loss_hist[-1])
                print(log_entry)
                with open(LOG, "a") as log:
                    log.write(log_entry+"\n")

            if len(loss_hist) % prs['time_freq']==0:
                warmup = prs['warmup']   # We do time
            else:
                warmup = 0
            # Run and time
            x_local_dense = x_local.to_dense()
            x_local_dense[x_local_dense[:] > 1.] = 1.
            loss_val, fwd_time, bwd_time, fwd_alloc, bwd_alloc, spks1, spks2, act1, act2 = time_snn(x_local_dense, y_local, params, prs, warmup=warmup)
            # Update gradients
            optimizer.step()
            # Record
            loss_hist.append(loss_val.item())
            fwd_times.append(fwd_time)
            bwd_times.append(bwd_time)
            fwd_mems.append(fwd_alloc)
            bwd_mems.append(bwd_alloc)
            spike_counts1.append(spks1)
            spike_counts2.append(spks2)
            active_counts1.append(act1)
            active_counts2.append(act2)
        print("EPOCH FINISHED in {:.2f} s  FWD: {:.3f}s  BWD: {:.3f}s".format(time.time()-start,
                                                                              np.mean(np.array(fwd_times)),
                                                                              np.mean(np.array(bwd_times))))

    counts = {
        'spike_counts1': spike_counts1,
        'spike_counts2': spike_counts2,
        'active_counts1': active_counts1,
        'active_counts2': active_counts2,
              }
    return loss_hist, fwd_times, bwd_times, fwd_mems, bwd_mems, counts


def run_and_time(prs=None, s3gd=True):
    prs = {} if prs is None else prs
    prs['set_random_state'] = True
    if torch.cuda.is_available():
        prs['device'] = 'cuda'
    assert prs['device'] == 'cuda'

    if s3gd:  # Sparse
        prs['SNN_function'] = my_SparseSNNLayer
    else:  # Original
        from SurrGradSpike import SurrGradSpike
        prs['spike_fn'] = SurrGradSpike.apply
        prs['SNN_function'] = SNNLayerOrig

    fileh_train, fileh_test, x_train, x_test, y_train, y_test, prs, params = init(prs)
    loss_hist, fwd_times, bwd_times, fwd_mems, bwd_mems, counts = train_and_time(x_train, y_train, params, prs)
    train_acc = compute_classification_accuracy(x_train, y_train, params, prs)
    test_acc = compute_classification_accuracy(x_test, y_test, params, prs)
    if fileh_train is not None:
        fileh_train.close()
        fileh_test.close()

    return loss_hist, fwd_times, bwd_times, fwd_mems, bwd_mems, train_acc, test_acc, prs, counts, params



############################################ RUN AND PLOT ############################################
def run(dataset, hidden_list, nb_trials, prs=None):
    global CONFIG, LOG
    CONFIG = CONFIGS[dataset]

    prs = {} if prs is None else prs

    PREFIX = prs.get('PREFIX', CONFIG['PREFIX'])
    SEED = prs.get('seed', CONFIG['seed'])
    BASE_PATH = os.path.dirname(__file__)
    TIME_STRING = time.strftime("_%Y_%m_%d-%H_%M_%S")
    DIR_RESULTS = os.path.join('results',
                               PREFIX + '_' + CONFIG['dataset_id'] + TIME_STRING)
    PATH_RESULTS = os.path.join(BASE_PATH, DIR_RESULTS)
    dst = os.path.join(PATH_RESULTS, os.path.basename(__file__).split('.')[0] + TIME_STRING + '.py')
    if not os.path.isdir(PATH_RESULTS):
        os.makedirs(PATH_RESULTS)
    if not os.path.isfile(dst):
        copyfile(__file__, dst)
    LOG = os.path.join(PATH_RESULTS, "log" + ".txt")
    with open(LOG, "w+") as log:
        log.write('Starting run ... \n')

    # Save results here
    prs_orig = None
    prs_s3gd = None
    loss_orig = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    fwd_orig = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    bwd_orig = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    fwdm_orig = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    bwdm_orig = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    train_acc_orig = np.zeros((nb_trials, len(hidden_list)))
    test_acc_orig = np.zeros((nb_trials, len(hidden_list)))
    loss_s3gd = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    fwd_s3gd = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    bwd_s3gd = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    fwdm_s3gd = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    bwdm_s3gd = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    train_acc_s3gd = np.zeros((nb_trials, len(hidden_list)))
    test_acc_s3gd = np.zeros((nb_trials, len(hidden_list)))
    spike_counts1 = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    spike_counts2 = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    active_counts1 = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]
    active_counts2 = [[[] for _ in range(len(hidden_list))] for _ in range(nb_trials)]

    for trial in range(nb_trials):
        prs['seed'] = SEED + trial
        for n, nb_hidden in enumerate(hidden_list):
            start = time.time()
            prs['nb_hidden'] = nb_hidden
            prs['nb_hidden2'] = nb_hidden
            prs['nb_epochs'] = prs.get('nb_epochs', CONFIG['nb_epochs'])
            prs['warmup'] = 5

            # Sparse spiking backprop
            loss_hist, fwd_times, bwd_times, fwd_mems, bwd_mems, train_acc, test_acc, prs_s3gd, counts, params_s3gd = run_and_time(prs=prs,
                                                                                                                                   s3gd=True)
            loss_s3gd[trial][n] = loss_hist
            fwd_s3gd[trial][n] = fwd_times
            bwd_s3gd[trial][n] = bwd_times
            fwdm_s3gd[trial][n] = fwd_mems
            bwdm_s3gd[trial][n] = bwd_mems
            train_acc_s3gd[trial, n] = train_acc
            test_acc_s3gd[trial, n] = test_acc
            spike_counts1[trial][n] = counts['spike_counts1']
            spike_counts2[trial][n] = counts['spike_counts2']
            active_counts1[trial][n] = counts['active_counts1']
            active_counts2[trial][n] = counts['active_counts2']
            params_s3gd = {'w'+str(l): v for l, v in enumerate(params_s3gd)}
            torch.save(params_s3gd, os.path.join(PATH_RESULTS, 'params_s3gd.p'))

            # Pytorch original
            loss_hist, fwd_times, bwd_times, fwd_mems, bwd_mems, train_acc, test_acc, prs_orig, counts, params_orig = run_and_time(prs=prs,
                                                                                                                                   s3gd=False)
            loss_orig[trial][n] = loss_hist
            fwd_orig[trial][n] = fwd_times
            bwd_orig[trial][n] = bwd_times
            fwdm_orig[trial][n] = fwd_mems
            bwdm_orig[trial][n] = bwd_mems
            train_acc_orig[trial, n] = train_acc
            test_acc_orig[trial, n] = test_acc
            params_orig = {'w'+str(l): v for l, v in enumerate(params_orig)}
            torch.save(params_orig, os.path.join(PATH_RESULTS, 'params_orig.p'))

            # Checkpoint data
            prs_orig_save = {k: v for k, v in prs_orig.items() if k not in ['SNN_function', 'spike_fn', 'dtype']} if prs_orig is not None else None
            prs_s3gd_save = {k: v for k, v in prs_s3gd.items() if k not in ['SNN_function', 'spike_fn', 'dtype']} if prs_s3gd is not None else None
            CONFIG_save = {k: v for k, v in CONFIG.items() if k not in ['dtype']}
            d = {
                'prs_orig': prs_orig_save,
                'prs_s3gd': prs_s3gd_save,
                'CONFIG': CONFIG_save,
                
                'loss_orig': loss_orig,
                'fwd_orig': fwd_orig,                
                'bwd_orig': bwd_orig,
                'fwdm_orig': fwdm_orig,
                'bwdm_orig': bwdm_orig,
                'train_acc_orig': train_acc_orig,             
                'test_acc_orig': test_acc_orig,
                
                'loss_s3gd': loss_s3gd,
                'fwd_s3gd': fwd_s3gd,
                'bwd_s3gd': bwd_s3gd,
                'fwdm_s3gd': fwdm_s3gd,
                'bwdm_s3gd': bwdm_s3gd,
                'train_acc_s3gd': train_acc_s3gd,
                'test_acc_s3gd': test_acc_s3gd,

                'counts': counts,
                 }
            pickle.dump(d, open(os.path.join(PATH_RESULTS, 'data.p'), 'wb'))
            with open(os.path.join(PATH_RESULTS, 'prs.json'), 'w') as fp:
                json.dump([prs_orig_save, prs_s3gd_save, CONFIG_save], fp, sort_keys=True, indent=4)
            print('Checkpoint saved')

            log_entry = "################################################################## \n" + \
                        "TRIAL {:d}/{:d}".format(trial+1, nb_trials) + \
                        " NB_HIDDEN: {:d} of {:d}".format(nb_hidden, len(hidden_list)) + \
                        " in {:.2f} s".format(time.time()-start) + \
                        "\n################################################################## \n"
            print(log_entry)
            with open(LOG, "a") as log:
                log.write(log_entry+"\n")

    # Process data
    loss_orig = np.array(loss_orig)
    fwd_orig = np.array(fwd_orig)
    bwd_orig = np.array(bwd_orig)
    fwdm_orig = np.array(fwdm_orig) / (1024**2)  # MiB
    bwdm_orig = np.array(bwdm_orig) / (1024**2)
    loss_s3gd = np.array(loss_s3gd)
    fwd_s3gd = np.array(fwd_s3gd)
    bwd_s3gd = np.array(bwd_s3gd)
    fwdm_s3gd = np.array(fwdm_s3gd) / (1024**2)
    bwdm_s3gd = np.array(bwdm_s3gd) / (1024**2)
    active_counts1 = np.array(active_counts1)
    active_counts2 = np.array(active_counts2)

    train_acc_orig_mean = np.mean(train_acc_orig, axis=0)
    train_acc_orig_error = stats.sem(train_acc_orig)[0]
    train_acc_s3gd_mean = np.mean(train_acc_s3gd, axis=0)
    train_acc_s3gd_error = stats.sem(train_acc_s3gd)[0]
    test_acc_orig_mean = np.mean(test_acc_orig, axis=0)
    test_acc_orig_error = stats.sem(test_acc_orig)[0]
    test_acc_s3gd_mean = np.mean(test_acc_s3gd, axis=0)
    test_acc_s3gd_error = stats.sem(test_acc_s3gd)[0]


    # Plot
    dataset_name = prs.get('dataset_name', CONFIG['dataset_name'])

    ## Plots per each nb_hidden ##
    for n, nb_hidden in enumerate(hidden_list):
        plt.figure()
        plot_error(loss_orig[:, n, :], label='Original')
        plot_error(loss_s3gd[:, n, :], label='Sparse')
        plt.xlabel("Number of gradient updates")
        plt.ylabel("Loss")
        plt.title("{:s} Loss 2-FClayer [{:d}, {:d}]".format(dataset_name, nb_hidden, nb_hidden))
        plt.legend()
        plt.savefig(os.path.join(PATH_RESULTS, 'loss_h{:d}.pdf'.format(nb_hidden)))

        plt.figure()
        plot_error(fwd_orig[:, n, :], label='Original')
        plot_error(fwd_s3gd[:, n, :], label='Sparse')
        plt.xlabel("Number of gradient updates")
        plt.ylabel("Time (s)")
        plt.title("{:s} Forward time 2-FClayer [{:d}, {:d}]".format(dataset_name, nb_hidden, nb_hidden))
        plt.ylim(bottom=0.)
        plt.legend()
        plt.savefig(os.path.join(PATH_RESULTS, 'fwd_h{:d}.pdf'.format(nb_hidden)))

        plt.figure()
        plot_error(bwd_orig[:, n, :], label='Original')
        plot_error(bwd_s3gd[:, n, :], label='Sparse')
        plt.xlabel("Number of gradient updates")
        plt.ylabel("Time (s)")
        plt.title("{:s} Backward time 2-FClayer [{:d}, {:d}]".format(dataset_name, nb_hidden, nb_hidden))
        plt.ylim(bottom=0.)
        plt.legend()
        plt.savefig(os.path.join(PATH_RESULTS, 'bwd_h{:d}.pdf'.format(nb_hidden)))

        plt.figure()
        plot_error(bwd_orig[:, n, :] / (bwd_s3gd[:, n, :] + 1e-30), label='Backward')
        plot_error((fwd_orig[:, n, :]+bwd_orig[:, n, :]) / (fwd_s3gd[:, n, :] + bwd_s3gd[:, n, :] + 1e-30),
                   label='Overall (backward+forward)')
        plt.xlabel("Number of gradient updates")
        plt.ylabel("Speedup")
        plt.title("{:s} Speedup time 2-FClayer [{:d}, {:d}]".format(dataset_name, nb_hidden, nb_hidden))
        plt.ylim([-0.5, 100])
        plt.legend()
        plt.savefig(os.path.join(PATH_RESULTS, 'speedup_h{:d}.pdf'.format(nb_hidden)))

        plt.figure()
        plot_error(fwdm_orig[:, n, :], label='Original')
        plot_error(fwdm_s3gd[:, n, :], label='Sparse')
        plt.xlabel("Number of gradient updates")
        plt.ylabel("GPU Memory (MiB)")
        plt.title("{:s} Forward Memory 2-FClayer [{:d}, {:d}]".format(dataset_name, nb_hidden, nb_hidden))
        plt.ylim(bottom=0.)
        plt.legend()
        plt.savefig(os.path.join(PATH_RESULTS, 'mem_fwd_h{:d}.pdf'.format(nb_hidden)))

        plt.figure()
        plot_error(bwdm_orig[:, n, :], label='Original')
        plot_error(bwdm_s3gd[:, n, :], label='Sparse')
        plt.xlabel("Number of gradient updates")
        plt.ylabel("GPU Memory (MiB)")
        plt.title("{:s} Backward Memory 2-FClayer [{:d}, {:d}]".format(dataset_name, nb_hidden, nb_hidden))
        plt.ylim(bottom=0.)
        plt.legend()
        plt.savefig(os.path.join(PATH_RESULTS, 'mem_bwd_h{:d}.pdf'.format(nb_hidden)))

        plt.figure()
        plot_error(100*((bwdm_orig[:, n, :] / bwdm_s3gd[:, n, :])-1), label='Backward')
        plot_error(100*(((fwdm_orig[:, n, :]+bwdm_orig[:, n, :]) / (fwdm_s3gd[:, n, :] + bwdm_s3gd[:, n, :]))-1),
                   label='Overall (backward+forward)')
        plt.xlabel("Number of gradient updates")
        plt.ylabel(' GPU Memory Improvement (%) ')
        plt.title("{:s} Memory Improvement 2-FClayer [{:d}, {:d}]".format(dataset_name, nb_hidden, nb_hidden))
        plt.ylim(bottom=0.)
        plt.legend()
        plt.savefig(os.path.join(PATH_RESULTS, 'mem_improvement_h{:d}.pdf'.format(nb_hidden)))

        # Active neuron counts
        total_vars_first_hidden = prs['batch_size'] * prs['nb_steps'] *prs['nb_hidden']
        total_vars_second_hidden = prs['batch_size'] * prs['nb_steps'] *prs['nb_hidden2']
        plt.figure()
        plot_error(100*active_counts1[:, n, :]/total_vars_first_hidden, label='First Hidden Layer')
        plot_error(100*active_counts2[:, n, :]/total_vars_second_hidden, label='Second Hidden Layer')
        plt.xlabel("Number of gradient updates")
        plt.ylabel(' Percentage of active neurons (%) ')
        plt.title("{:s} Total active neurons 2-FClayer [{:d}, {:d}]".format(dataset_name, nb_hidden, nb_hidden))
        plt.ylim(bottom=0.)
        plt.legend()
        plt.savefig(os.path.join(PATH_RESULTS, 'total_active_h{:d}.pdf'.format(nb_hidden)))

    ## Plot accuracies ##
    x = np.arange(len(hidden_list))
    width = 0.35  # the width of the bars

    # Training
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, train_acc_orig_mean, width, yerr=train_acc_orig_error, label='Original')
    rects3 = ax.bar(x + width/2, train_acc_s3gd_mean, width, yerr=train_acc_s3gd_error, label='Sparse')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Number Hidden')
    ax.set_title('Train Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(hidden_list)  # This would be the dataset
    ax.legend()
    plt.savefig(os.path.join(PATH_RESULTS, 'train_acc.pdf'))

    # Testing
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, test_acc_orig_mean, width, yerr=test_acc_orig_error, label='Original')
    rects3 = ax.bar(x + width/2, test_acc_s3gd_mean, width, yerr=test_acc_s3gd_error, label='Sparse')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Number Hidden')
    ax.set_title('Test Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(hidden_list)  # This would be the dataset
    ax.legend()
    plt.savefig(os.path.join(PATH_RESULTS, 'test_acc.pdf'))


    plt.show()
