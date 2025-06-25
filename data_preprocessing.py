import numpy as np
import pickle
import torch
import scipy.stats.qmc as qmc

torch.set_default_dtype(torch.float64)

def normalize(x, minimum=None, maximum=None):
    if minimum is None:
        minimum = x.min()
    if maximum is None:
        maximum = x.max()
    return (2 * (x - minimum) / (maximum - minimum)) - 1

if __name__ == '__main__':
    with open('data/data_sps_4D.pkl', 'rb') as f:
        data = pickle.load(f)

    tensor_data = torch.tensor(np.array((data['x'], data['px'], data['y'], data['py'])))
    tensor_data = tensor_data.view(tensor_data.shape[1], tensor_data.shape[0], tensor_data.shape[2])

    # create time data
    n_turns = tensor_data.shape[2]
    
    t = torch.linspace(0, n_turns-1, n_turns)
    minimum_t = t.min()
    maximum_t = t.max()
    t = normalize(t, minimum_t, maximum_t)

    # Random sampling of training set
    idx_positions = np.random.binomial(size=tensor_data.shape[0], n=1, p=0.1) != 0
    idx_time = np.random.binomial(size=tensor_data.shape[2], n=1, p=0.1) != 0
    idx_time[0] = True
    idx_time[-1] = True

    train_set = tensor_data[idx_positions, :, :]
    train_set = train_set[:, :, idx_time]

    test_set_pos = tensor_data[~idx_positions, :, :]
    test_set_time = tensor_data[idx_positions, :, :]

    for dim in range(train_set.shape[1]):
        minimum = train_set[:, dim, 0].min()
        maximum = train_set[:, dim, 0].max()

        train_set[:, dim, :] = normalize(train_set[:, dim, :], minimum, maximum)
        test_set_pos[:, dim, :] = normalize(test_set_pos[:, dim, :], minimum, maximum)
        test_set_time[:, dim, :] = normalize(test_set_time[:, dim, :], minimum, maximum)

    def reshape_data(data, time):
        inputs = data[:, :, 0].repeat(data.shape[2], 1)
        time = time.repeat_interleave(data.shape[0]).unsqueeze(1)

        x = data[:, 0, :]
        px = data[:, 1, :]
        y = data[:, 2, :]
        py = data[:, 3, :]
        targets = torch.stack((torch.flatten(x.T), torch.flatten(px.T), torch.flatten(y.T), torch.flatten(py.T)), dim=1)
        

        # Fourier expansion
        time = torch.cat((time, torch.cos(time*torch.pi), torch.cos(time*2*torch.pi), 
                torch.cos(time*3*torch.pi), torch.sin(time*torch.pi), 
                torch.sin(time*2*torch.pi), torch.sin(time*3*torch.pi)), dim=1)

        inputs = torch.cat((inputs, time), dim=1)

        return inputs, targets

    train_inputs, train_targets = reshape_data(train_set, t[idx_time])
    test_pos_inputs, test_pos_targets = reshape_data(test_set_pos, t)
    test_time_inputs, test_time_targets = reshape_data(test_set_time, t)

    torch.save((train_inputs, train_targets), 'data/training_data.pt')
    torch.save((test_pos_inputs, test_pos_targets), 'data/test_pos_data.pt')
    torch.save((test_time_inputs, test_time_targets), 'data/test_time_data.pt')
