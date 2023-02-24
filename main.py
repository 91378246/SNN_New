# https://www.frontiersin.org/articles/10.3389/fnins.2021.756876/full#:~:text=SSTDP%3A%20Supervised%20Spike%20Timing%20Dependent%20Plasticity%20for%20Efficient%20Spiking%20Neural%20Network%20Training,-Fangxin%20Liu1&text=Spiking%20Neural%20Networks%20(SNNs)%20are,capability%20and%20high%20biological%20plausibility.
import math

import numpy as np
import torch
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt

# Hyperparameter
T_max = 100
Epochs = 10
Layer_sizes = np.asarray([784, 300, 700, 10])
Threshold = 5
Delta = 5
Strength_factor = 0.8
Restrain_factor = 0.2
W_min = 0
W_max = 50
Learning_rate = 0.005
Tau = 5
Mu = 0.0005

# Globals
Layer_count = len(Layer_sizes)
Layer_out_i = Layer_count - 1
Currents: np.ndarray
"""[t][layer_i][neuron_i] = current"""
Potentials: np.ndarray
"""[t][layer_i][neuron_i] = potential"""
Weights = np.full((Epochs, Layer_count, max(Layer_sizes[0] * Layer_sizes[1], Layer_sizes[1] * Layer_sizes[2], Layer_sizes[2] * Layer_sizes[3])), 0)
"""[epoch][layer_i][synapse_i] = weight"""
Weights[0, :, :] = np.random.uniform(1, 10, Weights[0, :, :].shape)
Weights[0, -1, :] = np.random.uniform(20, 50, Weights[0, -1, :].shape)
Neurons_spike_time = np.zeros((Epochs, Layer_count, np.max(Layer_sizes)))
"""[epoch][layer_i][neuron_i] = 0 || spike_t"""
Loss_gradients = np.full((Epochs, Layer_count, np.max(Layer_sizes)), -1)
"""[epoch][layer_i][neuron_i] = gradient"""
Current_epoch = 0
Losses = list()  # [epoch] = loss


def I(t: int, layer_i: int, neuron_i: int) -> float:
    """
    Calculates the current for the neuron (Eq. 1)
    :param t: Current time step
    :param layer_i: Index of the layer of the neuron to calculate the current for
    :param neuron_i: Index of the neuron to calculate the current for
    :return:
    """
    global Neurons_spike_time
    global Weights
    global Current_epoch

    result = 0
    for neuron_pre_i in range(Layer_sizes[layer_i - 1]):
        if Neurons_spike_time[Current_epoch][layer_i - 1][neuron_pre_i] == t:
            result += Weights[Current_epoch][layer_i - 1][get_synapse_i_between_neurons(layer_i - 1, neuron_pre_i, neuron_i)]

    return result


def V(t: int, layer_i: int, neuron_i: int) -> float:
    """
    Calculates the potential for the neuron (Eq. 2)
    :param t: Current timestep
    :param layer_i: Index of the layer of the neuron to calculate the current for
    :param neuron_i: Index of the neuron to calculate the current for
    :return:
    """
    global Potentials
    global Currents

    return float(Potentials[t - 1][layer_i][neuron_i] + Currents[t][layer_i][neuron_i])


def X(t: int, layer_i: int, neuron_i: int) -> bool:
    """
    Calculates whether the neuron is spiking (Eq. 3)
    :param t: Current timestep
    :param layer_i: Index of the layer of the neuron to calculate the current for
    :param neuron_i: Index of the neuron to calculate the current for
    :return:
    """
    global Neurons_spike_time
    global Current_epoch
    global Potentials

    return Neurons_spike_time[Current_epoch][layer_i][neuron_i] <= 0 \
        and Potentials[t][layer_i][neuron_i] > Threshold  # Error in paper?


def get_total_out_spike_count() -> int:
    global Neurons_spike_time
    global Current_epoch

    spike_count = dict(zip(*np.unique(Neurons_spike_time[Current_epoch][Layer_out_i], return_counts=True)))
    return sum(spike_count.values()) - spike_count[0]


def T_mean(total_out_spike_count: int) -> float:
    """
    Calculates the average firing time for all output neurons (Eq. 4)
    :param total_out_spike_count: The total amount of spikes in the output layer
    :return:
    """
    global Neurons_spike_time
    global Current_epoch

    return float(1 / total_out_spike_count * Neurons_spike_time[Current_epoch][Layer_out_i].sum())


def T(T_mean: float, total_out_spike_count: int, y: np.ndarray) -> list:
    """
    Calculates the expected firing times for the output neurons (Eq. 5)
    :param T_mean: Average firing time for all output neurons
    :param total_out_spike_count: The total amount of spikes in the output layer
    :param y: The expected label
    :return:
    """
    global Layer_sizes
    global Neurons_spike_time
    global Current_epoch
    global Delta

    expected_firing_times = list()
    for out_neuron_i in range(Layer_sizes[Layer_out_i]):
        spike_time = float(Neurons_spike_time[Current_epoch][Layer_out_i][out_neuron_i].max())

        if out_neuron_i == y:
            expected_firing_times.append(min(spike_time, T_mean - ((total_out_spike_count - 1) / total_out_spike_count) * Delta))
        else:
            expected_firing_times.append(max(spike_time, T_mean + (1 / total_out_spike_count) * Delta))

    return expected_firing_times


def get_spike_time(layer_i: int, neuron_i: int) -> int:
    """
    Returns the spike time for the given neuron
    :param layer_i:
    :param neuron_i:
    :return:
    """
    global Neurons_spike_time
    global Current_epoch

    return int(Neurons_spike_time[Current_epoch][layer_i][neuron_i].max())


def E(T: list) -> float:
    """
    Loss function (Eq. 6)
    :param T: Expected firing time
    :return:
    """
    global Layer_sizes

    result = 0
    for neuron_out_i in range(Layer_sizes[-1]):
        result += (get_spike_time(Layer_out_i, neuron_out_i) - T[neuron_out_i]) ** 2

    return 0.5 * result


def dE_dw(layer_i: int, neuron_i: int, neuron_pre_i: int) -> float:
    """
    Eq. 8
    :param layer_i:
    :param neuron_i:
    :param neuron_pre_i:
    :return:
    """
    global Loss_gradients
    global Current_epoch

    return Loss_gradients[Current_epoch][layer_i][neuron_i] * dt_dw(layer_i, neuron_i, neuron_pre_i)


def dt_dw(layer_i: int, neuron_i: int, neuron_pre_i: int) -> float:
    """
    Derivative between post-synaptic firing time and the weight (Eq. 9)
    :param layer_i:
    :param neuron_i:
    :param neuron_pre_i:
    :return:
    """
    global Strength_factor
    global Restrain_factor
    global Weights
    global Current_epoch

    ts_pre = get_spike_time(layer_i - 1, neuron_pre_i)
    ts_this = get_spike_time(layer_i, neuron_i)

    scaler = Strength_factor if ts_this > ts_pre else Restrain_factor
    t_dif = ts_this - ts_pre if ts_this > ts_pre else ts_pre - ts_this

    return scaler \
        * (math.e ** (-(t_dif / Tau)) - Delta) \
        * (W_max - Weights[Current_epoch][layer_i - 1][get_synapse_i_between_neurons(layer_i - 1, neuron_pre_i, neuron_i)]) ** Mu


def dE_dt(layer_i: int, neuron_i: int) -> float:
    """
    Firing time gradient (Eq. 10)
    :param layer_i:
    :param neuron_i:
    :return:
    """
    global Layer_sizes
    global Loss_gradients
    global Current_epoch

    result = 0
    for neuron_pre_i in range(Layer_sizes[layer_i - 1]):
        result += Loss_gradients[Current_epoch][layer_i - 1][neuron_pre_i] * dt_dt(layer_i, neuron_i, neuron_pre_i)

    return result


def get_synapse_i_between_neurons(neuron_pre_layer_i: int, neuron_pre_i: int, neuron_post_i: int) -> int:
    global Layer_sizes

    return neuron_pre_i * Layer_sizes[neuron_pre_layer_i + 1] + neuron_post_i


def dt_dt(layer_i: int, neuron_i: int, neuron_pre_i: int) -> float:
    """
    Firing time derivative (Eq. 11)
    :param layer_i: Current layer
    :param neuron_i: Current neuron to calculate the time derivative for
    :param neuron_pre_i: The pre neuron to use for the calculation
    :return:
    """
    global Weights
    global Current_epoch

    ts_pre = get_spike_time(layer_i - 1, neuron_pre_i)
    ts_this = get_spike_time(layer_i, neuron_i)

    if ts_pre > ts_this:
        return 0
    else:
        return Weights[Current_epoch][layer_i - 1][get_synapse_i_between_neurons(layer_i - 1, neuron_pre_i, neuron_i)]


def get_prediction() -> int:
    global Layer_sizes
    global Layer_out_i

    result = get_spike_time(Layer_out_i, 0)
    for neuron_out_i in range(1, Layer_sizes[-1]):
        ts = get_spike_time(Layer_out_i, neuron_out_i)
        if ts < result:
            result = ts

    return result


def soft_reset() -> None:
    global Layer_sizes
    global Layer_count
    global Epochs

    global Currents
    Currents = np.empty((T_max, Layer_count, np.max(Layer_sizes)))
    global Potentials
    Potentials = np.full((T_max, Layer_count, np.max(Layer_sizes)), 0)


def visualize_forward() -> None:
    global Currents
    global Potentials
    global Neurons_spike_time
    global Current_epoch

    for layer_i in range(Layer_count):
        plt.plot(np.average(Currents[:, layer_i, :Layer_sizes[layer_i]], axis=1), label=f"Layer {layer_i}")
    plt.title("Currents")
    plt.xlabel("t")
    plt.legend()
    plt.show()

    for layer_i in range(Layer_count):
        plt.plot(np.average(Potentials[:, layer_i, :Layer_sizes[layer_i]], axis=1), label=f"Layer {layer_i}")
    plt.title("Potentials")
    plt.xlabel("t")
    plt.legend()
    plt.show()

    for layer_i in range(Layer_count):
        plt.plot(Neurons_spike_time[Current_epoch][layer_i][:Layer_sizes[layer_i]])
        plt.title(f"Spike times layer {layer_i}")
        plt.xlabel("Neuron")
        plt.show()


def visualize_backward() -> None:
    global Losses
    global Weights
    global Loss_gradients
    global Current_epoch

    plt.plot(Losses)
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.show()

    for layer_i in range(Layer_count):
        plt.plot(np.average(Weights[:, layer_i, :Layer_sizes[layer_i]], axis=1), label=f"Layer {layer_i}")
    plt.title("Weights")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()

    for layer_i in range(Layer_count):
        plt.plot(np.average(Loss_gradients[:, layer_i, :Layer_sizes[layer_i]], axis=1), label=f"Layer {layer_i}")
    plt.title("Loss Gradients")
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()


def main():
    global Current_epoch
    global T_max
    global Neurons_spike_time
    global Layer_count
    global Layer_sizes
    global Learning_rate
    global Currents
    global Potentials
    global Weights
    global Loss_gradients

    train_dataset = torchvision.datasets.MNIST(
        root="../mnist",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    test_dataset = torchvision.datasets.MNIST(
        root="../mnist",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True)

    Current_epoch = 0
    for img, label in train_data_loader:
        print(f"EPOCH {Current_epoch}")
        soft_reset()

        # Input encoding
        encoded = ((1 - img) * (T_max - 1)).round().int()

        # Set input neurons
        for input_i, input in enumerate(encoded.numpy().reshape(-1)):
            Neurons_spike_time[Current_epoch][0][input_i] = input

        # Forward pass
        for t in tqdm(range(1, T_max), desc="Doing forward pass"):
            for layer_i in range(1, Layer_count):
                for neuron_i in range(Layer_sizes[layer_i]):
                    if Neurons_spike_time[Current_epoch][layer_i][neuron_i] > 0:
                        continue

                    Currents[t][layer_i][neuron_i] = I(t, layer_i, neuron_i)
                    Potentials[t][layer_i][neuron_i] = V(t, layer_i, neuron_i)

                    # Check for spike
                    if X(t, layer_i, neuron_i):
                        Neurons_spike_time[Current_epoch][layer_i][neuron_i] = t

        for layer_i in range(1, Layer_count):
            if np.max(Neurons_spike_time[Current_epoch][layer_i]) == 0:
                print(f"WARNING: Layer {layer_i} hasn't produced any spikes")

        visualize_forward()

        # Backward pass
        # total_out_spike_count = n
        total_out_spike_count = get_total_out_spike_count()
        # avg_firing_time = T_mean
        avg_firing_time = T_mean(total_out_spike_count)
        # expected_firing_times[neuron_i] = T_j^L
        expected_firing_times = T(avg_firing_time, total_out_spike_count, label)
        # loss_gradients = dE/dt_j^L
        # weight_gradients = dt_j^l/dw_ij^l, [layer_i][synapse_i] = gradient
        weight_gradients = np.empty_like(Weights[Current_epoch])

        Losses.append(E(expected_firing_times))
        print(f"\nPredicted: {get_prediction()} | Actual: {int(label)} | Loss: {Losses[-1]}")

        for layer_i in tqdm(range(Layer_count - 1, 0, -1), desc="Doing backward pass"):
            # Synapses between last hidden and output
            if layer_i == Layer_count - 1:
                for neuron_i in range(Layer_sizes[layer_i]):
                    # spike_time = t_j^L
                    spike_time = get_spike_time(layer_i, neuron_i)
                    Loss_gradients[Current_epoch][layer_i][neuron_i] = expected_firing_times[neuron_i] - spike_time
            # Hidden synapses
            else:
                for neuron_i in range(Layer_sizes[layer_i]):
                    Loss_gradients[Current_epoch][layer_i][neuron_i] = dE_dt(layer_i, neuron_i)

            for neuron_i in range(Layer_sizes[layer_i]):
                for neuron_pre_i in range(Layer_sizes[layer_i - 1]):
                    synapse_i = get_synapse_i_between_neurons(layer_i - 1, neuron_pre_i, neuron_i)
                    weight_gradients[layer_i][synapse_i] = dE_dw(layer_i, neuron_i, neuron_pre_i)
                    Weights[Current_epoch][layer_i][synapse_i] -= Learning_rate * weight_gradients[layer_i][synapse_i]

        Current_epoch += 1
        visualize_backward()

        if Current_epoch == Epochs:
            break

    print("Done")


if __name__ == "__main__":
    main()
