# https://thesai.org/Downloads/IJARAI/Volume4No7/Paper_1-A_Minimal_Spiking_Neural_Network_to_Rapidly_Train.pdf
import math
import numpy as np

tau: float = 2  # Peak conductance time

a: float = 0.03  # Timescale of u
b: float = -2  # Sensitivity of u
c: float = -50  # Reset membrane potential
d: float = 100  #
V_rest: float = -60  # Default potential
V_threshold: float = -40  # Threshold
V_peak: float = 30  # Max membrane potential
U_0: float = 0
I_inj: float = 0

N_in = 28
N_hidden = 28
N_out = 7
neuron_vs = np.full((2, max(N_hidden, N_out)), V_rest)  # [layer_i][neuron_i] = potential
spike_times = [list(), list(), list()]  # [layer_i][neuron_i] = spike_t


def G_syn(t: int, K_syn: float) -> float:
    """
    Postsynaptic conductance
    :param t: Timestep
    :param K_syn: Peak conductance value
    :return:
    """
    return K_syn * t * math.exp(-t / tau)


def G_tot(t: int, N: int, N_rec: np.ndarray, t_f: np.ndarray, K_syn: np.ndarray) -> float:
    """
    Total conductance
    :param t: Timestep
    :param N: Input synapse count
    :param N_rec: [synapse_i] = spike_count, Spike count for each input synapse
    :param t_f: [synapse_i][spike_i] = spike_time, Spike time for each spike for each input synapse
    :param K_syn: [synapse_i] = K_syn, Peak conductance value for each input synapse
    :return:
    """
    result = 0

    # For each input synapse k
    for k in range(N):
        # For each spike j from k
        for j in range(N_rec[k]):
            result += K_syn[k] * (t - t_f[k][j]) * math.exp(-(t - t_f[k][j]) / tau)

    return result


def I_syn(t: int, N: int, N_rec: np.ndarray, t_f: np.ndarray, K_syn: np.ndarray) -> float:
    sum_0 = 0
    sum_1 = 0

    # For each input synapse k
    for k in range(N):
        g_tot = G_tot(t, N, N_rec, t_f, K_syn)
        sum_0 += E_syn(t) * g_tot
        sum_1 += g_tot

    return sum_0 - V(t) * sum_1


def E_syn(t: int) -> float:
    pass


# https://www.seti.net/Neuron%20Lab/MatLab/izhikevich/Spinking%20Neuron%20GUI/Izh_spikes.pdf
def V(v: float, u: float, I: float) -> float:
    """
    Membrane potential
    :param v: Current membrane potential
    :param u: Membrane recovery cariable
    :param I: Synaptic currents
    :return:
    """
    return 0.04 * v ** 2 + 5 * v + 140 - u + I


# https://www.seti.net/Neuron%20Lab/MatLab/izhikevich/Spinking%20Neuron%20GUI/Izh_spikes.pdf
def U(v: float, u: float) -> float:
    """
    Membrane recovery factor
    :param v:
    :param u:
    :return:
    """
    return a * (b * v - u)



def main():
    pass


if __name__ == "__main__":
    main()
