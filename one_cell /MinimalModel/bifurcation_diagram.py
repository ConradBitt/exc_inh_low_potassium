from matplotlib import pyplot as plt
from numba import jit, float64

import os
import numpy as np
import pickle
import matplotlib.colors as mcolors
from scipy.signal import find_peaks

import datetime


cores = list(mcolors.TABLEAU_COLORS.keys())
cores = [cor.split(':')[-1] for cor in cores]


def plot_params():
    import locale
    import latex

    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')  
    plt.rc('text', usetex=True)
    plt.rc('font', size=13)
    plt.rc('xtick', labelsize=11)
    plt.rc('ytick', labelsize=11)
    plt.rc('axes', labelsize=14)
    plt.rc('legend', fontsize=8)
    plt.rc('lines', linewidth=1.0)
    plt.rcParams["axes.formatter.limits"] = (-3, 4)
    plt.rcParams['axes.formatter.use_locale'] = True
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    plt.rcParams['pcolor.shading'] = 'nearest'
plot_params()

# @jit(nopython=True)
def isi_cv_freq(tpeaks):
    """
    Calcula o inter-spike-interval (ISI) de uma lista de tspikes.

    Args:
        tpeaks (list of arrays): A lista contendo n-arrays com o tempo em que ocorre os spikes.

    Returns:
        tupla: uma tupla contendo as seguintes matrizes:
             - isi_bar (numpy.ndarray): Matriz de ISIs médios para cada neurônio.
             - cv (numpy.ndarray): Matriz de coeficiente de variação (CV) dos ISIs para cada neurônio.
             - freq_bar (numpy.ndarray): Matriz de frequências médias de disparo (em Hz) para cada neurônio.
    """
    num_neurons = len(tpeaks)
    isi_bar = np.zeros(num_neurons)
    cv = np.zeros(num_neurons)
    freq_bar = np.zeros(num_neurons)
    
    for i in range(num_neurons):
        isis = np.empty(len(tpeaks[i]) - 1, dtype=np.float64)
        
        for j in range(len(tpeaks[i]) - 1):
            isis[j] = tpeaks[i][j + 1] - tpeaks[i][j]


        # # filter 
        # for spks in range(11):
        #     if len(isis) == spks:
        #         isi_bar[i] = spks
        #         freq_bar[i] = spks
        #         cv[i] = spks
        # if len(isis) > 10:
        indice = int((3/4)*len(isis))

        isi_bar[i] = np.mean(isis[-indice:])
        freq_bar[i] = (1 / isi_bar[i]) * 1e3  # Convert ISI to firing frequency (Hz)
        cv[i] = np.std(isis[-indice:]) / isi_bar[i]
        
    return isi_bar, cv, freq_bar


def find_peaks_cfb(t_arr, v_arr, only_id=False):
    """
    Encontra os picos em um sinal de forma de onda.

    Args:
        t_arr (array-like): Uma matriz de tempos correspondentes aos valores do sinal de forma de onda.
        v_arr (array-like): Uma matriz de valores do sinal de forma de onda.
        only_id (bool, optional): Indica se apenas os IDs dos picos devem ser retornados. O padrão é False.

    Returns:
        tuple or numpy.ndarray: Se only_id for False, retorna uma tupla contendo os IDs dos picos, tempos correspondentes e valores correspondentes. Se only_id for True, retorna apenas os IDs dos picos.
    """
    peaks_id, _ = find_peaks(v_arr, height=0)
    t = t_arr[peaks_id]
    v = v_arr[peaks_id]
    if only_id:
        peaks_id
    else:
        return peaks_id, t, v
        


gm_values = np.linspace(4,8,50) * 1e-5
volts_forward = np.zeros_like(gm_values)
freqs_forward = np.zeros_like(gm_values)
isi_bars_forward = np.zeros_like(gm_values)
cvs_forward = np.zeros_like(gm_values)

volts_backward = np.zeros_like(gm_values)
freqs_backward = np.zeros_like(gm_values)
isi_bars_backward = np.zeros_like(gm_values)
cvs_backward = np.zeros_like(gm_values)


tf = 2000
amp = 0.170
dur = tf
delay = 0
v_init = -84
nome = 'yamada'


inicio = datetime.datetime.now()
for i, gm in enumerate(gm_values):
    inicio_for = datetime.datetime.now()
    nome = 'yamada_forward'
    call = f'python3 createCell.py {gm} {tf} {amp} {dur} {delay} {v_init} {nome}'

    print(call)
    os.system(call)

    with open(f'cell{nome}.pkl', 'rb') as f:
        data_tmp = pickle.load(f)
    
    time = np.array(data_tmp["time"].to_python())
    voltage_f = np.array(data_tmp["voltage"].to_python())

    peaks_id, t, v = find_peaks_cfb(time, voltage_f)
    isi_bar, cv, freq_bar = isi_cv_freq([t])

    volts_forward[i] = voltage_f[-1]
    freqs_forward[i] = freq_bar
    isi_bars_forward[i] = isi_bar
    cvs_forward[i] = cv

    nome = 'yamada_backward'
    os.system(f'python3 createCell.py {gm} {tf} {amp} {dur} {delay} {volts_forward[i]} {nome}')

    with open(f'cell{nome}.pkl', 'rb') as f:
        data_tmp = pickle.load(f)
    
    time = np.array(data_tmp["time"].to_python())
    voltage_b = np.array(data_tmp["voltage"].to_python())

    peaks_id, t, v = find_peaks_cfb(time, voltage_b)
    isi_bar, cv, freq_bar = isi_cv_freq([t])

    volts_backward[i] = voltage_b[-1]
    freqs_backward[i] = freq_bar
    isi_bars_backward[i] = isi_bar
    cvs_backward[i] = cv
    fim_for = datetime.datetime.now()

    print('\n\n')
    print(20*'--')
    print(f'Tempo decorrido: {fim_for - inicio} \t Tempo por loop: {fim_for - inicio_for}')
    print(20*'--')

fim = datetime.datetime.now()

print(20*'=--=')
print(10*' ' + f'Total time running simulation: {fim - inicio}')
print(20*'----')
print('\n\n')

data = {
    'im' : gm_values,
    'voltages_f' : volts_forward,
    'freqs_f':freqs_forward,
    'isi_bars_f':isi_bars_forward,
    'cvs_f':cvs_forward,
    'voltages_b': volts_backward,
    'freqs_b':freqs_backward,
    'isi_bars_b':isi_bars_backward,
    'cvs_b':cvs_backward,
    'time_simulation':fim - inicio
}

with open('bifurcation_data.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.plot(gm_values, freqs_forward, alpha=0.1)
plt.scatter(gm_values, freqs_forward, color='blue',label='forward', s=5)

plt.scatter(gm_values, freqs_backward, color='red' ,label='backward', s=5)
plt.plot(gm_values, freqs_backward, alpha=0.1)

plt.xlabel('$g_m (S/mm^2)$')
plt.ylabel('Fr(Hz)')

plt.savefig('bifurcation_diagram_fr_im.png')