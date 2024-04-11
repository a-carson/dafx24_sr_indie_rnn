import time
from rnn import get_SRIndieRNN, get_AudioRNN_from_json
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import math
from utils import cheb_fft, bandlimited_harmonic_signal, get_harmonics, snr_dB


# SETTINGS
filenames = ['MesaMiniRec_HighGain_DirectOut.json']
base_path = '../../../Proteus_Tone_Packs/Selection'
methods = ['naive', 'stn', 'lidl', 'apdl', 'cidl', 'lagrange']
os_factors = np.array([48/44.1], dtype=np.double)
dur_seconds = 1.0
start_seconds = 0.0
sr_base = 44100
gain = 0.1
dtype = torch.double
midi = torch.arange(21, 109, dtype=dtype)

# init
f0_freqs = torch.floor(440 * 2 ** ((midi - 69) / 12))       # need int values
snr_aliases = np.zeros((len(f0_freqs), len(methods), len(filenames), len(os_factors)))
snr_harmonics = np.zeros((len(f0_freqs), len(methods), len(filenames), len(os_factors)))
thd = np.zeros((len(f0_freqs), len(filenames)))

start_time = time.time()

# base time and input -------
t_ax = torch.arange(0, math.ceil(sr_base * dur_seconds), dtype=dtype) / sr_base
x = gain * torch.sin(2.0 * f0_freqs.view(-1, 1, 1) * torch.pi * t_ax.view(1, -1, 1))
num_samples = math.ceil(sr_base * dur_seconds)


for o, os_factor in enumerate(os_factors):

    print(os_factor)
    # oversampled time and input ---------
    sr = math.ceil(sr_base * os_factor)
    num_samples_up = math.ceil(sr * dur_seconds)
    t_ax_up = torch.arange(0, num_samples_up) / sr
    x_up = gain * torch.sin(2.0 * f0_freqs.view(-1, 1, 1) * torch.pi * t_ax_up.view(1, -1, 1))

    for f, filename in enumerate(filenames):
        print(filename)
        # process baseline output --------------------------
        base_model = get_AudioRNN_from_json(os.path.join(base_path, filename))
        base_model.double()
        base_model.reset_state()
        base_model.eval()
        with torch.no_grad():
            y_base, _ = base_model.forward(x)
            y_base = y_base[:, :, 0].detach()
            Y_base = cheb_fft(y_base)

        for m, method in enumerate(methods):

            # process modified model output ------------------
            model = get_SRIndieRNN(base_model=base_model, method=method)
            model.eval()
            model.double()
            model.reset_state()
            with torch.no_grad():

                model.reset_state()
                model.rec.os_factor = os_factor
                y, _ = model.forward(x_up)
                y = y[:, :, 0].detach()
                Y = cheb_fft(y)

                for i, f0 in enumerate(f0_freqs):
                    f0 = int(f0)

                    # baseline bl harmonic sig
                    freqs_base, amps_base, phase_base, dc_base = get_harmonics(Y_base[i, ...], f0, sr_base)
                    y_base_bl = bandlimited_harmonic_signal(freqs_base, amps_base, phase_base, dc_base, t_ax, sr_base)

                    # model bl harmonic sig
                    freqs, amps, phase, dc = get_harmonics(Y[i, ...], f0, sr)
                    num_harmonics = len(freqs_base)
                    y_bl = bandlimited_harmonic_signal(freqs[:num_harmonics],
                                                       amps[:num_harmonics],
                                                       phase[:num_harmonics],
                                                        dc, t_ax, sr)

                    # adjust to ensure equal first harmonic amps
                    Y_bl = cheb_fft(y_bl)
                    Y_bl *= Y[i, f0].abs() / Y_bl[f0].abs()

                    # compute SNRs
                    snra = snr_dB(sig=Y_bl[:num_samples//2].abs(),
                                  noise=Y_bl[:num_samples//2].abs() - Y[i, :num_samples//2].abs())
                    snr_aliases[i, m, f, o] = snra.numpy()

                    snrh = snr_dB(sig=y_base_bl,
                                  noise=y_bl - y_base_bl)
                    snr_harmonics[i, m, f, o] = snrh.numpy()


    plt.figure(figsize=[10, 5])
    plt.semilogx(f0_freqs, np.mean(snr_aliases[..., o], axis=-1), marker='+')
    plt.title('Aliasing SNR -- OS = {}'.format(os_factor))
    plt.xlabel('f0 [Hz]'), plt.ylabel('dB')
    plt.legend(methods)

    plt.figure(figsize=[10, 5])
    plt.semilogx(f0_freqs.numpy(), np.mean(snr_harmonics[..., o], axis=-1), marker='+')
    plt.title('OS = {}'.format(os_factor))
    plt.xlabel('f0 [Hz]'), plt.ylabel('SNHR [dB]')
    plt.legend(methods)
    plt.show()
print("elapsed time: ", time.time() - start_time)
#np.save('snr_aliases_mesamini.npy', snr_aliases)
#np.save('snr_harmonics_mesamini.npy', snr_harmonics)