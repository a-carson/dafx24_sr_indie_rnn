from rnn import get_SRIndieRNN, get_AudioRNN_from_json
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import windows
import librosa
import os

#mpl.use("macosx")

def chebwin_fft(x, N=None, at=-120):
    if N is None:
        N = x.shape[-1]
    win = windows.chebwin(N, at=at, sym=False)
    return np.fft.fft(x * win, n=N)

def get_harmonics(Y, f0, Fs):
    L = len(Y)
    fax = Fs * np.arange(0, len(Y)) / len(Y)
    harmonic_freqs = fax[f0: L//2: f0]
    harmonic_amps = np.abs(Y[f0: L//2: f0])
    harmonic_phase = np.angle(Y[f0: L//2: f0])
    dc_bin = np.real(Y[0])
    return harmonic_freqs, harmonic_amps, harmonic_phase, dc_bin

def bandlimited_harmonic_signal(freqs, amps, phase, DC_amp, t_ax, Fs):

    fourier_synth = DC_amp + 2 * torch.sum(
        torch.Tensor(amps).view(-1, 1) * torch.cos(
            2 * torch.pi * torch.Tensor(t_ax) * torch.Tensor(freqs).view(-1, 1) + torch.Tensor(phase).view(
                -1, 1)
        ), dim=0)

    fourier_synth *= 2 / Fs

    return fourier_synth.numpy()



filenames = ['MesaMiniRec_HighGain_DirectOut.json']
base_path = '../../../Proteus_Tone_Packs/Selection'
os_factors = [48/44.1]

# SETTINGS
methods = ['naive', 'stn', 'lidl', 'apdl', 'lagrange']
dur_seconds = 1.0
start_seconds = 0.0
sample_rate_base = 44100
gain = 0.1

midi = np.arange(100, 109)
f0_freqs = np.floor(440 * 2 ** ((midi - 69) / 12))

snr_aliases = np.zeros((len(f0_freqs), len(methods), len(filenames), len(os_factors)))
snr_harmonics = np.zeros((len(f0_freqs), len(methods), len(filenames), len(os_factors)))
thd = np.zeros((len(f0_freqs), len(filenames)))

for o, os_factor in enumerate(os_factors):
    print(os_factor)

    for f, filename in enumerate(filenames):
        print(filename)

        for m, method in enumerate(methods):
            model = get_SRIndieRNN(base_model=get_AudioRNN_from_json(os.path.join(base_path, filename)),
                                   method=method)

            model.eval()
            with torch.no_grad():

                L = int(np.ceil(sample_rate_base * dur_seconds))
                t_ax = np.arange(0, L) / sample_rate_base
                x = gain * torch.sin(2.0 * torch.Tensor(f0_freqs).view(-1, 1, 1) * torch.pi * torch.Tensor(t_ax).view(1, -1, 1))
                model.reset_state()
                model.rec.os_factor = 1
                y_base, _ = model.forward(x)
                y_base = y_base[:, :, 0].detach().numpy()
                Y_base = chebwin_fft(y_base)
                f_ax_base = np.arange(0, L) / L * sample_rate_base
                a_weight_base = 10 ** (librosa.A_weighting(f_ax_base) / 10)

                sample_rate = np.round(sample_rate_base * os_factor)

                L_up = int(np.ceil(sample_rate * dur_seconds))
                t_ax_up = np.arange(0, L_up) / sample_rate
                x = gain * torch.sin(2.0 * torch.Tensor(f0_freqs).view(-1, 1, 1) * torch.pi * torch.Tensor(t_ax_up).view(1, -1, 1))
                model.reset_state()
                model.rec.os_factor = os_factor
                y, _ = model.forward(x)
                y = y[:, :, 0].detach().numpy()

                Y = chebwin_fft(y)
                f_ax_up = np.arange(0, L_up) / L_up * sample_rate
                a_weight_up = 10 ** (librosa.A_weighting(f_ax_up) / 10)

                for i, f0 in enumerate(f0_freqs):
                    f0 = int(f0)
                    freqs, amps, phase, DC = get_harmonics(Y[i, ...], f0, sample_rate)
                    freqs_base, amps_base, phase_base, DC_base = get_harmonics(Y_base[i, ...], f0, sample_rate_base)
                    y_base_bl = bandlimited_harmonic_signal(freqs_base, amps_base, phase_base, DC_base, t_ax, sample_rate_base)

                    M = len(freqs_base)
                    y_down_bl = bandlimited_harmonic_signal(freqs[:M], amps[:M], phase[:M], DC, t_ax, sample_rate)

                    # NEW -- aliasing
                    Y_bl = np.abs(chebwin_fft(y_down_bl))
                    Y_bl *= np.abs(Y[i, f0]) / np.abs(Y_bl[f0])
                    aliases = Y_bl[:L//2] - np.abs(Y[i, :L//2])
                    Y_bl = Y_bl[:L//2]

                    snr = 10 * np.log10(
                        np.sum(Y_bl ** 2) / (np.sum(aliases ** 2))
                    )

                    esr = 10 * np.log10(
                        np.sum(y_base_bl ** 2) / (np.sum((y_down_bl - y_base_bl) ** 2))
                    )
                    snr_aliases[i, m, f, o] = snr
                    snr_harmonics[i, m, f, o] = esr
                    thd[i, f] = np.sqrt(np.sum(amps_base[1:]**2)) / amps_base[0]



    plt.figure(figsize=[10, 5])
    plt.semilogx(f0_freqs, np.mean(snr_aliases[..., o], axis=-1), marker='+')
    plt.title('Aliasing SNR -- OS = {}'.format(os_factor))
    plt.xlabel('f0 [Hz]'), plt.ylabel('dB')
    plt.legend(methods)

    plt.figure(figsize=[10, 5])
    plt.semilogx(f0_freqs, np.mean(snr_harmonics[..., o], axis=-1), marker='+')
    plt.title('OS = {}'.format(os_factor))
    plt.xlabel('f0 [Hz]'), plt.ylabel('SNHR [dB]')
    plt.legend(methods)
    plt.show()

#np.save('snr_aliases_mesamini.npy', snr_aliases)
#np.save('snr_harmonics_mesamini.npy', snr_harmonics)