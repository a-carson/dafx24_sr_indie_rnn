from scipy.signal import windows
import torch
from torch import Tensor as T


def cheb_fft(x: T) -> T:
    win = torch.from_numpy(windows.chebwin(x.shape[-1], at=-120, sym=False))
    return torch.fft.fft(x * win)


def get_harmonics(complex_spec: T, f0: int, sr: int):
    L = complex_spec.shape[0]
    fax = sr * torch.arange(0, L) / L
    freqs = fax[f0: L//2: f0]
    amps = complex_spec[f0: L//2: f0].abs()
    phase = complex_spec[f0: L//2: f0].angle()
    dc_amp = torch.real(complex_spec[0])
    return freqs, amps, phase, dc_amp


def bandlimited_harmonic_signal(freqs: T, amps: T, phase: T, dc_amp: T, t_ax: T, sr: int) -> T:

    x = dc_amp + 2 * torch.sum(
        amps.view(-1, 1) * torch.cos(2 * torch.pi * t_ax * freqs.view(-1, 1) + phase.view(-1, 1)),
        dim=0)

    return x * 2 / sr


def snr_dB(sig: T, noise: T):
    snr = torch.sum(sig**2) / torch.sum(noise**2)
    return 10 * torch.log10(snr)