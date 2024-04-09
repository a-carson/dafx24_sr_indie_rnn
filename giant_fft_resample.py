import torch
from torch import tensor as T

'''
Giant FFT resampling

Based on method proposed in:
Vesa Välimäki and Stefan Bilbao, "Giant FFTs for Sample-Rate Conversion" in Journal Audio Eng. Soc. (JAES), 2023
https://www.aes.org/e-lib/browse.cfm?elib=22033

'''
def giant_fft_resample(x: T, orig_freq: int, new_freq: int):
    if orig_freq > new_freq:
        return _giant_fft_upsample(x, orig_freq, new_freq)
    else:
        return _giant_fft_downsample(x, orig_freq, new_freq)


def _giant_fft_upsample(x: T, orig_freq: int, new_freq: int):

    N = x.shape[-1]
    X = torch.fft.fft(x)
    N_up = new_freq * N // orig_freq
    X_up = torch.zeros((1, N_up), dtype=X.dtype)

    X_up[..., 0:N//2] = X[..., 0:N//2].clone()
    X_up[..., N//2] = 0.5 * X[..., N//2].clone()
    X_up[..., N_up - N//2] = 0.5 * X[..., N//2].clone()
    X_up[..., N_up - N // 2 + 1:] = torch.conj(X_up[..., 1:N//2].clone()).flip(-1)
    x_up = torch.fft.ifft(X_up)
    return torch.real(x_up) * new_freq / orig_freq


def _giant_fft_downsample(x: T, orig_freq: int, new_freq: int):

    N = x.shape[-1]
    X = torch.fft.fft(x)
    N_down = new_freq * N // orig_freq
    X_down = torch.zeros((1, N_down), dtype=X.dtype)

    X_down[..., 0:N_down//2] = X[..., 0:N_down//2].clone()
    X_down[..., N_down//2] = 0.0
    X_down[..., N_down // 2 + 1:] = torch.conj(X_down[..., 1:N_down//2].clone()).flip(-1)
    x_down = torch.fft.ifft(X_down)
    return torch.real(x_down) * new_freq / orig_freq


