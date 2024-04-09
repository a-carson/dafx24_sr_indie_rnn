import math
import torch
from torch import tensor as T

'''
Giant FFT resampling

Based on method proposed in:
Vesa Välimäki and Stefan Bilbao, "Giant FFTs for Sample-Rate Conversion" in Journal of the Audio Eng. Soc. (JAES), 2023
https://www.aes.org/e-lib/browse.cfm?elib=22033

for best results x should be even-length

'''


def giant_fft_resample(x: T, orig_freq: int, new_freq: int):

    # lengths
    n_in = x.shape[-1]
    m = 2 * math.ceil(n_in / 2 / orig_freq)  # fft zero-pad factor
    n_fft_orig = m * orig_freq
    n_fft_new = m * new_freq
    n_out = math.ceil(new_freq / orig_freq * n_in)

    # fft
    x_fft_og = torch.fft.rfft(x, n_fft_orig)
    x_fft_new = torch.zeros((1, n_fft_new // 2 + 1), dtype=x_fft_og.dtype, device=x_fft_og.device)

    if new_freq > orig_freq:
        # pad fft
        x_fft_new[..., 0:n_fft_orig // 2] = x_fft_og[..., 0:n_fft_orig // 2].clone()
        x_fft_new[..., n_fft_orig // 2] = 0.5 * x_fft_og[..., n_fft_orig // 2].clone()
    else:
        # truncate fft
        x_fft_new[..., 0:n_fft_new // 2] = x_fft_og[..., 0:n_fft_new // 2].clone()

    # ifft
    x_new = torch.fft.irfft(x_fft_new)

    # truncate and scale
    return x_new[..., :n_out] * new_freq / orig_freq




