
import torch
import torchaudio
from giant_fft_resample import giant_fft_resample
import rnn


audio_path = 'audio/test_signal_input.wav'
model_path = 'Proteus_Tone_Packs/6505Plus_Red_DirectOut.json'
os_ratio = [160, 147]       # [new_freq, orig_freq] e.g [160, 147] for SR conversion from 44.1kHz to 48kHz
method = 'apdl'             # sample rate conversion method
num_samples = -1            # option to truncate input audio

# load audio
in_sig, base_sample_rate = torchaudio.load(audio_path)
in_sig = in_sig[..., :num_samples]



# load models
base_model = rnn.get_AudioRNN_from_json(model_path)
sr_indie_model = rnn.get_SRIndieRNN(base_model=base_model, method=method)
sr_indie_model.rec.os_factor = os_ratio[0] / os_ratio[1]

with torch.no_grad():

    # oversample
    in_sig_os = giant_fft_resample(in_sig, orig_freq=os_ratio[1], new_freq=os_ratio[0])
    #  process
    out_sig_os, _ = sr_indie_model(in_sig_os)
    # downsample
    out_sig = giant_fft_resample(out_sig_os, orig_freq=os_ratio[0], new_freq=os_ratio[1])

    # target
    out_sig_target, _ = base_model(in_sig)

# compute SNR compared to target
diff = out_sig[..., :out_sig_target.shape[-1]].flatten() - out_sig_target.flatten()
snr = out_sig_target.flatten().square().sum() / diff.square().sum()
snr_dB = 10 * torch.log10(snr)
print('SNR = {} dB'.format(snr_dB))

