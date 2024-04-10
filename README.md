<head>
<style>
h1 {text-align: center;}
</style>
</head>
<body>


# Sample Rate Independent Recurrent Neural Networks for Audio Effects Processing</h1> 

Accompanying code for DAFx24 submission by Alistair Carson, Alec Wright, Jatin Chowdhury, Vesa Välimäki and Stefan Bilbao.

Audio examples: https://a-carson.github.io/sr_indie_rnn/


#### Description of repo contents

`rnn.py` contains the baseline and sample rate independent RNN implementations.

`giant_fft_resample.py` contains an implementation of high-fidelity <a href="https://www.aes.org/e-lib/browse.cfm?elib=22033">sample rate conversion using Giant FFTs</a> for audio resampling.

`process_audio.py` shows an example use case and computes SNR of modified-RNN output compared to the original RNN at the training sample-rate.

`Proteus_Tone_Packs/` contains some pre-trained LSTM models, obtained from the <a href="https://guitarml.com/tonelibrary/tonelib-pro.html">Guitar ML Tone Library</a>. More can be downloaded from this link.

`audio/` contains the test signal used to generate results in the paper. 

#### Requirements
Install requirements via:
```angular2html
conda env create -f conda_env_cpu.yaml
conda activate sr_indie_rnn
```


</body>