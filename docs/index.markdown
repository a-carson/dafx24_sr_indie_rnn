---
layout: splash
classes:
  - wide
---
<h2 align="center">Sample Rate Independent Recurrent Neural Networks for Audio Effects Processing</h2>
<p style="font-size: 0.75em" align="center">
Alistair Carson, Alec Wright, Jatin Chowdhury, Vesa Välimäki and Stefan Bilbao</p>
<p style="font-size: 0.75em" align="center">
Welcome to the accompanying web-page for our DAFx24 submission.</p>
<p style="font-size: 0.75em" align="center">
For example code see <a href="https://github.com/a-carson/dafx24_sr_indie_rnn?tab=readme-ov-file" target="_blank" rel="noopener noreferrer"> here</a>.</p>




###### <b>Abstract</b>
<p style="font-size: 0.75em">
In recent years, machine learning approaches to modelling guitar amplifiers and effects pedals have been widely investigated and have become standard practice in some consumer products. In particular, recurrent neural networks (RNNs) are a popular choice for modelling non-linear devices such as vacuum tube amplifiers and distortion circuitry. One limitation of such models is that they are trained on audio at a specific sample rate and therefore give unreliable results when operating at another rate. Here, we investigate several methods of modifying RNN structures to make them approximately sample rate independent, with a focus on oversampling. In the case of integer oversampling, we demonstrate that a previously proposed delay-based approach provides high fidelity sample rate conversion whilst additionally reducing aliasing. For non-integer sample rate adjustment, we propose two novel methods and show that one of these, based on cubic Lagrange interpolation of a delay-line, provides a significant improvement over existing methods. To our knowledge, this work provides the first in-depth study into this problem.
</p>


###### <b>Audio Examples</b>
<p style="font-size: 0.75em">
Below are examples of five LSTM RNN models from the <a href="https://guitarml.com/tonelibrary/tonelib-pro.html" target="_blank" rel="noopener noreferrer">GuitarML Tone Library</a>. 
These models are designed for operation at a sample rate (SR) of 44.1kHz. The audio examples below are the output signals when operating at different inference SRs, using the methods outlined in the paper:
</p>
<ul>
  <li style="font-size: 0.75em"> Naive (operating original RNN at different SRs, first column)</li>
  <li style="font-size: 0.75em">State-trajectory network (STN)</li>
  <li style="font-size: 0.75em">Linearly interpolated delay line (LIDL) </li>
  <li style="font-size: 0.75em">All-pass filter delay line (APDL) </li>
  <li style="font-size: 0.75em">Cubic interpolated delay-line (CIDL) -- this is our recommended method for the highest quality sample rate conversion. </li>
</ul>
<p style="font-size: 0.75em">
Note that for integer oversampling (e.g. 44.1kHz to 88.2kHz), the latter three methods produce an identical output, so only one is shown.
</p>
<br>
1) Peavey 6505+ tube amp -- high gain
<table>
  <thead>
    <tr>
      <th style="background: white; text-align: center; font-weight: normal">Input signal</th>
      <th style="background: white; text-align: center;">
        <audio controls style="width: 12em">
          <source src="audio/input_riff1.wav" type="audio/wav">
        </audio></th>
      <th style="border-left-style: hidden; border-top-style: hidden; visibility:  hidden; background: white; text-align: center"></th>
      <th colspan="3" style="text-align: center">Delay-based methods</th>
    </tr>
    <tr>
      <th style="text-align: center">Inference SR</th>
      <th style="text-align: center">Original RNN <br> (trained at 44.1kHz)</th>
      <th style="text-align: center">STN  </th>
      <th style="text-align: center">LIDL  </th>
      <th style="text-align: center">APDL* (ours) </th>
      <th style="text-align: center">CIDL (ours) </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">44.1kHz</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_44100_naive_riff1.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        - </td>
    </tr>
    <tr>
      <td style="text-align: center">48kHz</td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_48000_naive_riff1.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_48000_stn_riff1.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_48000_lidl_riff1.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_48000_apdl_riff1.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_48000_lagrange_riff1.wav" type="audio/wav">        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">88.2kHz</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_88200_naive_riff1.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_88200_stn_riff1.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center" colspan="3" style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_88200_lidl_riff1.wav" type="audio/wav">        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">96kHz</td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_96000_naive_riff1.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_96000_stn_riff1.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_96000_lidl_riff1.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_96000_apdl_riff1.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/6505Plus_Red_DirectOut_96000_lagrange_riff1.wav" type="audio/wav">        </audio></td>
    </tr>
  </tbody>
</table>
<p style="font-size: 0.75em">
*here the APDL method produces unwanted artefacts at 48kHz, as noted in the paper. This audio clip was specifically chosen to demonstrate these artefacts.
</p>
<br>

2) Blackstar HT40 tube amp -- clean
<table>

  <thead>
    <tr>
      <th style="background: white; text-align: center; font-weight: normal">Input signal</th>
      <th style="background: white; text-align: center;">
        <audio controls style="width: 12em">
          <source src="audio/input_riff2.wav" type="audio/wav">
        </audio></th>
      <th style="border-left-style: hidden; border-top-style: hidden; visibility:  hidden; background: white; text-align: center"></th>
      <th colspan="3" style="text-align: center">Delay-based methods</th>
    </tr>
    <tr>
      <th style="text-align: center">Inference SR</th>
      <th style="text-align: center">Original RNN <br> (trained at 44.1kHz) </th>
      <th style="text-align: center">STN  </th>
      <th style="text-align: center">LIDL  </th>
      <th style="text-align: center">APDL (ours) </th>
      <th style="text-align: center">CIDL (ours) </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">44.1kHz</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_44100_naive_riff2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        - </td>
    </tr>
    <tr>
      <td style="text-align: center">48kHz</td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_48000_naive_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_48000_stn_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_48000_lidl_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_48000_apdl_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_48000_lagrange_riff2.wav" type="audio/wav">        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">88.2kHz</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_88200_naive_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_88200_stn_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center" colspan="3" style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_88200_lidl_riff2.wav" type="audio/wav">        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">96kHz</td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_96000_naive_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_96000_stn_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_96000_lidl_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_96000_apdl_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpClean_96000_lagrange_riff2.wav" type="audio/wav">        </audio></td>
    </tr>
  </tbody>
</table>
<br>

3) Blackstar HT40 tube amp -- high gain
<table>
  <thead>
    <tr>
      <th style="background: white; text-align: center; font-weight: normal">Input signal</th>
      <th style="background: white; text-align: center;">
        <audio controls style="width: 12em">
          <source src="audio/input_riff2.wav" type="audio/wav">
        </audio></th>
      <th style="border-left-style: hidden; border-top-style: hidden; visibility:  hidden; background: white; text-align: center"></th>
      <th colspan="3" style="text-align: center">Delay-based methods</th>
    </tr>
    <tr>
      <th style="text-align: center">Inference SR</th>
      <th style="text-align: center">Original RNN <br> (trained at 44.1kHz) </th>
      <th style="text-align: center">STN  </th>
      <th style="text-align: center">LIDL  </th>
      <th style="text-align: center">APDL (ours) </th>
      <th style="text-align: center">CIDL (ours) </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">44.1kHz</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_44100_naive_riff2.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        - </td>
    </tr>
    <tr>
      <td style="text-align: center">48kHz</td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_48000_naive_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_48000_stn_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_48000_lidl_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_48000_apdl_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_48000_lagrange_riff2.wav" type="audio/wav">        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">88.2kHz</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_88200_naive_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_88200_stn_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center" colspan="3" style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_88200_lidl_riff2.wav" type="audio/wav">        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">96kHz</td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_96000_naive_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_96000_stn_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_96000_lidl_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_96000_apdl_riff2.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/BlackstarHT40_AmpHighGain_96000_lagrange_riff2.wav" type="audio/wav">        </audio></td>
    </tr>
  </tbody>
</table>
<br>

4) Rockman acoustic simulator pedal
<table>
  <thead>
    <tr>
      <th style="background: white; text-align: center; font-weight: normal">Input signal</th>
      <th style="background: white; text-align: center;">
        <audio controls style="width: 12em">
          <source src="audio/input_riff3.wav" type="audio/wav">
        </audio></th>
      <th style="border-left-style: hidden; border-top-style: hidden; visibility:  hidden; background: white; text-align: center"></th>
      <th colspan="3" style="text-align: center">Delay-based methods</th>
    </tr>
    <tr>
      <th style="text-align: center">Inference SR</th>
      <th style="text-align: center">Original RNN <br> (trained at 44.1kHz) </th>
      <th style="text-align: center">STN  </th>
      <th style="text-align: center">LIDL  </th>
      <th style="text-align: center">APDL (ours) </th>
      <th style="text-align: center">CIDL (ours) </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">44.1kHz</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_44100_naive_riff3.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        - </td>
    </tr>
    <tr>
      <td style="text-align: center">48kHz</td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_48000_naive_riff3.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_48000_stn_riff3.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_48000_lidl_riff3.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_48000_apdl_riff3.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_48000_lagrange_riff3.wav" type="audio/wav">        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">88.2kHz</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_88200_naive_riff3.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_88200_stn_riff3.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center" colspan="3" style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_88200_lidl_riff3.wav" type="audio/wav">        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">96kHz</td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_96000_naive_riff3.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_96000_stn_riff3.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_96000_lidl_riff3.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_96000_apdl_riff3.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/RockmanAcoustic_Pedal_96000_lagrange_riff3.wav" type="audio/wav">        </audio></td>
    </tr>
  </tbody>
</table>
<br>

5) Xotic SP compressor pedal
<table>
  <thead>
    <tr>
      <th style="background: white; text-align: center; font-weight: normal">Input signal</th>
      <th style="background: white; text-align: center;">
        <audio controls style="width: 12em">
          <source src="audio/input_riff4.wav" type="audio/wav">
        </audio></th>
      <th style="border-left-style: hidden; border-top-style: hidden; visibility:  hidden; background: white; text-align: center"></th>
      <th colspan="3" style="text-align: center">Delay-based methods</th>
    </tr>
    <tr>
      <th style="text-align: center">Inference SR</th>
      <th style="text-align: center">Original RNN <br> (trained at 44.1kHz) </th>
      <th style="text-align: center">STN  </th>
      <th style="text-align: center">LIDL  </th>
      <th style="text-align: center">APDL (ours) </th>
      <th style="text-align: center">CIDL (ours) </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center">44.1kHz</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_44100_naive_riff4.wav" type="audio/wav">
        </audio></td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        -</td>
      <td style="text-align: center">
        - </td>
    </tr>
    <tr>
      <td style="text-align: center">48kHz</td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_48000_naive_riff4.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_48000_stn_riff4.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_48000_lidl_riff4.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_48000_apdl_riff4.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_48000_lagrange_riff4.wav" type="audio/wav">        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">88.2kHz</td>
      <td style="text-align: center" >
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_88200_naive_riff4.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_88200_stn_riff4.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center" colspan="3" style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_88200_lidl_riff4.wav" type="audio/wav">        </audio></td>
    </tr>
    <tr>
      <td style="text-align: center">96kHz</td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_96000_naive_riff4.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_96000_stn_riff4.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_96000_lidl_riff4.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_96000_apdl_riff4.wav" type="audio/wav">        </audio></td>
      <td style="text-align: center">
        <audio controls style="width: 12em">
          <source src="audio/XComp_Pedal_96000_lagrange_riff4.wav" type="audio/wav">        </audio></td>
    </tr>
  </tbody>
</table>

