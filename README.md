# Acoustic_Shadowfront
A Windows laptop experiment using stereo speakers and stereo microphone input to test apparent acoustic transition-boundary speed across a two-microphone baseline.
Acoustic Shadowfront / Nullfront Demo

A Windows laptop experiment using stereo speakers and stereo microphone
input to test apparent acoustic transition-boundary speed across a
two-microphone baseline.

Install

py -m pip install numpy sounddevice scipy matplotlib

List audio devices

py acoustic_shadowfront_windows_demo.py –list-devices

Basic run

py acoustic_shadowfront_windows_demo.py –mic-spacing 0.0254

Phase-null test near the successful low-frequency range

py acoustic_shadowfront_windows_demo.py –mode phase_null –mic-spacing
0.0254 –frequency 200

Specify stereo input/output devices

py acoustic_shadowfront_windows_demo.py –mode phase_null –input-device 2
–output-device 5 –mic-spacing 0.0254 –frequency 200

Try nearby frequencies

py acoustic_shadowfront_windows_demo.py –mode phase_null –mic-spacing
0.0254 –frequency 210 py acoustic_shadowfront_windows_demo.py –mode
phase_null –mic-spacing 0.0254 –frequency 230 py
acoustic_shadowfront_windows_demo.py –mode phase_null –mic-spacing
0.0254 –frequency 240

More robust first test

py acoustic_shadowfront_windows_demo.py –mode gated_tone –mic-spacing
0.0254 –frequency 200

Output files

acoustic_shadowfront_recording.npz acoustic_shadowfront_result.png

Notes

Use a true stereo microphone input. Keep volume low. The result measures
apparent boundary traversal across the microphone baseline, not
faster-than-sound energy or information transfer.
