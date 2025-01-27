## Noise Suppression Models for Virtual Assistant
This repository contains Python scripts for implementing and evaluating two noise suppression models: DeepFilterNet and SepFormer. The goal of these models is to enhance the audio quality by reducing background noise in noisy recordings, making them suitable for applications such as virtual assistants.

# Overview
DeepFilterNet: A deep learning model that removes background noise from audio signals, improving the clarity and intelligibility of speech.
SepFormer: Another deep learning-based noise suppression model designed for real-time applications, optimized for reduced latency and resource usage.
Both models have been implemented and tested with pre-trained weights on a noisy audio file to produce denoised output. The scripts include metrics for evaluating performance such as SNR improvement, PESQ, and STOI.

# Repository Structure
```
whitepaper.pdf: A whitepaper comparing the two models, discussing their performance, and providing recommendations for their use in real-time virtual assistants like Ale.
README.md: This file.
requirements.txt: A list of dependencies required to run the scripts.
denoise_model_1.py: Python script for implementing the DeepFilterNet model.
denoise_model_2.py: Python script for implementing the SepFormer model.
input_audio.wav: A sample noisy audio file for testing the models.
output_audio_1.wav: Output denoised audio using DeepFilterNet.
output_audio_2.wav: Output denoised audio using SepFormer.
```
# Install the dependencies by running:
```
pip install -r requirements.txt
```
# Usage
1. Clone the repository:
```
git clone https://github.com/awerne22/noise-suppression.git
cd noise-suppression
```
3. Run the noise suppression models:
To process the noisy audio file using DeepFilterNet:
```
python denoise_model_1.py --input input_audio.wav --output output_audio_1.wav
```
To process the noisy audio file using SepFormer:
```
python denoise_model_2.py --input input_audio.wav --output output_audio_2.wav
```
# 3. Evaluate performance:
The output audio files (output_audio_1.wav and output_audio_2.wav) can be evaluated using SNR, PESQ, and STOI scores, which are calculated within the scripts.

# Results
The following files are provided as part of the demo:
```
input_audio.wav: A noisy sample audio file.
output_audio_1.wav: Denoised output using DeepFilterNet.
output_audio_2.wav: Denoised output using SepFormer.
```


