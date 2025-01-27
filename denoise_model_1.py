# Implementation of noise suppression using DeepFilterNet3

import time
import torchaudio
from df.enhance import enhance, init_df, load_audio, save_audio
from pystoi import stoi
from pesq import pesq
import numpy as np
import torch
from typing import Tuple

def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample audio to the target sample rate.

    Args:
        audio (torch.Tensor): Input audio tensor.
        orig_sr (int): Original sample rate of the audio.
        target_sr (int): Target sample rate to resample to.

    Returns:
        torch.Tensor: Resampled audio tensor.
    """
    if orig_sr != target_sr:
        resample_transform = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        return resample_transform(audio)
    return audio

def calculate_metrics(noisy: torch.Tensor, enhanced: torch.Tensor, sample_rate: int) -> Tuple[float, float, float]:
    """
    Calculate SNR, PESQ, and STOI metrics for audio quality evaluation.

    Args:
        noisy (torch.Tensor): Noisy input audio tensor.
        enhanced (torch.Tensor): Enhanced output audio tensor.
        sample_rate (int): Sample rate of the audio.

    Returns:
        Tuple[float, float, float]: SNR improvement, PESQ score, and STOI score.
    """
    noisy_np = noisy.squeeze().numpy()
    enhanced_np = enhanced.squeeze().numpy()

    # SNR Improvement
    snr = 10 * np.log10(np.mean(enhanced_np**2) / np.mean((noisy_np - enhanced_np)**2))

    # PESQ Score (use 'nb' for narrowband or 'wb' for wideband)
    pesq_score = pesq(sample_rate, noisy_np, enhanced_np, 'wb')

    # STOI Score
    stoi_score = stoi(noisy_np, enhanced_np, sample_rate, extended=False)

    return snr, pesq_score, stoi_score

def main(input_file: str, output_file: str) -> None:
    """
    Main function to perform noise suppression with DeepFilterNet3.

    Args:
        input_file (str): Path to the input noisy audio file.
        output_file (str): Path to save the denoised output audio file.

    Returns:
        None
    """
    # Load DeepFilterNet3 model
    model, df_state, _ = init_df()
    
    # Load the noisy audio file
    noisy_audio, sr = load_audio(input_file, sr=df_state.sr())

    # Perform noise suppression
    start_time = time.time()
    enhanced_audio = enhance(model, df_state, noisy_audio)
    end_time = time.time()
    
    # Save the enhanced audio to disk
    save_audio(output_file, enhanced_audio, df_state.sr())

    # Resample and calculate metrics
    noisy_audio_resampled = resample_audio(torch.tensor(noisy_audio), df_state.sr(), 16000)
    enhanced_resampled = resample_audio(torch.tensor(enhanced_audio), df_state.sr(), 16000)
    snr, pesq_score, stoi_score = calculate_metrics(noisy_audio_resampled, enhanced_resampled, 16000)

    # Log results
    print(f"Inference Time: {end_time - start_time:.2f} seconds")
    print(f"SNR Improvement: {snr:.2f} dB")
    print(f"PESQ Score: {pesq_score:.2f}")
    print(f"STOI Score: {stoi_score:.2f}")

if __name__ == "__main__":
    # Define input and output file paths
    input_file = "input_audio.wav"  # Replace with your noisy audio file
    output_file = "output_audio_1.wav"  # Denoised output file for DeepFilterNet3

    # Run the main function
    main(input_file, output_file)  
