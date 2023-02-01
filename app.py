
import argparse
import os
import io
import requests
import base64

import torch
import torchaudio

from scipy.io import wavfile

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_voice, mp3_bytes_from_wav_bytes


def download_custom_voice(url):
    response = requests.get(url)
    custom_voice_folder = f"tortoise/voices/custom"
    os.makedirs(custom_voice_folder)
    with open(os.path.join(custom_voice_folder, 'input.wav'), 'wb') as f:
        f.write(response.content)


def base64_encode(buffer: io.BytesIO) -> str:
    """
    Encode the given buffer as base64.
    """
    return base64.encodebytes(buffer.getvalue()).decode("ascii")

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    device = 0 if torch.cuda.is_available() else -1
    model = TextToSpeech()

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    
    text = model_inputs.get('text', None)
    voice = model_inputs.get('voice', 'random')
    preset = model_inputs.get('preset', 'fast')

    # get custom wav file
    if voice == 'custom':
        custom_voice_url = model_inputs.get('custom_voice_url', None)
        if custom_voice_url == None:
            return {'message': "Custom voice requires url of audio sample"}
        download_custom_voice(custom_voice_url)

    voice_samples, conditioning_latents = load_voice(voice)
    
    # Run the model
    gen = model.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents, 
                          preset=preset)

    if voice == 'custom':
        custom_voice_folder = f"tortoise/voices/{voice}"
        os.remove(os.path.join(custom_voice_folder, 'input.wav'))
    
    wav_bytes = io.BytesIO()
    wavfile.write(wav_bytes, 24000, gen.squeeze().cpu().numpy())
    wav_bytes.seek(0)
    mp3_bytes = mp3_bytes_from_wav_bytes(wav_bytes)

    result = { 'audio': "data:audio/mpeg;base64," + base64_encode(mp3_bytes)}

    # Return the results as a dictionary
    return result