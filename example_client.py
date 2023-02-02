import os # to save generated audio to workspace
import base64 # for decoding response
import IPython # for playing audio in notebook
import banana_dev as banana


api_key = "<BANANA API KEY>"        # replace with your key 
model_key = "<BANANA MODEL KEY>"    # replace with your model's key


# Generate audio
model_inputs = {
    'text': "House to vote on resolution to remove Ilhan Omar from Foreign Affairs Committee.",
    'voice': 'custom', # requires `custom_voice_url`
    'preset': 'fast',
    'custom_voice_url': '<URL TO WAV FILE>' # (optional) required only if voice is `custom`. Host the custom voice sample file online somewhere (ex.: https://tmpfiles.org/)
}
out = banana.run(api_key, model_key, model_inputs)


# Convert response to mp3 and save audio to disk
encoded_bytes = out['modelOutputs'][0]['audio'].split(',')[1].encode("ascii")
decoded_bytes = base64.decodebytes(encoded_bytes)

with open("temp.mp3", "wb") as mp3_file:
    mp3_file.write(decoded_bytes)
