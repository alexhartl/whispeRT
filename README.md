# whispeRT

A very simple near real-time voice transcription tool in Python based on [OpenAI's whisper](https://openai.com/research/whisper).

Since context is important for a transcription task, this implementation performs inference on a single sentence every 2s, but then sequentially performs inference on audio blocks with lengths of approx. 24s and 72s on the same data. It attempts to detect sentence boundaries based on the signal level dropping below a certain relative threshold.

## Usage
### Stand-alone
Install requirements (`pip install -r requirements.txt`) and PortAudio (`apt install libportaudio2`), run
```
$ python3 transcribe.py
```
to record from the default microphone device and print the transcription to the screen. There are a few command-line options:
```
$ python3 transcribe.py --help
usage: transcribe.py [-h] [--model MODEL] [--show-models] [--device DEVICE] [--show-devices] [--stdin]
                     [--save SAVE]

Simple near real-time voice transcription based on OpenAI's whisper

optional arguments:
  -h, --help       show this help message and exit
  --model MODEL    which whisper model to use. Affects speed, performance and supported language
  --show-models    show available models
  --device DEVICE  number of audio input device to use
  --show-devices   show available audio devices
  --stdin          read audio data (16kHz, 16bit mono) from stdin
  --save SAVE      regularly save transcription to this file
```

### From Python
You can use the generator function `yield_transcription()`, which takes an iterator of 50msec audio blocks (16kHz, 16bit mono), and yields the constantly growing list of transcription blocks. Have fun.

## Hardware Requirements

To run this, you will need a somewhat recent GPU (or extraordinarily potent CPU).