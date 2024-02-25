#!/usr/bin/env python3

import argparse
import collections
import os
import signal
import sys
from typing import Iterable, Optional

import numpy as np
import scipy.signal as sps
import whisper
import torch

BLOCK_LENGTH = 0.05 # seconds
VOL_OBSERVATION_WINDOW = 60 # seconds
SILENCE_THRESH_DB = 50
STAGES = 2
CONSOLIDATION_LENGTH = 3
MIN_STATEMENT_LENGTH = 8 # seconds
MAX_STATEMENT_LENGTH = 60 # seconds

SAMPLING_RATE = 16000
BLOCK_SIZE = int(SAMPLING_RATE * BLOCK_LENGTH)

def is_downsampling_needed(device: int):
    # lazily import sounddevice, which requires libportaudio.
    # if we just want to read form stdin, we don't need this
    import sounddevice
    try:
        sounddevice.check_input_settings(device, channels=1, samplerate=SAMPLING_RATE)
    except sounddevice.PortAudioError:
        return True
    return False

def yield_from_device(device: int, downsample_from_48k: bool = False, shutdown: Optional[list] = None):
    import sounddevice
    effective_block_size = BLOCK_SIZE
    if downsample_from_48k:
        effective_block_size *= 3
    with sounddevice.InputStream(device=device, channels=1, samplerate=48000 if downsample_from_48k else SAMPLING_RATE,
                        dtype=np.int16, blocksize=effective_block_size) as stream:
        while not shutdown:
            audio_block, _ = stream.read(effective_block_size)
            yield audio_block[:,0]

def yield_from_stdin(shutdown: Optional[list] = None):
        while not shutdown:
            yield np.frombuffer(sys.stdin.buffer.read(BLOCK_SIZE*2), dtype=np.int16)

def yield_transcription(audio_input: Iterable[np.ndarray], model_name: str ='tiny', downsample_from_48k: bool = False):
    model = whisper.load_model(model_name)
    statement = []
    blocks = [ [] for _ in range(STAGES+1) ]
    energies = collections.deque(maxlen=int(VOL_OBSERVATION_WINDOW/BLOCK_LENGTH))
    transcription = [ '' ]

    def do_inference(audio):
        if downsample_from_48k:
            audio = sps.decimate(audio, 3).copy()
        return model.transcribe(audio, condition_on_previous_text=False, fp16=torch.cuda.is_available())['text']

    def consolidate(force):
        stage = 0
        consolidated_statement_cnt = 1
        while stage < STAGES and (len(blocks[stage]) >= CONSOLIDATION_LENGTH or force):
            consolidated_statement_cnt += len(blocks[stage])-1
            blocks[stage+1].append(np.concatenate(blocks[stage]))
            blocks[stage].clear()
            stage += 1
        transcription[-consolidated_statement_cnt:] = [ do_inference(blocks[stage][-1]) ]
        blocks[-1].clear()

    for audio_block in audio_input:
        audio_block = audio_block.astype(np.float32) / 32768.0
        assert audio_block.size == (BLOCK_SIZE*3 if downsample_from_48k else BLOCK_SIZE)
        energy = np.sum(audio_block*audio_block)
        energies.append(energy)
        statement.append(audio_block)

        if len(statement)*BLOCK_LENGTH >= MIN_STATEMENT_LENGTH and (len(statement)*BLOCK_LENGTH >= MAX_STATEMENT_LENGTH or np.mean(energies[-6]) < np.mean(energies) / 10**(SILENCE_THRESH_DB/10)):
            blocks[0].append(np.concatenate(statement))
            transcription[-1] = do_inference(blocks[0][-1])
             
            if len(blocks[0]) >= CONSOLIDATION_LENGTH:
                consolidate(force=False)

            statement.clear()
            transcription.append('')
        elif len(statement) % 40 == 0:
            transcription[-1] = do_inference(np.concatenate(statement))
        yield transcription
    blocks[0].append(np.concatenate(statement))
    consolidate(force=True)
    yield transcription

def main():
    parser = argparse.ArgumentParser(description="Simple near real-time voice transcription based on OpenAI's whisper")
    parser.add_argument('--model', type=str, default='tiny', help='which whisper model to use. Affects speed, performance and supported languages')
    parser.add_argument('--show-models', action='store_true', help='show available models')
    parser.add_argument('--device', type=int, help='number of audio input device to use')
    parser.add_argument('--show-devices', action='store_true', help='show available audio devices')
    parser.add_argument('--stdin', action='store_true', help='read audio data (16kHz, 16bit mono) from stdin')
    parser.add_argument('--save', type=str, help='regularly save transcription to this file')
    args = parser.parse_args()

    if args.show_models:
        print('\n'.join(whisper.available_models()))
    elif args.show_devices:
        import sounddevice
        print(sounddevice.query_devices())
    else:
        shutdown = []
        signal.signal(signal.SIGINT, lambda sig,frame: shutdown.append(None))

        if args.stdin:
            downsample_from_48k = False
            source = yield_from_stdin(shutdown)
        else:
            downsample_from_48k = is_downsampling_needed(args.device)
            source = yield_from_device(args.device, downsample_from_48k, shutdown)

        for t in yield_transcription(source, model_name=args.model, downsample_from_48k=downsample_from_48k):
            text = ' '.join(t)
            os.system('cls' if os.name == 'nt' else 'clear')
            print(text)
            if args.save is not None:
                open(args.save, 'w').write(text)

if __name__ == '__main__':
    main()
