from queue import Queue
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import pvporcupine
import dotenv
import argparse
import os
import struct
import pyaudio
import pvrecorder
import numpy as np
import whisper
import pvcheetah
import openai
from scipy.io.wavfile import write
import pyttsx3

import torch
parser = argparse.ArgumentParser(description='Run the OAI assistant')
parser.add_argument('--prompt', type=str, help='Prompt to start the conversation with')
parser.add_argument('--show-devices', action='store_true', help='Show available audio devices')
parser.add_argument('--device-id', type=int, help='Audio device ID to use', default=-1)

engine = pyttsx3.init()


dotenv.load_dotenv()

def main():
    args = parser.parse_args()
    whisper_model = whisper.load_model("medium.en")
    try:
        porcupine = pvporcupine.create(access_key=os.environ["PORCUPINE_API_KEY"], keyword_paths=["porcupine-models/habibi.ppn", "porcupine-models/lym.ppn"])
        recorder = pvrecorder.PvRecorder(device_index=args.device_id, frame_length=porcupine.frame_length)
        cheetah = pvcheetah.create(os.environ["PORCUPINE_API_KEY"])

        if args.show_devices:
            devices = recorder.get_available_devices()
            print("Available devices:")
            for i, device in enumerate(devices):
                print(f"{i}: {device}")
            exit(0)

        def get_next_audio_frame():
            pcm = recorder.read()
            #pcm = struct.unpack_from("h" * porcupine.frame_length, bytes(pcm))
            return pcm
        
        def process_transcription(transcribed):
            config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
            assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
            user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding"}, human_input_mode="NEVER")
            user_proxy.initiate_chat(assistant, message=transcribed)
            return user_proxy.last_message

        
        def text_to_speech(text):
            engine.say(text)
            engine.runAndWait()


        def record_next_sentence():
            """Record the next sentence, until the user stops talking"""
            print("Recording...")
            audio = []
            while True:
                pcm = get_next_audio_frame()
                # Test if pcm has values above threshold
                processed = porcupine.process(pcm)
                if processed == 1:
                    print("Hotword detected, stopping recording")
                    break
                audio.extend(np.array(pcm))
            #audio = np.concatenate(audio)
            print("Done recording")
            print(f"Sample rate: {porcupine.sample_rate}")
            write("test.wav", porcupine.sample_rate, np.array(audio, dtype=np.float32))
            return torch.from_numpy(np.array(audio, dtype=np.float32) )

        recorder.start()
        print("Listening for 'habibi'...")
        while True:
            pcm = get_next_audio_frame()
            result = porcupine.process(pcm)
            if result == 0:
                print('Hotword Detected!')
                sentence = record_next_sentence()
                print("Transcribing...")
                transcribed = whisper_model.transcribe("test.wav", temperature=0)
                print(transcribed)
                reply = process_transcription(transcribed["text"])
                print(reply)
                text_to_speech(reply)

    except KeyboardInterrupt:
        print('Stopping')

    finally:
        if 'porcupine' in locals():
            porcupine.delete()
        if 'recorder' in locals():
            recorder.delete()
    
if __name__ == '__main__':
    main()