import os
from dotenv import load_dotenv
import pvporcupine
import pyaudio
import struct
import speech_recognition as sr
import agent  # Change back to importing the whole module
import subprocess
from openai import OpenAI  # Add OpenAI import

# Load environment variables from .env file
load_dotenv()

access_key = os.getenv('ACCESS_KEY')
keyword_path = os.getenv('KEYWORD_PATH')

# Initialize Porcupine with the downloaded keyword file
porcupine = pvporcupine.create(
    access_key=access_key,
    keyword_paths=[keyword_path]
)

pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length * 2,
    input_device_index=2
)

for i in range(pa.get_device_count()):
    print(pa.get_device_info_by_index(i))

recognizer = sr.Recognizer()  # Initialize the recognizer

# Create a short beep sound file if it doesn't exist
def generate_beep_file():
    import numpy as np
    import wave
    
    frequency = 880  # A5 note
    duration = 0.1  # seconds
    sample_rate = 48000
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    note = np.sin(2 * np.pi * frequency * t)
    
    # Add fade in/out
    fade_length = int(0.005 * sample_rate)
    fade_in = np.linspace(0, 1, fade_length)
    fade_out = np.linspace(1, 0, fade_length)
    note[:fade_length] *= fade_in
    note[-fade_length:] *= fade_out
    
    audio = note * (2**15 - 1) / np.max(np.abs(note))
    audio = audio.astype(np.int16)
    
    # Save as WAV file
    with wave.open('notification.wav', 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())

# Generate the beep file when the script starts
generate_beep_file()

# Add this function after generate_beep_file()
def speak_text(text):
    client = OpenAI()  # Uses API key from environment variables
    
    try:
        # Generate speech using OpenAI's TTS with streaming
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text
        ).with_streaming_response()
        
        # Save the audio to a temporary file
        temp_file = "temp_speech.mp3"
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
        
        # Play the audio using afplay (for macOS)
        subprocess.run(['afplay', temp_file])
        
        # Clean up the temporary file
        os.remove(temp_file)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

try:
    while True:
        try:
            pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Hotword detected!")
                # Play the notification sound using afplay
                try:
                    subprocess.Popen(['afplay', 'notification.wav'])
                except Exception as e:
                    print(f"Error playing sound: {e}")

                print("Listening for speech...")
                audio_data = []
                for _ in range(0, int(porcupine.sample_rate / porcupine.frame_length * 5)):  # Capture for 5 seconds
                    pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
                    audio_data.append(pcm)

                # Convert the captured audio to a format suitable for SpeechRecognition
                audio_data = b''.join(audio_data)
                audio = sr.AudioData(audio_data, porcupine.sample_rate, 2)

                try:
                    # Convert audio to text
                    text = recognizer.recognize_google(audio)
                    print(f"Recognized Text: {text}")

                    # Use the module-style import and extract just the answer text
                    response = agent.process_query_sync(text)
                    returned_text = response.answer
                    print(f"Returned Text: {returned_text}")
                    speak_text(returned_text)  # Speak the response

                except sr.UnknownValueError:
                    print("Could not understand audio")
                except sr.RequestError as e:
                    print(f"Could not request results; {e}")
        except OSError as e:
            print(f"Audio input overflowed: {e}")
finally:
    audio_stream.close()
    porcupine.delete()