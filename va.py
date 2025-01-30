import os
from dotenv import load_dotenv
import pvporcupine
import pyaudio
import struct
import speech_recognition as sr
import agent  # Change back to importing the whole module
import subprocess
from openai import OpenAI  # Add OpenAI import
import numpy as np
import wave
from playsound import playsound
from scipy import signal  # Add this import at the top

# Load environment variables from .env file
load_dotenv()

access_key = os.getenv('ACCESS_KEY')
keyword_path = os.getenv('KEYWORD_PATH')

# Initialize Porcupine with the downloaded keyword file and specified sample rate
porcupine = pvporcupine.create(
    access_key=access_key,
    keyword_paths=[keyword_path],
)

pa = pyaudio.PyAudio()  # Initialize PyAudio first

# Print available audio devices
print("Available audio devices:")
for i in range(pa.get_device_count()):
    device_info = pa.get_device_info_by_index(i)
    print(f"Device {i}: {device_info['name']}")

# Try to find the default input device
default_input_device = None
for i in range(pa.get_device_count()):
    device_info = pa.get_device_info_by_index(i)
    if device_info['maxInputChannels'] > 0:
        default_input_device = i
        print(f"Using device: {device_info['name']}")
        print(f"Default sample rate: {int(device_info['defaultSampleRate'])}")
        break

if default_input_device is None:
    raise RuntimeError("No input device found")

# Get device info to check supported sample rate
device_info = pa.get_device_info_by_index(default_input_device)
supported_sample_rate = int(device_info['defaultSampleRate'])

# Check if the device's sample rate matches Porcupine's required rate
if supported_sample_rate != porcupine.sample_rate:
    print(f"Warning: Device sample rate ({supported_sample_rate}) differs from Porcupine's required rate ({porcupine.sample_rate})")
    # You might need to use a different device or configure your system's audio settings

# Try to open the stream with the supported sample rate
audio_stream = pa.open(
    rate=supported_sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=int(porcupine.frame_length * supported_sample_rate / porcupine.sample_rate),  # Adjust buffer size
    input_device_index=default_input_device
)

recognizer = sr.Recognizer()  # Initialize the recognizer

# Create a short beep sound file if it doesn't exist
def generate_beep_file():
    frequency = 880  # A5 note
    duration = 0.1  # seconds
    sample_rate = 16000
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
        # Generate speech using OpenAI's TTS
        response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",  # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        
        # Save the audio to a temporary file
        temp_file = "temp_speech.mp3"
        response.stream_to_file(temp_file)
        
        # Cross-platform audio playback
        try:
            playsound(temp_file)
        except Exception as e:
            print(f"Error playing audio: {e}")
            # Fallback for Linux systems if playsound fails
            if os.name == 'posix':
                try:
                    subprocess.run(['aplay' if os.system('which aplay') == 0 else 'paplay', temp_file])
                except Exception as e:
                    print(f"Fallback audio playback failed: {e}")
        
        # Clean up the temporary file
        os.remove(temp_file)
    except Exception as e:
        print(f"Error in text-to-speech: {e}")

def resample_audio(audio_data, original_rate, target_rate):
    """Resample audio data to target sample rate."""
    # Calculate resampling ratio
    ratio = target_rate / original_rate
    # Calculate new length
    new_length = int(len(audio_data) * ratio)
    # Resample using scipy.signal.resample
    resampled = signal.resample(audio_data, new_length)
    return resampled.astype(np.int16)

try:
    while True:
        try:
            # Calculate required input frames based on sample rate ratio
            input_frame_length = int(porcupine.frame_length * supported_sample_rate / porcupine.sample_rate)
            pcm = audio_stream.read(input_frame_length, exception_on_overflow=False)
            pcm = struct.unpack_from("h" * input_frame_length, pcm)
            
            # Resample if the rates don't match
            if supported_sample_rate != porcupine.sample_rate:
                pcm = resample_audio(
                    np.array(pcm), 
                    supported_sample_rate, 
                    porcupine.sample_rate
                )
                
            # Ensure we have exactly the number of frames Porcupine expects
            if len(pcm) > porcupine.frame_length:
                pcm = pcm[:porcupine.frame_length]
            elif len(pcm) < porcupine.frame_length:
                pcm = np.pad(pcm, (0, porcupine.frame_length - len(pcm)))

            keyword_index = porcupine.process(pcm)
            if keyword_index >= 0:
                print("Hotword detected!")
                # Play the notification sound using cross-platform method
                try:
                    # Might need this on Ubuntu `sudo apt-get install alsa-utils pulseaudio``
                    playsound('notification.wav')
                except Exception as e:
                    print(f"Error playing sound: {e}")
                    # Fallback for Linux systems if playsound fails
                    if os.name == 'posix':
                        try:
                            subprocess.run(['aplay' if os.system('which aplay') == 0 else 'paplay', 'notification.wav'])
                        except Exception as e:
                            print(f"Fallback audio playback failed: {e}")

                print("Listening for speech...")
                # Create a microphone source from the stream
                with sr.Microphone(
                    sample_rate=porcupine.sample_rate,
                    chunk_size=porcupine.frame_length,
                    device_index=default_input_device
                ) as source:
                    # Adjust the recognizer sensitivity to ambient noise
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    # Listen until silence is detected
                    # phrase_time_limit=None means no time limit
                    audio = recognizer.listen(source, phrase_time_limit=None)
                
                # Convert audio to text
                text = recognizer.recognize_google(audio)
                print(f"Recognized Text: {text}")

                # Use the module-style import and extract just the answer text
                response = agent.process_query_sync(text)
                returned_text = response.answer
                print(f"Returned Text: {returned_text}")
                speak_text(returned_text)  # Speak the response

                # Add automatic follow-up listening if required
                if response.requires_followup:
                    print("Listening for follow-up response...")
                    with sr.Microphone(
                        sample_rate=porcupine.sample_rate,
                        chunk_size=porcupine.frame_length,
                        device_index=default_input_device
                    ) as source:
                        recognizer.adjust_for_ambient_noise(source, duration=0.5)
                        audio = recognizer.listen(source, phrase_time_limit=None)
                    
                    follow_up_text = recognizer.recognize_google(audio)
                    print(f"Follow-up response: {follow_up_text}")
                    
                    # Process the follow-up response
                    follow_up_response = agent.process_query_sync(follow_up_text)
                    returned_text = follow_up_response.answer
                    print(f"Returned Text: {returned_text}")
                    speak_text(returned_text)

        except OSError as e:
            print(f"Audio input overflowed: {e}")
finally:
    audio_stream.close()
    porcupine.delete()