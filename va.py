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
import sys
import time

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

# Modify the audio stream setup to be more robust
def create_audio_stream(pa, device_index, sample_rate, frame_length):
    try:
        return pa.open(
            rate=sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=frame_length,
            input_device_index=device_index
        )
    except OSError as e:
        print(f"Error opening audio stream: {e}")
        return None

# Replace the audio_stream initialization with:
audio_stream = create_audio_stream(
    pa,
    default_input_device,
    supported_sample_rate,
    int(porcupine.frame_length * supported_sample_rate / porcupine.sample_rate)
)

if audio_stream is None:
    print("Failed to initialize audio stream. Exiting.")
    sys.exit(1)

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
            voice="onyx",
            input=text
        )
        
        temp_file = "temp_speech.mp3"
        temp_wav = "temp_speech.wav"
        
        with open(temp_file, 'wb') as f:
            response.write_to_file(temp_file)
        
        # Convert MP3 to WAV first using ffmpeg for Linux systems
        if os.name == 'posix':
            try:
                subprocess.run(['ffmpeg', '-y', '-i', temp_file, temp_wav], 
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
                # Use -D parameter to specify ALSA device
                subprocess.run(['aplay', '-D', 'plughw:1,0', temp_wav],  # Modify 1,0 to match your device
                             stdout=subprocess.DEVNULL, 
                             stderr=subprocess.DEVNULL)
                os.remove(temp_wav)
            except Exception as e:
                print(f"Error with ffmpeg/aplay: {e}")
        else:
            try:
                # For Windows, we can use the winmm audio backend with device selection
                import sounddevice as sd
                import soundfile as sf
                
                # Convert MP3 to WAV for sounddevice compatibility
                subprocess.run(['ffmpeg', '-y', '-i', temp_file, temp_wav],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
                
                # Read the WAV file
                data, samplerate = sf.read(temp_wav)
                
                # Play through specified device (change device number as needed)
                sd.play(data, samplerate, device=1)  # Modify device number as needed
                sd.wait()  # Wait until audio is finished playing
                
                os.remove(temp_wav)
            except Exception as e:
                print(f"Error playing audio: {e}")
                # Fallback to playsound if sounddevice fails
                try:
                    playsound(temp_file)
                except Exception as e:
                    print(f"Error playing audio with playsound: {e}")
        
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

# Modify the Microphone setup to use the supported sample rate instead of Porcupine's rate
def create_speech_recognizer(device_index, sample_rate):
    try:
        mic = sr.Microphone(
            device_index=device_index,
            sample_rate=sample_rate,  # Use the device's supported rate
            chunk_size=1024  # Use a standard chunk size
        )
        return mic
    except Exception as e:
        print(f"Error creating speech recognizer: {e}")
        return None

try:
    while True:
        try:
            # Ensure we have an open audio stream
            if audio_stream is None or not audio_stream.is_active():
                audio_stream = create_audio_stream(
                    pa,
                    default_input_device,
                    supported_sample_rate,
                    int(porcupine.frame_length * supported_sample_rate / porcupine.sample_rate)
                )
                if audio_stream is None:
                    print("Failed to create audio stream. Retrying in 1 second...")
                    time.sleep(1)
                    continue

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

                # Close the existing stream before creating new one
                # audio_stream.stop_stream()
                # audio_stream.close()
                
                # time.sleep(1.0)  # Increase delay to give more time for audio system to stabilize
                
                try:
                    mic = create_speech_recognizer(default_input_device, supported_sample_rate)
                    if mic is None:
                        raise Exception("Failed to create microphone instance")
                        
                    with mic as source:
                        # print("Adjusting for ambient noise...")
                        # recognizer.adjust_for_ambient_noise(source, duration=1.0)
                        print("Listening...")
                        try:
                            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                            print("Processing speech...")
                            text = recognizer.recognize_google(audio)
                            print(f"Recognized Text: {text}")
                        except sr.WaitTimeoutError:
                            print("No speech detected within timeout period")
                            continue
                        except sr.UnknownValueError:
                            print("Could not understand audio")
                            continue
                        except sr.RequestError as e:
                            print(f"Could not request results; {e}")
                            continue
                    
                except Exception as e:
                    print(f"Error during speech recognition: {str(e)}")


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

                # After speech recognition is complete, reopen the stream for hotword detection
                audio_stream = create_audio_stream(
                    pa,
                    default_input_device,
                    supported_sample_rate,
                    int(porcupine.frame_length * supported_sample_rate / porcupine.sample_rate)
                )
                if audio_stream is None:
                    print("Failed to reopen audio stream. Retrying...")
                    time.sleep(1)
                    continue

        except OSError as e:
            print(f"Audio stream error: {e}")
            # Close the stream if it exists
            if audio_stream is not None:
                try:
                    audio_stream.stop_stream()
                    audio_stream.close()
                except:
                    pass
            audio_stream = None
            time.sleep(1)  # Wait before retrying
            continue
finally:
    if audio_stream is not None:
        audio_stream.stop_stream()
        audio_stream.close()
    porcupine.delete()
    pa.terminate()