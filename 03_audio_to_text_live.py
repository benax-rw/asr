import whisper
import warnings
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

# Ignore CPU warning from Whisper
warnings.filterwarnings("ignore", category=UserWarning)

# Load Whisper model once
model = whisper.load_model("base")

# Globals to hold recording state
is_recording = False
audio_data = []

# Parameters
sample_rate = 16000
channels = 1

def callback(indata, frames, time, status):
    global audio_data
    if is_recording:
        audio_data.append(indata.copy())

def start_recording():
    global is_recording, audio_data
    audio_data = []
    is_recording = True
    print("Recording started. Press Enter to stop...")
    with sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback):
        input()  # Wait for user to press Enter
        stop_recording()

def stop_recording():
    global is_recording
    is_recording = False
    print("Recording stopped.")
    process_and_transcribe()

def process_and_transcribe():
    print("Processing audio...")
    full_recording = np.concatenate(audio_data, axis=0)
    temp_filename = "live_input.wav"
    wav.write(temp_filename, sample_rate, full_recording)

    print("Transcribing...")
    result = model.transcribe(temp_filename)
    lyrics = result['text']

    output_textfile = "live_input.txt"
    with open(output_textfile, "w", encoding="utf-8") as f:
        f.write(lyrics)

    print(f"Done! Lyrics saved as '{output_textfile}'")
    print("Transcript:")
    print(lyrics)

if __name__ == "__main__":
    print("Press Enter to start recording...")
    input()
    start_recording()