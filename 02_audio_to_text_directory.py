import os
import whisper
import warnings

#Let's ignore warnings!
warnings.filterwarnings("ignore", category=UserWarning)

# Load Whisper model (base is decent, small is faster)
model = whisper.load_model("base")  # you can use "small" or "medium" for better accuracy

# Path to your folder with mp3 songs
folder_path = "audio_files"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(".mp3"):
        mp3_path = os.path.join(folder_path, filename)
        base_filename = os.path.splitext(filename)[0]
        txt_path = os.path.join(folder_path, base_filename + ".txt")

        print(f"Transcribing {filename}...")
        result = model.transcribe(mp3_path, language='en')  # 'en' for English, auto-detect also works
        lyrics = result['text']

        # Save lyrics to text file
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(lyrics)

        print(f"Lyrics saved as {txt_path}")