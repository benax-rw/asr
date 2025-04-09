from faster_whisper import WhisperModel

# =============================
# 1. INPUT
# =============================
# Specify the input audio file
input_audiofile = "sample1.mp3"

# =============================
# 2. LOAD MODEL (Sequence Modeling Engine)
# =============================
# Load the Faster-Whisper model with optimized compute_type for CPU (no FP16 warning)
# compute_type options: "int8", "int8_float16", "float16", "float32"
model = WhisperModel("base", compute_type="float32")

# =============================
# 3. FULL PIPELINE: Preprocessing + Sequence Modeling + Decoding
# =============================
# Transcribe the audio file:
# - Preprocessing (audio loading, resampling) is handled automatically
# - Sequence modeling is done using optimized CPU inference
# - Decoding converts audio into readable text
print(f"Transcribing '{input_audiofile}'...")
segments, info = model.transcribe(input_audiofile, language="en")

# Combine all segments into a single string
lyrics = ""
for segment in segments:
    lyrics += segment.text.strip() + "\n"

# =============================
# 4. OUTPUT
# =============================
# Specify the output text filename
output_textfile = input_audiofile.replace(".mp3", ".txt")
# Save the transcribed text to a .txt file
with open(output_textfile, "w", encoding="utf-8") as f:
    f.write(lyrics)

print(f"Transcription complete. Lyrics saved to '{output_textfile}'")