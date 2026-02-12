import whisper

model = whisper.load_model("base")
result = model.transcribe("0.wav")
print(result["text"])
