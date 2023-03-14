from transformers import pipeline
import voiceToText

cls = pipeline("automatic-speech-recognition")
input = cls("sample1.mp3")
print(input)
