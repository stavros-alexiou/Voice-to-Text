from transformers import pipeline

cls = pipeline("automatic-speech-recognition")
res = cls("sample1.mp3")
print(res)