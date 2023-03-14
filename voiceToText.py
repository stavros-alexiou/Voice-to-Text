from transformers import pipeline

def voiceToText(inputSample):
    cls = pipeline("zuu/automatic-speech-recognition")
    res = cls(inputSample)
    print(res)

# from transformers import AutoProcessor, AutoModelForCTC

# processor = AutoProcessor.from_pretrained("zuu/automatic-speech-recognition")

# model = AutoModelForCTC.from_pretrained("zuu/automatic-speech-recognition")

# res = model("sample1.mp3")
# print(res)