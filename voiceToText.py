import torch
from transformers import AutoProcessor, AutoModelForCTC
import speech_recognition as sr
import io

# Alternative model
# from transformers import AutoProcessor, AutoModelForCTC
# processor = AutoProcessor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
# model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

processor = AutoProcessor.from_pretrained("zuu/automatic-speech-recognition")
model = AutoModelForCTC.from_pretrained("zuu/automatic-speech-recognition")

recognition = sr.Recognizer()

audio = "sample1.mp3"
data = audio.get_wav_data()
clip = AudioSegment.from_file(data)
x = torch.FloatTensor(clip.get_array_of_samples())

inputs = processor(x, sampling_rate = 16000, return_tensors='pt', padding='longest').input_values
logits = model(inputs).logits
tokens = torch.argmax(logits, axis = -1)
text = processor.batch_decode(tokens) 


