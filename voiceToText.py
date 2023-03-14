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

with sr.Microphone(sample_rate = 16000) as source:                                                          # 16 Kbit/second sampling rate
    print("\n\nStart talking...\n")
    while True:
        audio = recognition.listen(source)                                                                  # pyAudio object
        data = io.BytesIO(audio.get_wav_data())                                                             # list of bytes                  
        clip = AudioSegment.from_file(data)                                                                 # NumPy array
        x = torch.FloatTensor(clip.get_array_of_samples())                                                  # tensor

        inputs = processor(x, sampling_rate = 16000, return_tensors='pt', padding='longest').input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis = -1)
        text = processor.batch_decode(tokens)                                                               # tokens to strings

        print("You said: ", str(text).lower())


