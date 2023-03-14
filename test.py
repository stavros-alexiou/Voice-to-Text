import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import speech_recognition as sr
import io
from pydub import AudioSegment

if torch.cuda.is_available():
    device = "cuda:0"  
else:
    "cpu"

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
model = model.to(device)

recognition = sr.Recognizer()

with sr.Microphone(sample_rate = 16000) as source:                                                          # 16 Kbit/second sampling rate
    print("\n\nStart talking...\n")
    while True:
        audio = recognition.listen(source)                                                                  # pyAudio object
        data = io.BytesIO(audio.get_wav_data())                                                             # list of bytes                  
        clip = AudioSegment.from_file(data)                                                                 # NumPy array
        x = torch.FloatTensor(clip.get_array_of_samples())                                                  # tensor

        inputs = processor(x, sampling_rate = 16000, return_tensors='pt', padding='longest').to(device).input_values
        logits = model(inputs).logits
        tokens = torch.argmax(logits, axis = -1)
        text = processor.batch_decode(tokens)                                                               # tokens to strings

        print("You said: ", str(text).lower())

# torch.cuda.is_available()
# torch.cuda.device_count()
# torch.cuda.current_device()
# torch.cuda.device(0)
# torch.cuda.get_device_name(0)



