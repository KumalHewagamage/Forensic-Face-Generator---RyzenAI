import os
import torchaudio
import time
from transformers import WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import PretrainedConfig

model_name = 'openai/whisper-small'
model_path = ''
processor = WhisperProcessor.from_pretrained(model_name)


# Path to the actual audio file
audio_file_path = 'testaudio.wav'  # Update this with the path to your audio file

# Load the audio file
waveform, sample_rate = torchaudio.load(audio_file_path)


# Load ONNX model
sessions = ORTModelForSpeechSeq2Seq.load_model(
            os.path.join(model_path, 'encoder_model.onnx'),
            os.path.join(model_path, 'decoder_model.onnx'),
            os.path.join(model_path, 'decoder_with_past_model.onnx'),
            os.path.join(model_path, 'decoder_with_past_model.onnx'),
providers=['VitisAIExecutionProvider'], provider_options=[{"config_file":vaip_config}]))
model_config = PretrainedConfig.from_pretrained(model_name)
model = ORTModelForSpeechSeq2Seq(sessions[0], sessions[1], model_config, model_path, sessions[2])

# Process the audio
input_features = processor(waveform.numpy(), sampling_rate=sample_rate, return_tensors="pt").input_features

# Measure time taken for generation
start_time = time.time()
predicted_ids = model.generate(input_features)[0]
end_time = time.time()
generation_time = end_time - start_time

transcription = processor.decode(predicted_ids)
prediction = processor.tokenizer.normalize(transcription)

# Calculate token speed
num_tokens = len(predicted_ids)
token_speed = num_tokens / generation_time

print("Transcription:", transcription)
print("Normalized Prediction:", prediction)
print("Time taken to generate:", generation_time, "seconds")
print("Token Speed:", token_speed, "tokens per second")
