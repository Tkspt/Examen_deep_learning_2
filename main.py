# Importer les modules et charger les modèles
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline
import torch
import librosa

# Chargement et prétraitement de l'audio ...
def speech_file_to_array(audio_path):
    speech, _ = librosa.load(audio_path, sr=16000)
    return speech

def transcribe_audio(audio_path):
    speech = speech_file_to_array(audio_path)
    inputs = processor(speech, return_tensors="pt", sampling_rate=16000, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# Analyse de sentiment ...
def analyze_sentiment(text):
    result = sentiment_model(text)
    return result

# Génération du rapport ...
def generate_report(transcription, sentiment):
    report = f"Transcription:\n{transcription}\n\nSentiment Analysis:\n{sentiment}"
    with open("asr_sentiment_report.txt", "w") as file:
        file.write(report)
    print("Rapport généré avec succès!")




model_name = "facebook/wav2vec2-large-xlsr-53-french"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

audio_path = "critiques_audio.mp3"
transcription = transcribe_audio(audio_path)
sentiment = analyze_sentiment(transcription)
generate_report(transcription, sentiment)
