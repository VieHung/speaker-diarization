from pyannote.audio import Pipeline
import torch

# Tải về và lưu vào thư mục 'my_local_model'
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token="HF_TOKEN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline.to(device)

diarization = pipeline("audio_file.wav")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"Bắt đầu: {turn.start:.1f}s | Kết thúc: {turn.end:.1f}s | Người nói: {speaker}")
