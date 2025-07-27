import os
import pickle
import subprocess
from io import BytesIO
from flask import Flask, request, jsonify, abort
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
import requests
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import lyricsgenius

# ────────── 설정 ──────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Audd API
AUDD_API_KEY = "c2acb8c40d20a8916566c4e5997ab2cc"
# Spotify API
SPOTIFY_CLIENT_ID = "04df9d7a817d4709a27eee2e1ecfb2f2"
SPOTIFY_CLIENT_SECRET = "d36b326fc5df4a97b3ba1a96f13280a2"
# Genius API (lyricsgenius)
GENIUS_CLIENT_ID     = "h51A9KZlo5S37m7wD5QCbMTkbIGnrUb-Uh-YgnBKFgD6oyZcM2kPrPLA3Nh5ekSg"
GENIUS_CLIENT_SECRET = "utv2XRZPQOlHwgLBE0mBRHt_8iFXoxmrQ-VpvWz120a8nzsZQKVNXgWBddnrMHPkNv4C_mJXAG24m2Ww8n8X_A"
GENIUS_ACCESS_TOKEN  = "jcRZgkpiSxJCy0epxHzm1Hs4jGAKz6GXk71kJWmXGNm0Pzg0nh_SnuvE2NcrVkL8"

# Flask 앱 생성 및 CORS 허용
app = Flask(__name__)
CORS(app)

# ────────── 라이브러리 초기화 ──────────
# Spotify client
sp_oauth = SpotifyClientCredentials(
    client_id=SPOTIFY_CLIENT_ID,
    client_secret=SPOTIFY_CLIENT_SECRET
)
# Genius client
genius = lyricsgenius.Genius(
    access_token=GENIUS_ACCESS_TOKEN,
    timeout=15,
    retries=3
)
genius.verbose = False  # 콘솔 출력 최소화

# ────────── 헬퍼 함수 ──────────
def convert_wav_to_mp3(wav: str, mp3: str):
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav, mp3],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

def recognize_with_audd(mp3: str) -> dict:
    url = "https://api.audd.io/"
    with open(mp3, "rb") as f:
        files = {"file": f}
        data = {"api_token": AUDD_API_KEY, "return": "spotify"}
        r = requests.post(url, data=data, files=files)
    return r.json().get("result", {}) if r.status_code == 200 else {}

def search_spotify_cover(artist: str, title: str) -> dict:
    sp = spotipy.Spotify(auth_manager=sp_oauth)
    for q in (f"{title} {artist}", f"{artist} {title}"):
        res = sp.search(q=q, limit=1, type="track")
        items = res.get("tracks", {}).get("items", [])
        if items:
            t = items[0]
            return {
                "title":     t["name"],
                "artist":    t["artists"][0]["name"],
                "cover_url": t["album"]["images"][0]["url"]
            }
    return {}

def fetch_lyrics(artist: str, title: str) -> str:
    try:
        song = genius.search_song(title=title, artist=artist)
        if song and song.lyrics:
            return song.lyrics.strip()
    except Exception as e:
        print(f"[ERROR] Genius lyrics fetch failed: {e}")
    return ""

# ────────── 레이블 인코더 & 모델 로드 ──────────
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
num_labels = len(label_encoder.classes_)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")
bert_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-multilingual-cased",
    num_labels=num_labels
).to(DEVICE)
bert_model.load_state_dict(torch.load("bert_model_re.pt", map_location=DEVICE), strict=False)
bert_model.eval()

class CNNClassifier(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        backbone = models.resnet50(pretrained=False)
        in_feats = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Identity(),
            nn.Linear(in_feats, num_labels)
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

cnn_model = CNNClassifier(num_labels).to(DEVICE)
cnn_model.load_state_dict(torch.load("cnn_model_res50ver3.pt", map_location=DEVICE), strict=True)
cnn_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ────────── 엔드포인트: 텍스트 예측 ──────────
@app.route("/predict/text", methods=["POST"])
def route_predict_text():
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        abort(400, description="No text provided")

    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(DEVICE)
    with torch.no_grad():
        logits = bert_model(**enc).logits
    idx = logits.argmax(dim=1).item()
    label = label_encoder.inverse_transform([idx])[0]
    artist, title = label.split(" - ", 1)

    info   = search_spotify_cover(artist, title)
    lyrics = fetch_lyrics(info.get("artist", artist), info.get("title", title))

    return jsonify({
        "title":     info.get("title", title),
        "artist":    info.get("artist", artist),
        "cover_url": info.get("cover_url", ""),
        "lyrics":    lyrics
    })


# ────────── 엔드포인트: 이미지 예측 ──────────
@app.route("/predict_image", methods=["POST"])
def route_predict_image():
    print(f"[LOG] /predict/image endpoint called. Content-Length: {request.content_length}")
    file = request.files.get("file")
    if not file:
        abort(400, description="No file provided")
    try:
        data = file.read()
        buffer = BytesIO(data)
        img = Image.open(buffer).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        print(f"[DEBUG] image load failed: {e}")
        abort(400, description="Invalid image file")
    tensor = image_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = cnn_model(tensor)
    idx = logits.argmax(dim=1).item()
    label = label_encoder.inverse_transform([idx])[0]
    artist, title = label.split(" - ", 1)
    info = search_spotify_cover(artist, title)
    return jsonify({
        "title": info.get("title", title),
        "artist": info.get("artist", artist),
        "cover_url": info.get("cover_url", "")
    })

# ────────── 엔드포인트: 오디오 예측 ──────────
@app.route("/predict/audio", methods=["POST"])
@app.route("/predict_audio", methods=["POST"])
def route_predict_audio():
    file = request.files.get("file")
    if not file:
        abort(400, description="No file provided")

    tmp_wav = "temp_audio.wav"
    tmp_mp3 = "temp_audio.mp3"
    file.save(tmp_wav)
    convert_wav_to_mp3(tmp_wav, tmp_mp3)
    result = recognize_with_audd(tmp_mp3)
    os.remove(tmp_wav)
    os.remove(tmp_mp3)

    if result:
        artist = result.get("artist", "")
        title  = result.get("title", "")
        info   = search_spotify_cover(artist, title)
        lyrics = fetch_lyrics(info.get("artist", artist), info.get("title", title))
        return jsonify({
            "title":     info.get("title", title),
            "artist":    info.get("artist", artist),
            "cover_url": info.get("cover_url", ""),
            "lyrics":    lyrics
        })

    return jsonify({
        "title":     "인식 실패",
        "artist":    "",
        "cover_url": "",
        "lyrics":    ""
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)