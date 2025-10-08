import os, queue, time
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv

from openwakeword.model import Model as OWWModel
import openwakeword.utils as oww_utils

from faster_whisper import WhisperModel

load_dotenv()

SR = 16_000
BLOCK = 512                  # ~32 ms
COOLDOWN_S = 1.5
WAKE_ALIASES = [w.strip().lower() for w in os.getenv("WAKEWORD", "victoria|hey jarvis|jarvis").split("|")]
OWW_THRESHOLD = float(os.getenv("OWW_THRESHOLD", "0.35"))
OWW_VAD = float(os.getenv("OWW_VAD", "0.30"))
CONFIRM = os.getenv("CONFIRM_WITH_WHISPER", "1") == "1"
TARGET_MODEL = os.getenv("TARGET_MODEL", "hey_jarvis_v0.1")
MIC_DEVICE = os.getenv("MIC_DEVICE", "").strip()
MIC_DEVICE = int(MIC_DEVICE) if MIC_DEVICE.isdigit() else None

# Descarga modelos por si faltan
oww_utils.download_models()
oww = OWWModel(vad_threshold=OWW_VAD)

# Whisper en CPU sin float16 drama
whisper = WhisperModel("tiny", device="cpu", compute_type="int8")

audio_q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio callback status:", status)
    audio_q.put(indata.copy().reshape(-1))

def contains_wakeword(text: str) -> bool:
    tl = text.lower()
    return any(alias in tl for alias in WAKE_ALIASES)

def confirm_with_whisper(buf: np.ndarray) -> bool:
    segs, _ = whisper.transcribe(buf, vad_filter=True)
    txt = " ".join([s.text for s in segs]).strip()
    if txt:
        print("↳ confirm (whisper):", txt)
    return contains_wakeword(txt)

def speak(msg: str):
    print("Victoria dice:", msg)

def print_model_keys_once():
    # Lanza una pasada “en vacío” para obtener las claves de salida
    dummy = np.zeros(BLOCK, dtype=np.float32)
    scores = oww.predict(dummy) or {}
    if scores:
        keys = ", ".join(scores.keys())
        print(f"[DEBUG] Modelos cargados en OWW: {keys}")

def main():
    print(f"🎧 Victoria escuchando... (alias: {', '.join(WAKE_ALIASES)})  thr={OWW_THRESHOLD}, vad={OWW_VAD}")
    print_model_keys_once()
    last_fire = 0.0
    ring = np.zeros(0, dtype=np.float32)
    last_dbg = time.time()

    with sd.InputStream(device=MIC_DEVICE, samplerate=SR, channels=1,
                        dtype="float32", blocksize=BLOCK, callback=audio_callback):
        while True:
            chunk = audio_q.get()
            ring = np.concatenate([ring, chunk])

            while len(ring) >= BLOCK:
                frame = ring[:BLOCK]
                ring = ring[BLOCK:]
                scores = oww.predict(frame) or {}

                if not scores:
                    continue

                # Mostrar top-3 cada ~0.5s para ver si entra audio
                now = time.time()
                if now - last_dbg > 0.5:
                    top3 = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    dbg = " | ".join([f"{k}={v:.2f}" for k,v in top3])
                    print(f"[DBG] {dbg}")
                    last_dbg = now

                if TARGET_MODEL in scores:
                    label = TARGET_MODEL
                    score = scores[label]
                else:
                    label, score = max(scores.items(), key=lambda kv: kv[1])

                if score >= OWW_THRESHOLD and (now - last_fire) > COOLDOWN_S:
                    print(f"🟢 Wakeword detectado -> {label} (score={score:.2f})")
                    last_fire = now

                    # 1s de audio para confirmación
                    need = SR
                    extra = [frame]
                    while need > 0:
                        nxt = audio_q.get()
                        take = min(need, len(nxt))
                        extra.append(nxt[:take])
                        need -= take
                    buf = np.concatenate(extra).astype(np.float32)

                    if not CONFIRM or confirm_with_whisper(buf):
                        speak("¿Sí?")
                        # TODO: aquí disparas tu pipeline GPT/TTS/LOG
                        time.sleep(0.3)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] Detenido por usuario.")
