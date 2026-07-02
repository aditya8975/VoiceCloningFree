"""
VoiceForge — Free Voice Cloning Studio
Supports: Speech TTS (OpenVoice v2) + Singing (RVC)
UI: Gradio | Backend: Python | GPU: Optional (Colab compatible)
"""

import os, sys, shutil, subprocess, tempfile, json, time
from pathlib import Path
import gradio as gr
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
SAMPLES_DIR = ROOT / "samples"
MODELS_DIR  = ROOT / "models"
OUTPUT_DIR  = ROOT / "output"
RVC_DIR     = ROOT / "rvc_models"
CKPT_DIR    = ROOT / "checkpoints"

for d in [SAMPLES_DIR, MODELS_DIR, OUTPUT_DIR, RVC_DIR, CKPT_DIR]:
    d.mkdir(exist_ok=True)

# ── Lazy imports (installed at runtime) ────────────────────────────────────
def try_import(name):
    try:
        return __import__(name)
    except ImportError:
        return None

torch      = try_import("torch")
torchaudio = try_import("torchaudio")
librosa    = try_import("librosa")
sf         = try_import("soundfile")

# ═══════════════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def run_cmd(cmd: str) -> tuple[int, str]:
    """Run shell command, return (returncode, combined output)."""
    result = subprocess.run(
        cmd, shell=True, capture_output=True, text=True
    )
    return result.returncode, result.stdout + result.stderr


def check_ffmpeg() -> bool:
    code, _ = run_cmd("ffmpeg -version")
    return code == 0


def get_audio_info(path: str) -> dict:
    """Return duration, sample_rate, channels via ffprobe."""
    cmd = (
        f'ffprobe -v quiet -print_format json -show_streams "{path}"'
    )
    code, out = run_cmd(cmd)
    if code != 0:
        return {}
    try:
        streams = json.loads(out).get("streams", [{}])
        s = streams[0]
        return {
            "duration": float(s.get("duration", 0)),
            "sample_rate": int(s.get("sample_rate", 0)),
            "channels": int(s.get("channels", 1)),
            "codec": s.get("codec_name", "unknown"),
        }
    except Exception:
        return {}


def convert_to_wav(src: str, dst: str, sr: int = 22050, mono: bool = True) -> bool:
    """Convert any audio to WAV using ffmpeg."""
    mono_flag = "-ac 1" if mono else ""
    code, out = run_cmd(
        f'ffmpeg -y -i "{src}" -ar {sr} {mono_flag} -sample_fmt s16 "{dst}"'
    )
    return code == 0


def normalize_audio(path: str, target_lufs: float = -23.0) -> str:
    """Normalize audio loudness, return output path."""
    out = str(OUTPUT_DIR / ("norm_" + Path(path).name))
    code, _ = run_cmd(
        f'ffmpeg -y -i "{path}" '
        f'-af loudnorm=I={target_lufs}:TP=-1.5:LRA=11 "{out}"'
    )
    return out if code == 0 else path


def mix_vocals_with_instrumental(
    vocal_path: str, instrumental_path: str, out_path: str,
    vocal_vol: float = 1.0, instr_vol: float = 0.8
) -> bool:
    """Mix cloned vocals over instrumental track."""
    code, _ = run_cmd(
        f'ffmpeg -y -i "{vocal_path}" -i "{instrumental_path}" '
        f'-filter_complex "'
        f'[0:a]volume={vocal_vol}[v];'
        f'[1:a]volume={instr_vol}[i];'
        f'[v][i]amix=inputs=2:duration=longest:dropout_transition=2[out]'
        f'" -map "[out]" "{out_path}"'
    )
    return code == 0


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 1 — AUDIO PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def preprocess_samples(audio_files, progress=gr.Progress()):
    """
    Pipeline:
      1. Convert to 22050 Hz mono WAV
      2. Demucs vocal isolation (if music is present)
      3. Silero VAD silence removal
      4. Save cleaned files to samples/
    """
    if not audio_files:
        return "❌ No files uploaded.", None

    logs = []
    cleaned_paths = []

    progress(0, desc="Starting preprocessing…")

    for i, f in enumerate(audio_files):
        src = f.name if hasattr(f, "name") else str(f)
        name = Path(src).stem
        progress(i / len(audio_files), desc=f"Processing {name}…")

        # Step 1: Convert to WAV
        wav_path = str(SAMPLES_DIR / f"{name}_raw.wav")
        if not convert_to_wav(src, wav_path):
            logs.append(f"⚠️  {name}: conversion failed, skipping.")
            continue
        info = get_audio_info(wav_path)
        dur = info.get("duration", 0)
        logs.append(f"✅ {name}: {dur:.1f}s  →  converted to WAV")

        # Step 2: Demucs vocal isolation
        demucs_out = str(SAMPLES_DIR / f"{name}_vocals.wav")
        try:
            import demucs.separate
            demucs_dir = SAMPLES_DIR / "demucs_out"
            code, out = run_cmd(
                f'python -m demucs.separate --two-stems=vocals '
                f'-o "{demucs_dir}" "{wav_path}"'
            )
            # Demucs outputs to htdemucs/<stem>/vocals.wav
            vocal_candidate = list(demucs_dir.glob(f"*/{name}_raw/vocals.wav"))
            if vocal_candidate:
                shutil.copy(str(vocal_candidate[0]), demucs_out)
                logs.append(f"   🎤 Demucs: vocals isolated")
            else:
                shutil.copy(wav_path, demucs_out)
                logs.append(f"   ℹ️  Demucs output not found, using raw audio")
        except ImportError:
            shutil.copy(wav_path, demucs_out)
            logs.append(f"   ℹ️  Demucs not installed, skipping isolation")

        # Step 3: Silence removal via ffmpeg silenceremove
        cleaned_path = str(SAMPLES_DIR / f"{name}_clean.wav")
        code, _ = run_cmd(
            f'ffmpeg -y -i "{demucs_out}" '
            f'-af "silenceremove=start_periods=1:start_silence=0.2:'
            f'start_threshold=-50dB:stop_periods=-1:stop_silence=0.3:'
            f'stop_threshold=-50dB" "{cleaned_path}"'
        )
        if code == 0:
            info2 = get_audio_info(cleaned_path)
            dur2 = info2.get("duration", 0)
            logs.append(f"   ✂️  Silence removed: {dur2:.1f}s remaining")
        else:
            shutil.copy(demucs_out, cleaned_path)
            logs.append(f"   ℹ️  Silence removal skipped")

        cleaned_paths.append(cleaned_path)

    progress(1.0, desc="Done!")
    total_dur = sum(
        get_audio_info(p).get("duration", 0) for p in cleaned_paths
    )
    logs.append(f"\n📊 Total clean audio: {total_dur:.1f}s  ({total_dur/60:.1f} min)")

    if total_dur < 30:
        logs.append("⚠️  Less than 30s of audio — quality will be limited.")
    elif total_dur < 300:
        logs.append("👍 30s–5min: good for zero-shot cloning (OpenVoice/XTTS).")
    else:
        logs.append("🚀 5min+: enough for RVC training — use the Training tab!")

    return "\n".join(logs), cleaned_paths


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 2A — SPEECH CLONING  (XTTS v2 primary, OpenVoice v2 optional)
# ═══════════════════════════════════════════════════════════════════════════
#
# Design note: XTTS v2 (Coqui TTS) is the PRIMARY engine. It ships clean,
# prebuilt wheels for Windows + Python 3.10/3.11 and does zero-shot voice
# cloning natively — no separate embedding/training step needed.
#
# OpenVoice v2 + MeloTTS are an OPTIONAL secondary engine. Their dependency
# chain (fairseq, unidic, etc.) frequently fails to build on Windows, so the
# app must work fully on XTTS alone even if OpenVoice was never installed.

_xtts_model = None   # lazy-loaded, cached across calls
_ov_model   = None


def _device() -> str:
    return "cuda" if (torch and torch.cuda.is_available()) else "cpu"


def load_xtts():
    """Load Coqui XTTS v2 once and cache it."""
    global _xtts_model
    if _xtts_model is not None:
        return _xtts_model, f"✅ XTTS v2 already loaded ({_device()})"
    try:
        from TTS.api import TTS as CoquiTTS
    except ImportError:
        return None, (
            "❌ Coqui TTS not installed.\n"
            "Fix: pip install coqui-tts\n"
            "(NOT 'pip install TTS' -- that package is unmaintained and fails "
            "to build on Windows. 'coqui-tts' is the maintained fork with "
            "prebuilt Windows wheels, same import path.)"
        )
    try:
        device = _device()
        model = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        _xtts_model = model
        return model, f"✅ XTTS v2 loaded on {device}"
    except Exception as e:
        return None, f"❌ XTTS v2 failed to load: {e}"


def load_openvoice():
    """Optional secondary engine — only used if explicitly selected and installed."""
    global _ov_model
    if _ov_model is not None:
        return _ov_model, "✅ OpenVoice already loaded"
    try:
        from openvoice.api import ToneColorConverter
    except ImportError:
        return None, (
            "❌ openvoice-cli / melo-tts not installed (this is optional).\n"
            "XTTS v2 will be used instead. To enable OpenVoice:\n"
            "pip install openvoice-cli melo-tts\n"
            "python setup.py --download-openvoice"
        )
    try:
        ckpt_path = CKPT_DIR / "converter"
        if not (ckpt_path / "checkpoint.pth").exists():
            return None, (
                "❌ OpenVoice checkpoint not found.\n"
                "Run: python setup.py --download-openvoice"
            )
        device = _device()
        converter = ToneColorConverter(f"{ckpt_path}/config.json", device=device)
        converter.load_ckpt(f"{ckpt_path}/checkpoint.pth")
        _ov_model = converter
        return converter, f"✅ OpenVoice v2 loaded on {device}"
    except Exception as e:
        return None, f"❌ OpenVoice failed to load: {e}"


LANG_MAP_XTTS = {
    "English": "en", "Hindi": "hi", "Auto-detect": "en",
}


def clone_speech(
    reference_audio,
    text_input: str,
    language: str,
    speed: float,
    engine: str,
    progress=gr.Progress(),
):
    """Zero-shot speech cloning. Engine = 'XTTS v2 (recommended)' or 'OpenVoice v2'."""
    if reference_audio is None:
        return "❌ Upload a reference voice sample first.", None
    if not text_input or not text_input.strip():
        return "❌ Type some text to speak.", None

    ref_path = reference_audio if isinstance(reference_audio, str) else reference_audio.name

    if engine.startswith("OpenVoice"):
        result = _run_openvoice(ref_path, text_input, language, speed, progress)
        if result is not None:
            return result
        progress(0.1, desc="Falling back to XTTS v2…")
        # fall through to XTTS below

    return _run_xtts(ref_path, text_input, language, speed, progress)


def _run_xtts(ref_path, text, language, speed, progress):
    progress(0.1, desc="Loading XTTS v2 (first run downloads ~1.8GB, cached after)…")
    model, msg = load_xtts()
    if model is None:
        return msg, None
    try:
        lang = LANG_MAP_XTTS.get(language, "en")
        out_path = str(OUTPUT_DIR / f"xtts_{int(time.time())}.wav")
        progress(0.5, desc="Synthesising cloned voice (CPU: 30-120s per sentence)…")
        model.tts_to_file(
            text=text,
            speaker_wav=ref_path,
            language=lang,
            file_path=out_path,
            speed=speed,
        )
        progress(1.0, desc="Done!")
        return f"✅ XTTS v2 cloned voice saved:\n{out_path}", out_path
    except Exception as e:
        return f"❌ XTTS synthesis error: {e}", None


def _run_openvoice(ref_path, text, language, speed, progress):
    """Returns (msg, path) on success, or None to signal 'fall back to XTTS'."""
    progress(0.1, desc="Loading OpenVoice v2…")
    converter, msg = load_openvoice()
    if converter is None:
        return None

    try:
        from openvoice import se_extractor
        import melo.api as melo_api

        progress(0.3, desc="Extracting speaker embedding…")
        device = _device()
        target_se, _ = se_extractor.get_se(ref_path, converter, vad=True)

        progress(0.5, desc="Generating base TTS…")
        lang_map = {"English": "EN", "Hindi": "EN-Newest", "Auto-detect": "EN"}
        tts_lang = lang_map.get(language, "EN")
        tts_model = melo_api.TTS(language=tts_lang, device=device)
        spk_id = tts_model.hps.data.spk2id.get(tts_lang, 0)

        tmp_tts = str(OUTPUT_DIR / "tmp_tts.wav")
        tts_model.tts_to_file(text, spk_id, tmp_tts, speed=speed)

        progress(0.75, desc="Applying voice tone…")
        src_se_path = CKPT_DIR / f"base_speakers/ses/{tts_lang.lower()}.pth"
        if not src_se_path.exists():
            src_se_path = CKPT_DIR / "base_speakers/ses/en-default.pth"

        out_path = str(OUTPUT_DIR / f"speech_{int(time.time())}.wav")
        converter.convert(
            audio_src_path=tmp_tts,
            src_se=torch.load(src_se_path, map_location=device),
            tgt_se=target_se,
            output_path=out_path,
            message="@MyShell",
        )
        progress(1.0, desc="Done!")
        return f"✅ OpenVoice v2 cloned voice saved:\n{out_path}", out_path
    except Exception:
        # Any runtime failure -> signal caller to fall back to XTTS
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 2B — SINGING VOICE CLONING  (RVC, via cloned RVC WebUI repo)
# ═══════════════════════════════════════════════════════════════════════════
#
# IMPORTANT: the PyPI package "rvc-python" is NOT used here. It depends on
# fairseq, which pins an old omegaconf release with invalid PyYAML metadata --
# this triggers a permanent ResolutionImpossible error on pip 24.1+ with no
# clean fix. Instead, `python setup.py` clones the official RVC WebUI repo
# into rvc_webui/, which has its own pinned, working requirements.txt.
# Inference here calls into that repo's code directly (added to sys.path).

RVC_WEBUI_DIR = ROOT / "rvc_webui"


def list_rvc_models() -> list[str]:
    models = [p.stem for p in RVC_DIR.glob("*.pth")]
    return models if models else ["(no models trained yet)"]


def rvc_webui_ready() -> tuple[bool, str]:
    if not RVC_WEBUI_DIR.exists():
        return False, (
            "❌ RVC WebUI not found.\n"
            "Run: python setup.py --rvc-only\n"
            "(requires git to be installed)"
        )
    if not (RVC_WEBUI_DIR / "infer").exists() and not (RVC_WEBUI_DIR / "infer-web.py").exists():
        return False, "❌ RVC WebUI folder looks incomplete. Delete rvc_webui/ and re-run setup.py --rvc-only"
    return True, "✅ RVC WebUI found"


def run_rvc_inference(
    input_audio,
    model_name: str,
    pitch_shift: int,
    index_rate: float,
    protect: float,
    progress=gr.Progress(),
):
    """
    Run RVC inference on a vocal/melody audio file via the cloned RVC WebUI repo.
    Converts the singing voice to the trained target voice.
    """
    if input_audio is None:
        return "❌ Upload a vocal/melody track first.", None
    if model_name == "(no models trained yet)" or not model_name:
        return "❌ Train an RVC model first (see Training tab).", None

    ready, msg = rvc_webui_ready()
    if not ready:
        return msg, None

    src = input_audio if isinstance(input_audio, str) else input_audio.name
    model_path = RVC_DIR / f"{model_name}.pth"
    index_path = RVC_DIR / f"{model_name}.index"

    if not model_path.exists():
        return f"❌ Model file not found: {model_path}", None

    progress(0.1, desc="Loading RVC WebUI inference modules…")
    try:
        if str(RVC_WEBUI_DIR) not in sys.path:
            sys.path.insert(0, str(RVC_WEBUI_DIR))

        from infer.modules.vc.modules import VC
        from configs.config import Config

        progress(0.3, desc="Loading model…")
        config = Config()
        vc = VC(config)
        vc.get_vc(str(model_path))

        progress(0.5, desc="Converting voice…")
        wav_in = str(SAMPLES_DIR / "rvc_input.wav")
        convert_to_wav(src, wav_in, sr=40000, mono=True)

        tgt_sr, audio_opt, times, _ = vc.vc_single(
            0,                                          # speaker id
            wav_in,                                      # input audio
            pitch_shift,                                  # pitch shift semitones
            None,                                         # f0 file (auto)
            "rmvpe",                                       # pitch algorithm
            str(index_path) if index_path.exists() else "",  # feature index
            None,                                          # big_npy (unused)
            index_rate,                                     # index rate
            3,                                              # filter radius
            0,                                              # resample sr (0 = no resample)
            0.25,                                           # RMS mix rate
            protect,                                        # consonant protect
        )

        progress(0.9, desc="Saving output…")
        out_path = str(OUTPUT_DIR / f"rvc_{model_name}_{int(time.time())}.wav")
        sf.write(out_path, audio_opt, tgt_sr)

        progress(1.0)
        info = get_audio_info(out_path)
        return (
            f"✅ RVC conversion done!\n"
            f"Duration: {info.get('duration', 0):.1f}s\n"
            f"Saved: {out_path}"
        ), out_path

    except ImportError as e:
        return (
            f"❌ RVC WebUI import failed: {e}\n"
            "Run: python setup.py --rvc-only\n"
            "Make sure git is installed and the clone completed successfully.", None
        )
    except Exception as e:
        return f"❌ RVC error: {e}", None


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 3 — FINAL MIX
# ═══════════════════════════════════════════════════════════════════════════

def final_mix(
    vocal_audio,
    instrumental_audio,
    vocal_vol: float,
    instr_vol: float,
    normalize: bool,
    export_format: str,
    progress=gr.Progress(),
):
    """Mix cloned vocals with an instrumental track."""
    if vocal_audio is None:
        return "❌ No vocal track provided.", None

    v_path = vocal_audio if isinstance(vocal_audio, str) else vocal_audio.name
    progress(0.2, desc="Normalizing vocal…")

    if normalize:
        v_path = normalize_audio(v_path)

    out_stem = f"final_mix_{int(time.time())}"

    if instrumental_audio is not None:
        i_path = instrumental_audio if isinstance(instrumental_audio, str) else instrumental_audio.name
        progress(0.5, desc="Mixing tracks…")
        out_path = str(OUTPUT_DIR / f"{out_stem}.wav")
        ok = mix_vocals_with_instrumental(v_path, i_path, out_path, vocal_vol, instr_vol)
        if not ok:
            return "❌ FFmpeg mixing failed. Check ffmpeg installation.", None
    else:
        out_path = v_path  # Just export vocals as-is

    # Convert format if needed
    if export_format != "WAV":
        ext = export_format.lower()
        final_path = str(OUTPUT_DIR / f"{out_stem}.{ext}")
        bitrate = "320k" if ext == "mp3" else "256k"
        code, _ = run_cmd(
            f'ffmpeg -y -i "{out_path}" -b:a {bitrate} "{final_path}"'
        )
        if code == 0:
            out_path = final_path

    progress(1.0)
    info = get_audio_info(out_path)
    return (
        f"✅ Mix complete!\n"
        f"Duration: {info.get('duration', 0):.1f}s\n"
        f"Saved: {out_path}"
    ), out_path


# ═══════════════════════════════════════════════════════════════════════════
#  MODULE 4 — SYSTEM STATUS
# ═══════════════════════════════════════════════════════════════════════════

def get_system_status():
    lines = ["## 🖥️ System Status\n"]

    # Python
    lines.append(f"**Python:** {sys.version.split()[0]}")

    # PyTorch
    if torch:
        lines.append(f"**PyTorch:** {torch.__version__}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            lines.append(f"**GPU:** {name} ({mem}GB VRAM) ✅")
        else:
            lines.append("**GPU:** Not available — running on CPU")
    else:
        lines.append("**PyTorch:** ❌ Not installed")

    # FFmpeg
    lines.append(f"**FFmpeg:** {'✅ Available' if check_ffmpeg() else '❌ Not found'}")

    # Packages
    pkg_checks = [
        ("TTS",         "Coqui XTTS v2 (primary speech engine, pkg: coqui-tts)"),
        ("openvoice",   "OpenVoice v2 (optional)"),
        ("demucs",      "Demucs"),
        ("faiss",       "FAISS (for RVC)"),
        ("melo",        "MeloTTS (optional)"),
        ("librosa",     "librosa"),
        ("soundfile",   "soundfile"),
    ]
    lines.append("\n**Installed packages:**")
    for mod, label in pkg_checks:
        installed = try_import(mod) is not None
        lines.append(f"  {'✅' if installed else '❌'} {label}")

    # RVC WebUI clone status
    rvc_ready, rvc_msg = rvc_webui_ready()
    lines.append(f"\n**RVC WebUI:** {'✅ Ready' if rvc_ready else '❌ Not set up — run: python setup.py --rvc-only'}")

    # RVC models
    models = list_rvc_models()
    lines.append(f"**RVC models:** {', '.join(models)}")

    # Samples
    sample_files = list(SAMPLES_DIR.glob("*_clean.wav"))
    lines.append(f"**Cleaned samples:** {len(sample_files)} file(s)")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════

CSS = """
.gradio-container { max-width: 900px !important; margin: auto; }
.tab-header { font-size: 1.1em; font-weight: 600; }
footer { display: none !important; }
"""

with gr.Blocks(title="VoiceForge", css=CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
# 🎤 VoiceForge — Free Voice Cloning Studio
**Speech cloning** (OpenVoice v2 / XTTS v2) + **Singing voice conversion** (RVC)  
Upload voice samples → clone any voice → export songs or speech.
""")

    with gr.Tabs():

        # ── TAB 1: PREPROCESSING ──────────────────────────────────────────
        with gr.TabItem("📂 1. Preprocess Samples"):
            gr.Markdown("""
**Upload your raw voice samples here.** Any format (MP3, WAV, M4A, FLAC).  
The pipeline will: isolate vocals → remove silence → save clean audio.  
*Minimum: 30 seconds. Best: 5+ minutes for RVC training.*
""")
            with gr.Row():
                with gr.Column():
                    sample_upload = gr.File(
                        label="Upload voice samples (multiple files OK)",
                        file_count="multiple",
                        file_types=["audio", ".mp3", ".wav", ".m4a", ".flac", ".ogg"],
                    )
                    preprocess_btn = gr.Button("🔧 Preprocess Samples", variant="primary")
                with gr.Column():
                    preprocess_log   = gr.Textbox(label="Preprocessing log", lines=12, interactive=False)
                    cleaned_files_state = gr.State([])

            preprocess_btn.click(
                preprocess_samples,
                inputs=[sample_upload],
                outputs=[preprocess_log, cleaned_files_state],
            )

        # ── TAB 2: SPEECH CLONING ─────────────────────────────────────────
        with gr.TabItem("🗣️ 2. Clone Speech"):
            gr.Markdown("""
**Zero-shot speech cloning** — no training needed.  
Upload a 6–60 second reference clip, type your text, get the cloned voice.  
Default engine is **XTTS v2** (installs cleanly on Windows). OpenVoice v2 is
available as an optional alternate if you've installed it separately.
""")
            with gr.Row():
                with gr.Column():
                    speech_ref     = gr.Audio(label="Reference voice (6s minimum)", type="filepath")
                    speech_text    = gr.Textbox(
                        label="Text to speak",
                        placeholder="Enter the words you want spoken in the cloned voice…",
                        lines=4
                    )
                    speech_engine  = gr.Dropdown(
                        label="Engine",
                        choices=["XTTS v2 (recommended)", "OpenVoice v2 (optional, requires separate install)"],
                        value="XTTS v2 (recommended)"
                    )
                    speech_lang    = gr.Dropdown(
                        label="Language",
                        choices=["English", "Hindi", "Auto-detect"],
                        value="English"
                    )
                    speech_speed   = gr.Slider(0.6, 1.5, value=1.0, step=0.05, label="Speed")
                    speech_btn     = gr.Button("🎙️ Clone Voice", variant="primary")
                with gr.Column():
                    speech_log     = gr.Textbox(label="Status", lines=5, interactive=False)
                    speech_output  = gr.Audio(label="Cloned speech output", type="filepath")

            speech_btn.click(
                clone_speech,
                inputs=[speech_ref, speech_text, speech_lang, speech_speed, speech_engine],
                outputs=[speech_log, speech_output],
            )

        # ── TAB 3: SINGING (RVC) ──────────────────────────────────────────
        with gr.TabItem("🎵 3. Clone Singing (RVC)"):
            gr.Markdown("""
**Singing voice conversion** — convert any vocal to your trained target voice.  
Upload a melody/acapella → RVC transforms it into the cloned voice.  
*Train your model first on Colab (see Training tab), then import the `.pth` file.*
""")
            with gr.Row():
                with gr.Column():
                    rvc_input      = gr.Audio(label="Input vocal / melody", type="filepath")
                    rvc_model_dd   = gr.Dropdown(
                        label="RVC model",
                        choices=list_rvc_models(),
                        value=list_rvc_models()[0]
                    )
                    rvc_refresh_btn = gr.Button("🔄 Refresh model list")
                    pitch_shift    = gr.Slider(-12, 12, value=0, step=1, label="Pitch shift (semitones)")
                    index_rate     = gr.Slider(0.0, 1.0, value=0.75, step=0.05, label="Index rate (higher = more faithful to voice)")
                    protect        = gr.Slider(0.0, 0.5, value=0.33, step=0.01, label="Consonant protect (prevent over-pitch)")
                    rvc_btn        = gr.Button("🎤 Convert Singing Voice", variant="primary")
                with gr.Column():
                    rvc_log        = gr.Textbox(label="Status", lines=5, interactive=False)
                    rvc_output     = gr.Audio(label="RVC output", type="filepath")

            rvc_refresh_btn.click(
                lambda: gr.Dropdown(choices=list_rvc_models(), value=list_rvc_models()[0]),
                outputs=[rvc_model_dd]
            )
            rvc_btn.click(
                run_rvc_inference,
                inputs=[rvc_input, rvc_model_dd, pitch_shift, index_rate, protect],
                outputs=[rvc_log, rvc_output],
            )

        # ── TAB 4: FINAL MIX ──────────────────────────────────────────────
        with gr.TabItem("🎚️ 4. Final Mix"):
            gr.Markdown("""
**Mix your cloned vocals with an instrumental track.**  
Upload the vocal output from Tab 2 or 3, add a beat/instrumental, adjust volumes, export.
""")
            with gr.Row():
                with gr.Column():
                    mix_vocal      = gr.Audio(label="Cloned vocals (from Tab 2 or 3)", type="filepath")
                    mix_instr      = gr.Audio(label="Instrumental / beat (optional)", type="filepath")
                    vocal_vol      = gr.Slider(0.0, 2.0, value=1.0, step=0.05, label="Vocal volume")
                    instr_vol      = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Instrumental volume")
                    normalize_chk  = gr.Checkbox(label="Normalize loudness (EBU R128)", value=True)
                    export_fmt     = gr.Radio(["WAV", "MP3", "FLAC"], value="WAV", label="Export format")
                    mix_btn        = gr.Button("🎛️ Export Final Mix", variant="primary")
                with gr.Column():
                    mix_log        = gr.Textbox(label="Status", lines=5, interactive=False)
                    mix_output     = gr.Audio(label="Final mix", type="filepath")

            mix_btn.click(
                final_mix,
                inputs=[mix_vocal, mix_instr, vocal_vol, instr_vol, normalize_chk, export_fmt],
                outputs=[mix_log, mix_output],
            )

        # ── TAB 5: COLAB TRAINING GUIDE ───────────────────────────────────
        with gr.TabItem("🚀 5. Training (Colab Guide)"):
            gr.Markdown("""
## Training an RVC model on Google Colab (free GPU)

RVC training requires a GPU. Use Colab's free T4 GPU — it's enough.

### Step-by-step

**1. Open this Colab notebook:**  
`https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/Colab_train.ipynb`

**2. Mount your Google Drive** (for saving the trained model)

**3. Upload your cleaned samples**  
- From Tab 1, download your `*_clean.wav` files from the `samples/` folder  
- Upload them to a folder in Drive, e.g. `MyDrive/rvc_samples/`

**4. Run the notebook cells in order:**
```
Cell 1: Install dependencies
Cell 2: Set your speaker name (e.g. "artist_voice")
Cell 3: Point to your samples folder
Cell 4: Set epochs = 100-200 (more = better, but slower)
Cell 5: Train!  (30–60 min on T4 for 5min of audio)
Cell 6: Export .pth and .index files to Drive
```

**5. Download your model files** from Drive:
- `artist_voice.pth` → your RVC model
- `artist_voice.index` → feature index (improves similarity)

**6. Place both files in this project's `rvc_models/` folder**

**7. Hit "Refresh model list" in Tab 3** — your model will appear!

---

### Quick RVC settings guide

| Setting | Recommended | Effect |
|---|---|---|
| Pitch shift | 0 | Match original key; adjust ±1-2 for gender |
| Index rate | 0.75 | How much to force target voice features |
| Protect | 0.33 | Prevents consonant artifacts |

---

### Alternative: RVC WebUI locally (if you have NVIDIA GPU)
```bash
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
cd Retrieval-based-Voice-Conversion-WebUI
pip install -r requirements.txt
python infer-web.py
```
This opens a full browser-based training + inference UI.
""")

        # ── TAB 6: SYSTEM STATUS ──────────────────────────────────────────
        with gr.TabItem("⚙️ System Status"):
            with gr.Row():
                status_btn     = gr.Button("🔍 Check System", variant="secondary")
            status_out         = gr.Markdown("*Click Check System to run diagnostics*")
            status_btn.click(get_system_status, outputs=[status_out])

    gr.Markdown("""
---
**VoiceForge** | Built on: OpenVoice v2 · Coqui XTTS v2 · RVC · Demucs · FFmpeg · Gradio  
*All processing is local — no cloud API keys needed.*
""")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("\n🎤 VoiceForge starting…")
    print(f"   FFmpeg: {'✅' if check_ffmpeg() else '❌ install ffmpeg!'}")
    print(f"   GPU:    {'✅ ' + torch.cuda.get_device_name(0) if (torch and torch.cuda.is_available()) else '⚡ CPU mode'}")
    print(f"   Port:   {args.port}")
    if args.share:
        print("   Share:  public link will be generated")
    print()

    demo.launch(
        server_port=args.port,
        share=args.share,
        inbrowser=True,
    )
