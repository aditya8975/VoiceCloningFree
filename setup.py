"""
VoiceForge Setup Script
Installs all dependencies for speech + singing voice cloning.

Usage:
  python setup.py                    # install everything (recommended)
  python setup.py --check            # show what's installed, install nothing
  python setup.py --speech-only      # only XTTS v2 (+ optional OpenVoice)
  python setup.py --rvc-only         # only RVC/singing pipeline
  python setup.py --skip-heavy       # skip Demucs/preprocessing (lighter install)
  python setup.py --download-openvoice   # download OpenVoice checkpoints
  python setup.py --enable-openvoice     # attempt optional OpenVoice install

Design note: every package is installed in its own try/except. One package
failing to build (most often OpenVoice's fairseq dependency on Windows)
will NOT stop the rest of the install. XTTS v2 is the primary speech engine
and is installed first since it has clean wheels on Windows + Python 3.10/3.11.
"""

import subprocess, sys, os, argparse, shutil
from pathlib import Path

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def pip_one(package: str) -> bool:
    """Install a single package, isolated, so one failure doesn't block others."""
    cmd = f'"{sys.executable}" -m pip install "{package}"'
    print(f"  📦 pip install {package} …", end=" ", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅")
        return True
    else:
        print("❌")
        err_lines = (result.stderr or result.stdout).strip().splitlines()
        for line in err_lines[-5:]:
            print(f"      {line}")
        return False


def pip_many(packages: list, label: str) -> dict:
    print(f"\n📦 Installing {label}…")
    results = {}
    for pkg in packages:
        results[pkg] = pip_one(pkg)
    return results


def run(cmd: str) -> bool:
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def check_pkg(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ── Package groups ──────────────────────────────────────────────────────────
CORE_PACKAGES = [
    "gradio>=4.0",
    "numpy",
    "scipy",
    "librosa",
    "soundfile",
    "ffmpeg-python",
    "tqdm",
    "requests",
]

TORCH_PACKAGES_CPU = ["torch", "torchaudio"]

# XTTS v2 — PRIMARY speech cloning engine.
# IMPORTANT: package name is "coqui-tts", NOT "TTS". The original "TTS" package
# on PyPI is unmaintained and requires compiling native C extensions, which
# needs Microsoft C++ Build Tools on Windows and routinely fails. The
# community fork "coqui-tts" ships prebuilt wheels for Windows/macOS/Linux
# (Python 3.10-3.14) and is a drop-in replacement -- same `from TTS.api import TTS`
# import path, same XTTS v2 model.
XTTS_PACKAGES = ["coqui-tts"]

# OpenVoice v2 — OPTIONAL secondary engine, known to be fragile on Windows
# (pulls in fairseq, which often fails to build without C++ build tools)
OPENVOICE_PACKAGES = ["openvoice-cli", "melo-tts"]

# RVC singing voice conversion.
# IMPORTANT: the PyPI package "rvc-python" is NOT installed via pip here.
# It depends on fairseq, which in turn pins omegaconf<2.1 -- an old omegaconf
# release with invalid PyYAML metadata that pip 24.1+ refuses to resolve,
# causing a permanent ResolutionImpossible error with no clean fix.
# Instead we clone the official RVC WebUI repo, which ships its own pinned,
# tested requirements.txt and is run as a local subprocess server. See
# install_rvc_webui() below.
RVC_SUPPORT_PACKAGES = [
    "faiss-cpu",
    "praat-parselmouth",
    "pyworld",
    "torchcrepe",
]

# Preprocessing (vocal isolation, silence removal, speaker ID).
# IMPORTANT: resemblyzer depends on webrtcvad, which requires a C compiler to
# build on Windows (Microsoft Visual C++ Build Tools) and fails otherwise.
# We install the "webrtcvad-wheels" fork first -- it provides prebuilt wheels
# under the same import name (`import webrtcvad`), so resemblyzer picks it up
# with no compiler needed.
PREPROCESS_PACKAGES = [
    "webrtcvad-wheels",
    "demucs",
    "resemblyzer",
]


def install_ffmpeg():
    print("\n🔧 Checking ffmpeg…")
    if shutil.which("ffmpeg"):
        print("  ✅ ffmpeg already installed")
        return True
    if shutil.which("conda"):
        print("  Installing via conda…")
        return run("conda install -y -c conda-forge ffmpeg")
    if shutil.which("apt-get"):
        print("  Installing via apt-get…")
        return run("sudo apt-get install -y ffmpeg")
    if shutil.which("brew"):
        print("  Installing via Homebrew…")
        return run("brew install ffmpeg")
    print("  ⚠️  Could not auto-install ffmpeg.")
    print("     Windows: download a build from https://www.gyan.dev/ffmpeg/builds/")
    print("              then add the bin/ folder to your PATH and restart your terminal.")
    print("     Linux:   sudo apt install ffmpeg")
    print("     macOS:   brew install ffmpeg")
    return False


def install_rvc_webui():
    """
    Clone the official RVC WebUI repo and install ITS pinned requirements.txt
    (not the broken 'rvc-python' PyPI package). This avoids the fairseq/
    omegaconf/PyYAML ResolutionImpossible conflict entirely, since the repo's
    own requirements file is tested and pinned to compatible versions.
    """
    rvc_dir = ROOT / "rvc_webui"
    print("\n📦 Installing RVC WebUI (singing voice conversion)…")

    if not rvc_dir.exists():
        print("  ⬇️  Cloning RVC-Project/Retrieval-based-Voice-Conversion-WebUI…")
        ok = run(
            f'git clone --depth 1 '
            f'https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git '
            f'"{rvc_dir}"'
        )
        if not ok:
            print("  ❌ git clone failed. Is git installed? (https://git-scm.com/downloads)")
            return False
    else:
        print(f"  ✅ {rvc_dir} already exists, skipping clone")

    req_file = rvc_dir / "requirements.txt"
    if req_file.exists():
        print("  📦 Installing RVC WebUI's own pinned requirements…")
        cmd = f'"{sys.executable}" -m pip install -r "{req_file}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("  ✅ RVC WebUI dependencies installed")
            return True
        else:
            print("  ❌ Some RVC WebUI dependencies failed to install:")
            for line in (result.stderr or result.stdout).strip().splitlines()[-8:]:
                print(f"      {line}")
            print("  ℹ️  RVC may still partially work -- check rvc_webui/ manually if needed.")
            return False
    else:
        print(f"  ❌ requirements.txt not found in {rvc_dir}")
        return False


def download_openvoice_checkpoints():
    """Download OpenVoice v2 checkpoints from HuggingFace (only needed if OpenVoice is installed)."""
    import urllib.request
    ckpt_path = CKPT_DIR / "converter"
    ckpt_path.mkdir(parents=True, exist_ok=True)

    base = "https://huggingface.co/myshell-ai/OpenVoice/resolve/main/checkpoints_v2/"
    files = [
        ("converter/config.json",    "config.json"),
        ("converter/checkpoint.pth", "checkpoint.pth"),
    ]

    ses_path = CKPT_DIR / "base_speakers" / "ses"
    ses_path.mkdir(parents=True, exist_ok=True)
    ses_files = [
        ("base_speakers/ses/en-default.pth", "en-default.pth"),
        ("base_speakers/ses/en-newest.pth",   "en-newest.pth"),
        ("base_speakers/ses/zh.pth",           "zh.pth"),
        ("base_speakers/ses/jp.pth",           "jp.pth"),
    ]

    print("\n⬇️  Downloading OpenVoice v2 checkpoints (~200MB)…")
    all_files = [(base + src, ckpt_path / dst) for src, dst in files]
    all_files += [(base + src, ses_path / dst) for src, dst in ses_files]

    for url, dst in all_files:
        if dst.exists():
            print(f"  ✅ {dst.name} already exists")
            continue
        print(f"  ⬇️  {dst.name}…", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, str(dst))
            print("✅")
        except Exception as e:
            print(f"❌  {e}")

    print("\nCheckpoints saved to:", CKPT_DIR)


def print_status():
    checks = [
        ("gradio",      "Gradio"),
        ("librosa",     "librosa"),
        ("soundfile",   "soundfile"),
        ("torch",       "PyTorch"),
        ("TTS",         "Coqui XTTS v2 (primary speech engine, pkg: coqui-tts)"),
        ("openvoice",   "OpenVoice v2 (optional)"),
        ("melo",        "MeloTTS (optional)"),
        ("demucs",      "Demucs"),
        ("webrtcvad",   "webrtcvad (for Resemblyzer)"),
        ("resemblyzer", "Resemblyzer"),
        ("faiss",       "FAISS"),
        ("parselmouth", "parselmouth"),
        ("pyworld",     "pyworld"),
        ("torchcrepe",  "torchcrepe"),
    ]
    print("\n📋 Package status:")
    for mod, label in checks:
        ok = check_pkg(mod)
        print(f"   {'✅' if ok else '❌'}  {label}")

    has_ffmpeg = bool(shutil.which("ffmpeg"))
    print(f"   {'✅' if has_ffmpeg else '❌'}  FFmpeg (system)")

    has_git = bool(shutil.which("git"))
    print(f"   {'✅' if has_git else '❌'}  git (system, needed for RVC WebUI)")

    rvc_webui_dir = ROOT / "rvc_webui"
    print(f"   {'✅' if rvc_webui_dir.exists() else '❌'}  RVC WebUI cloned")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   ⚡  GPU: not available (CPU mode — XTTS will be slower but works)")
    except Exception:
        print("   ❓  GPU: torch not installed")

    ov_ckpt = CKPT_DIR / "converter" / "checkpoint.pth"
    ov_status = "found" if ov_ckpt.exists() else "(not downloaded — optional)"
    print(f"   {'✅' if ov_ckpt.exists() else 'ℹ️ '}  OpenVoice checkpoints {ov_status}")

    rvc_models_dir = ROOT / "rvc_models"
    rvc_models = list(rvc_models_dir.glob("*.pth")) if rvc_models_dir.exists() else []
    print(f"\n🎵 RVC trained models found: {len(rvc_models)}")
    for m in rvc_models:
        print(f"   - {m.stem}")

    print("\n" + "-" * 60)
    if check_pkg("TTS"):
        print("Speech cloning is READY (XTTS v2). Use Tab 2 in the app.")
    else:
        print("Speech cloning NOT ready. Run: pip install coqui-tts")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="VoiceForge setup")
    parser.add_argument("--check",              action="store_true", help="Show install status only, install nothing")
    parser.add_argument("--speech-only",        action="store_true", help="Install speech cloning only (XTTS + optional OpenVoice)")
    parser.add_argument("--rvc-only",           action="store_true", help="Install RVC/singing pipeline only")
    parser.add_argument("--enable-openvoice",   action="store_true", help="Also attempt OpenVoice v2 install (optional, may fail on Windows)")
    parser.add_argument("--download-openvoice", action="store_true", help="Download OpenVoice checkpoints only")
    parser.add_argument("--skip-heavy",         action="store_true", help="Skip Demucs/preprocessing (lighter install)")
    args = parser.parse_args()

    if args.check:
        print_status()
        return

    if args.download_openvoice:
        download_openvoice_checkpoints()
        return

    print("VoiceForge Setup\n")
    install_ffmpeg()

    pip_many(CORE_PACKAGES, "core packages")
    pip_many(TORCH_PACKAGES_CPU, "PyTorch (CPU build)")

    if not args.rvc_only:
        pip_many(XTTS_PACKAGES, "XTTS v2 (primary speech engine)")
        if args.enable_openvoice:
            print("\nOpenVoice install is optional and can fail on Windows")
            print("(requires a C++ build toolchain for one of its dependencies).")
            print("If it fails below, that's fine -- XTTS v2 still works standalone.")
            results = pip_many(OPENVOICE_PACKAGES, "OpenVoice v2 (optional)")
            if all(results.values()):
                download_openvoice_checkpoints()
            else:
                print("\nOpenVoice install incomplete -- app will use XTTS v2 only. This is fine.")

    if not args.speech_only:
        pip_many(RVC_SUPPORT_PACKAGES, "RVC support libraries (pitch/index tools)")
        if not shutil.which("git"):
            print("\n❌ git is not installed -- cannot clone RVC WebUI.")
            print("   Install git from https://git-scm.com/downloads, then re-run:")
            print("   python setup.py --rvc-only")
        else:
            install_rvc_webui()

    if not args.skip_heavy:
        pip_many(PREPROCESS_PACKAGES, "preprocessing tools (Demucs, Resemblyzer)")

    print("\nSetup complete!\n")
    print("To launch VoiceForge:")
    print("   python app.py")
    print("   python app.py --share   # public Gradio link (for Colab)")
    print()
    print_status()


if __name__ == "__main__":
    main()
