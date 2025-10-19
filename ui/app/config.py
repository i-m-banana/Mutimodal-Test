"""Global configuration, environment flags, and shared utilities for the UI app."""

from __future__ import annotations

import csv
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.chdir(BASE_DIR)

# ---------------------------------------------------------------------------
# Optional dependencies (configured exactly as the legacy entry point)
# ---------------------------------------------------------------------------
try:
    from ..services import backend_proxy

    HAS_MULTIMODAL = True
    multidata_start_collection = backend_proxy.multimodal_start_collection
    multidata_stop_collection = backend_proxy.multimodal_stop_collection
    multidata_get_snapshot = backend_proxy.multimodal_get_snapshot
    
    # EEG服务代理
    HAS_EEG_BACKEND = True
    eeg_start_collection = backend_proxy.eeg_start_collection
    eeg_stop_collection = backend_proxy.eeg_stop_collection
    eeg_get_snapshot = backend_proxy.eeg_get_snapshot
    eeg_get_file_paths = backend_proxy.eeg_get_file_paths
    
except ImportError:
    HAS_MULTIMODAL = False
    HAS_EEG_BACKEND = False

    def multidata_start_collection(*args, **kwargs):  # type: ignore[return-type]
        raise RuntimeError("多模态采集模块不可用,请检查后端是否运行。")

    def multidata_stop_collection(*args, **kwargs):
        return None

    def multidata_get_snapshot(*args, **kwargs):  # type: ignore[return-type]
        return {"status": "idle"}
    
    def eeg_start_collection(*args, **kwargs):  # type: ignore[return-type]
        raise RuntimeError("EEG采集模块不可用,请检查后端是否运行。")
    
    def eeg_stop_collection(*args, **kwargs):
        return None
    
    def eeg_get_snapshot(*args, **kwargs):  # type: ignore[return-type]
        return {"status": "idle", "latest_sample": None}
    
    def eeg_get_file_paths(*args, **kwargs):  # type: ignore[return-type]
        return {}

from ..services.av_service import (
    start_collection as av_start_collection,
    stop_collection as av_stop_collection,
    start_recording as av_start_recording,
    stop_recording as av_stop_recording,
    get_current_frame as av_get_current_frame,
    get_audio_paths as av_get_audio_paths,
    get_video_paths as av_get_video_paths,
    get_current_audio_level as av_get_current_audio_level,
)

try:
    from ..services import backend_proxy

    HAS_BP_BACKEND = True
    bp_start_measurement = backend_proxy.bp_start_measurement
    bp_stop_measurement = backend_proxy.bp_stop_measurement
    bp_get_snapshot = backend_proxy.bp_get_snapshot
    bp_get_status = backend_proxy.bp_get_status

except ImportError:
    HAS_BP_BACKEND = False

    def bp_start_measurement(*args, **kwargs):  # type: ignore[return-type]
        raise RuntimeError("血压后端服务不可用，请检查后端是否运行。")

    def bp_stop_measurement():  # type: ignore[return-type]
        raise RuntimeError("血压后端服务不可用，请检查后端是否运行。")

    def bp_get_snapshot():  # type: ignore[return-type]
        return {"status": "idle", "latest": None}

    def bp_get_status():  # type: ignore[return-type]
        return {"running": False, "available_ports": []}

HAS_MAIBOBO_BACKEND = HAS_BP_BACKEND

matplotlib.use("Qt5Agg")

try:
    from ..services.speech_recognition_service import (
        add_audio_for_recognition,
        get_recognition_results,
        clear_recognition_results,
        stop_recognition,
    )

    HAS_SPEECH_RECOGNITION = True
except ImportError:
    HAS_SPEECH_RECOGNITION = False

    def add_audio_for_recognition(*args, **kwargs):  # type: ignore[return-type]
        raise RuntimeError("语音识别功能不可用，请确保 tools.py 和 faster-whisper 已正确安装")

    def get_recognition_results(*args, **kwargs):
        return []

    def clear_recognition_results(*args, **kwargs):
        return None

    def stop_recognition(*args, **kwargs):
        return None

# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _extract_mode_override(argv: list[str]) -> str | None:
    cleaned: list[str] = []
    mode: str | None = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in {"--debug", "-d"}:
            mode = "debug"
            i += 1
            continue
        if arg in {"--normal"}:
            mode = "normal"
            i += 1
            continue
        if arg in {"--mode", "-m"}:
            if i + 1 < len(argv):
                mode = argv[i + 1].lower()
                i += 2
                continue
            i += 1
            continue
        if arg.startswith("--mode="):
            mode = arg.split("=", 1)[1].lower()
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    argv[:] = cleaned
    if mode not in {"debug", "normal"}:
        return None
    return mode


def _apply_mode_override(mode: str | None) -> None:
    if mode == "debug":
        os.environ["UI_DEBUG_MODE"] = "1"
    elif mode == "normal":
        os.environ["UI_DEBUG_MODE"] = "0"
        for key in ("UI_FORCE_SIMULATION", "UI_MULTIMODAL_SIMULATION", "UI_EEG_SIMULATION", "UI_BP_SIMULATION"):
            os.environ.pop(key, None)
    else:
        if "UI_DEBUG_MODE" not in os.environ:
            forced = os.getenv("UI_FORCE_SIMULATION", "").strip().lower() in {"1", "true", "yes", "on"}
            os.environ["UI_DEBUG_MODE"] = "1" if forced else "0"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.getenv(name)
    if value is None:
        return default
    stripped = value.strip()
    if not stripped:
        return default
    try:
        return int(stripped)
    except ValueError:
        return default


_MODE_OVERRIDE = _extract_mode_override(sys.argv)
_apply_mode_override(_MODE_OVERRIDE)

DEBUG_MODE = env_flag("UI_DEBUG_MODE")
FORCE_SIMULATION = env_flag("UI_FORCE_SIMULATION")
BP_SIMULATION = FORCE_SIMULATION or env_flag("UI_BP_SIMULATION")
SKIP_DATABASE = env_flag("UI_SKIP_DATABASE")
APP_MODE = "debug" if DEBUG_MODE else "normal"

_bp_port_raw = os.getenv("UI_BP_PORT", "COM8")
BP_PORT: Optional[str] = _bp_port_raw.strip() if _bp_port_raw else None

DEFAULT_CAMERA_INDEX = env_int("UI_CAMERA_INDEX", 1)
DEFAULT_AUDIO_DEVICE_INDEX = env_int("UI_AUDIO_DEVICE_INDEX", None)
ACTIVE_CAMERA_INDEX: int = DEFAULT_CAMERA_INDEX if DEFAULT_CAMERA_INDEX is not None else 0
ACTIVE_AUDIO_DEVICE_INDEX: Optional[int] = DEFAULT_AUDIO_DEVICE_INDEX

DB_HOST = os.getenv("UI_DB_HOST", "localhost")
DB_USER = os.getenv("UI_DB_USER", "root")
DB_PASSWORD = os.getenv("UI_DB_PASSWORD", "123456")
DB_NAME = os.getenv("UI_DB_NAME", "tired")

# ---------------------------------------------------------------------------
# Logging and data locations
# ---------------------------------------------------------------------------
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.getLogger("comtypes").setLevel(logging.CRITICAL)

LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_filename = LOG_DIR / f"app_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_filename), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据文件路径（统一放在data目录）
DATA_DIR = BASE_DIR / "data"
USER_CSV_FILE = DATA_DIR / "users" / "users.csv"
SCORES_CSV_FILE = DATA_DIR / "users" / "scores.csv"
SCHULTE_SCORES_CSV_FILE = DATA_DIR / "users" / "schulte_scores.csv"
QUESTIONNAIRE_YAML_FILE = DATA_DIR / "questionnaires" / "questionnaire.yaml"

# 确保数据目录存在
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "users").mkdir(exist_ok=True)
(DATA_DIR / "questionnaires").mkdir(exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
NO_CAMERA_MODE = False

# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------

def load_users_from_csv() -> dict[str, str]:
    """从CSV文件加载用户账户信息。"""
    users: dict[str, str] = {}
    if USER_CSV_FILE.exists():
        try:
            with USER_CSV_FILE.open('r', newline='', encoding='utf-8') as handle:
                reader = csv.reader(handle)
                for row in reader:
                    if len(row) >= 2:
                        users[row[0]] = row[1]
            logger.info("成功从CSV加载 %s 个用户。", len(users))
        except Exception as exc:
            logger.error("读取用户数据时出错: %s", exc)
    if not users:
        users = {"admin": "123456"}
        save_users_to_csv(users)
        logger.info("未找到用户文件，已创建默认用户 (admin/123456)。")
    return users


def save_users_to_csv(users: dict[str, str]) -> None:
    """将用户账户信息保存到CSV文件。"""
    try:
        with USER_CSV_FILE.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            for username, password in users.items():
                writer.writerow([username, password])
        logger.info("已将 %s 个用户保存到CSV。", len(users))
    except Exception as exc:
        logger.error("保存用户数据时出错: %s", exc)


def build_session_dir(root: str, username: str, timestamp: str) -> str:
    base = Path(root)
    if not base.is_absolute():
        project_root = BASE_DIR.parent
        base = (project_root / base).resolve()
    path = base / username / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


__all__ = [
    "ACTIVE_CAMERA_INDEX",
    "ACTIVE_AUDIO_DEVICE_INDEX",
    "APP_MODE",
    "BASE_DIR",
    "bp_get_snapshot",
    "bp_get_status",
    "bp_start_measurement",
    "bp_stop_measurement",
    "HAS_BP_BACKEND",
    "BP_SIMULATION",
    "BP_PORT",
    "DATA_DIR",
    "DEBUG_MODE",
    "FORCE_SIMULATION",
    "HAS_MAIBOBO_BACKEND",
    "HAS_MULTIMODAL",
    "HAS_EEG_BACKEND",
    "HAS_SPEECH_RECOGNITION",
    "NO_CAMERA_MODE",
    "QUESTIONNAIRE_YAML_FILE",
    "SCHULTE_SCORES_CSV_FILE",
    "SCORES_CSV_FILE",
    "SKIP_DATABASE",
    "USER_CSV_FILE",
    "add_audio_for_recognition",
    "av_get_audio_paths",
    "av_get_current_audio_level",
    "av_get_current_frame",
    "av_start_collection",
    "av_start_recording",
    "av_stop_collection",
    "av_stop_recording",
    "av_get_video_paths",
    "eeg_start_collection",
    "eeg_stop_collection",
    "eeg_get_snapshot",
    "eeg_get_file_paths",
    "multidata_start_collection",
    "multidata_stop_collection",
    "multidata_get_snapshot",
    "av_stop_recording",
    "build_session_dir",
    "clear_recognition_results",
    "env_flag",
    "env_int",
    "get_recognition_results",
    "logger",
    "matplotlib",
    "multidata_get_snapshot",
    "multidata_start_collection",
    "multidata_stop_collection",
    "plt",
    "font_manager",
    "Figure",
    "FigureCanvas",
    "save_users_to_csv",
    "load_users_from_csv",
    "stop_recognition",
]
