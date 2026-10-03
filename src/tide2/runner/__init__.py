"""
TIDE 2.0 Runner Module.

Provides job runners for recognition, anonymization, and transformer jobs.

Runner classes:
- LocalJobRunner: Single-node execution using Ray (laptop or GCP VM)

Usage:
    # CLI
    tide2-runner run recognizer -i ./data/input -o ./data/output

    # Python API
    from tide2.runner import LocalJobRunner

    runner = LocalJobRunner()
    result = runner.run_recognition("./input", "./output")
"""

from .fault_tolerance import GracefulShutdown
from .fault_tolerance import chunked
from .fault_tolerance import configure_data_context
from .fault_tolerance import get_ray_remote_args_cpu
from .fault_tolerance import get_ray_remote_args_gpu
from .hardware import Applied
from .hardware import HardwareFacts
from .hardware import NodeShape
from .hardware import Recommendations
from .hardware import SettingEntry
from .hardware import apply_recommendations
from .hardware import classify_profile
from .hardware import detect_hardware
from .hardware import recommend_object_store_gb
from .hardware import recommend_settings
from .hardware import render_settings_table
from .local_runner import LocalJobRunner
from .local_runner import run_anonymization_simple
from .local_runner import run_pipeline_simple
from .local_runner import run_recognition_simple
from .local_runner import run_transformer_simple
from .utils import DEFAULT_DASHBOARD_HOST
from .utils import detect_columns
from .utils import init_ray_local
from .utils import log_ray_cluster_info
from .utils import resolve_input_files

__all__ = [
    "DEFAULT_DASHBOARD_HOST",
    "Applied",
    "GracefulShutdown",
    "HardwareFacts",
    "LocalJobRunner",
    "NodeShape",
    "Recommendations",
    "SettingEntry",
    "apply_recommendations",
    "chunked",
    "classify_profile",
    "configure_data_context",
    "detect_columns",
    "detect_hardware",
    "get_ray_remote_args_cpu",
    "get_ray_remote_args_gpu",
    "init_ray_local",
    "log_ray_cluster_info",
    "recommend_object_store_gb",
    "recommend_settings",
    "render_settings_table",
    "resolve_input_files",
    "run_anonymization_simple",
    "run_pipeline_simple",
    "run_recognition_simple",
    "run_transformer_simple",
]
