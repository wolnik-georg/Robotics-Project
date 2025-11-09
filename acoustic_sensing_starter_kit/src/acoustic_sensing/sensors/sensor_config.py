"""
Real-time sensor configuration module
"""

from dataclasses import dataclass
from typing import Dict

@dataclass
class SensorConfig:
    """Configuration for optimized real-time sensor"""
    
    sample_rate: int = 44100
    buffer_size: int = 1024  # ~23ms at 44.1kHz
    hop_length: int = 512
    feature_mode: str = "OPTIMAL"  # Use 5-feature optimal set
    processing_timeout: float = 0.001  # 1ms maximum processing time
    contact_threshold: Dict[str, float] = None
    
    def __post_init__(self):
        if self.contact_threshold is None:
            self.contact_threshold = {
                "spectral_centroid": 500.0,  # Hz shift from baseline
                "high_energy_ratio": 0.15,  # Energy ratio threshold
                "spectral_bandwidth": 200.0,  # Hz bandwidth increase
                "ultra_high_energy_ratio": 0.05,  # High-freq energy threshold
                "temporal_centroid": 0.4,  # Temporal distribution shift
            }