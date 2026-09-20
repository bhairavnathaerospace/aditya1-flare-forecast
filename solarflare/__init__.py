"""Deep learning for solar flare nowcasting and forecasting from Aditya-L1.

Combines SoLEXS soft X-ray (1-22 keV, 340-channel spectra at 1 s) with HEL1OS
hard X-ray (18-160 keV in 5 bands) through a dual-encoder causal temporal
convolutional network with mask-aware cross-modal fusion.

Quick start::

    python -m solarflare inspect
    python -m solarflare train
"""

__version__ = "1.0.0"

__all__ = ["config", "io", "preprocess", "models", "pipeline",
           "train", "evaluate", "predict", "metrics", "plots"]
