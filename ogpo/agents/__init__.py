"""OGPO Agents Module.

Contains implementations of various RL agents:
- OGPO (OGPOAgent): Main OGPO algorithm
- QC/ACFQL (ACFQLAgent): Action-Chunked Flow Q-Learning
- BPTT (OGPOBPTTAgent): OGPO with BPTT
- DSRL (DSRLBestOfNAgent): Diffusion-based SRL with Best-of-N
- EXPO (EXPOAgent): EXPO algorithm
- DSRL+EXPO (DSRLEXPOAgent): Combined DSRL and EXPO

Note: Import agents lazily to avoid circular imports during module loading.
Use get_agents_registry() from main.py or import specific agents directly:
    from ogpo.agents.ogpo import OGPOAgent
"""

__all__ = [
    'ogpo',
    'qc',
    'bptt',
    'dsrl',
    'expo',
    'dsrl_plus_expo',
]
