"""OGPO: Online Goal-conditioned Policy Optimization.

A modular implementation of flow-matching based reinforcement learning algorithms.

Modules:
    agents: Agent implementations (OGPO, QC, BPTT, DSRL, EXPO, DSRL+EXPO)
    networks: Neural network architectures (actors, critics, encoders)
    utils: Utility functions (datasets, logging, evaluation)

Entry Points:
    main.py: Unified training script for all algorithms
    bc_runner.py: BC (Behavior Cloning) training runner
"""

__version__ = "0.1.0"
__author__ = "OGPO Team"
