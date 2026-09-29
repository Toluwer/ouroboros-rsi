"""Ouroboros: a self-improving language model system.

Phase 0 supervised fine-tuning is performed by a human operator before the
system is ever allowed to act. From then on, the autonomous daemon runs a
STaR-style bootstrapping loop: sample, filter by ground truth, fine-tune,
and accept or roll back based on held-out accuracy. Every accepted cycle
increments the model version; rejected cycles are rolled back.

See README.md for the method, SAFETY.md for the bounded self-modification
model, and config/constitution.json for the hard limits the daemon may not
cross.
"""

__version__ = "0.1.0"
