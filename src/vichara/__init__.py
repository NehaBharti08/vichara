"""Vichara -- a study agent that plans, calls tools, and cites its sources.

The agent loop is the least interesting part of this package. What the repo is
actually for lives in :mod:`vichara.eval` (trajectory metrics measured against
human-annotated gold paths), :mod:`vichara.guardrails.injection` (defences
against instructions arriving through tool output), and the threat model that
documents what the sandbox does not stop.
"""

__version__ = "0.1.0"
