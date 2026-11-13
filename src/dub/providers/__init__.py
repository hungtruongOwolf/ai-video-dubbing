"""Providers: one adapter per model family.

Each provider is described by a manifest in `configs/providers/<id>.yaml`. The core finds
it through the `impl` string in the manifest and never imports this package directly.
"""
