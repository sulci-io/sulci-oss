# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sulci Labs Inc.
"""
sulci/sinks/null.py — NullSink (v0.5.0)

The default sink used when Cache() is constructed without event_sink=.
Does nothing. Zero overhead.
"""
from sulci.sinks.protocol import CacheEvent, EventSink


class NullSink(EventSink):
    """Default no-op sink. All emits are dropped."""

    def emit(self, event: CacheEvent) -> None:
        pass

    def flush(self) -> None:
        pass
