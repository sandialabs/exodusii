# Copyright NTESS. See COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

"""Agent-facing bootstrap helpers (``python -m exodusii agent bootstrap``)."""

from exodusii.agent.bootstrap import BOOTSTRAP_TARGETS
from exodusii.agent.bootstrap import bootstrap_file_map
from exodusii.agent.bootstrap import render_bootstrap_markdown
from exodusii.agent.bootstrap import render_claude_guidance_markdown
from exodusii.agent.bootstrap import write_agent_bootstrap

__all__ = [
    "BOOTSTRAP_TARGETS",
    "bootstrap_file_map",
    "render_bootstrap_markdown",
    "render_claude_guidance_markdown",
    "write_agent_bootstrap",
]
