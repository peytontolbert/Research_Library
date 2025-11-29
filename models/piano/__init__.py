"""
PIANO-style control loop scaffold for the Code+Paper world.
Uses MirrorMind memory/twins as long-term memory and exposes a minimal agent shell.
"""

from .state import AgentState
from .controller import CognitiveController, Intent
from .modules import GoalGenerator, SkillExecutor, ActionAwareness, TalkingModule, SocialModule
from .agent import PianoAgent

__all__ = [
    "AgentState",
    "CognitiveController",
    "Intent",
    "GoalGenerator",
    "SkillExecutor",
    "ActionAwareness",
    "TalkingModule",
    "SocialModule",
    "PianoAgent",
]
