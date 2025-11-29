"""
Lightweight smoke test for PianoAgent + MirrorMind scaffolding.
Runs a single decision step and prints the resulting intent/action.
"""

import logging

from models.piano import PianoAgent


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    agent = PianoAgent(task="fix failing optimizer tests", repo_id="AgentLab")
    out = agent.step()
    print(out)


if __name__ == "__main__":
    main()
