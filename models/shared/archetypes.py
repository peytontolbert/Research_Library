"""
Mapping of model IDs to archetypes/objectives for quick reference.
This can be used by tooling or downstream trainers to pick heads/losses.
"""

ARCHETYPES = {
    # Tier 1
    "M1": {"archetype": "contrastive", "objective": "contrastive"},
    "M2": {"archetype": "classifier", "objective": "cross_entropy"},
    "M3": {"archetype": "graph", "objective": "link_prediction"},
    "M4": {"archetype": "graph", "objective": "link_prediction"},
    "M5": {"archetype": "contrastive", "objective": "contrastive"},
    # Tier 2
    "A1": {"archetype": "contrastive", "objective": "contrastive"},
    "A2": {"archetype": "generative", "objective": "cross_entropy"},
    "A3": {"archetype": "classifier", "objective": "cross_entropy"},
    "A4": {"archetype": "classifier", "objective": "cross_entropy"},
    # Tier 3
    "P0": {"archetype": "generative", "objective": "cross_entropy"},
    "P1": {"archetype": "generative", "objective": "cross_entropy"},
    "P2": {"archetype": "generative", "objective": "cross_entropy"},
    "P3": {"archetype": "generative", "objective": "cross_entropy"},
    "P4": {"archetype": "generative", "objective": "cross_entropy"},
    # Tier 4
    "R1": {"archetype": "contrastive", "objective": "contrastive"},
    "R2": {"archetype": "contrastive", "objective": "contrastive"},
    "R3": {"archetype": "generative", "objective": "cross_entropy"},
    "R4": {"archetype": "classifier", "objective": "cross_entropy"},
    "R5": {"archetype": "policy", "objective": "rl_ppo"},
    "R6": {"archetype": "contrastive", "objective": "contrastive"},
    # Tier 5
    "C1": {"archetype": "generative", "objective": "cross_entropy"},
    "C2": {"archetype": "contrastive", "objective": "contrastive"},
    "C3": {"archetype": "adapter", "objective": "cross_entropy"},
    "C4": {"archetype": "adapter", "objective": "cross_entropy"},
    "C5": {"archetype": "adapter", "objective": "cross_entropy"},
    "C6": {"archetype": "contrastive", "objective": "contrastive"},
    # Tier 6
    "U1": {"archetype": "generative", "objective": "cross_entropy"},
    "U2": {"archetype": "generative", "objective": "cross_entropy"},
    "U3": {"archetype": "policy", "objective": "rl_ppo"},
    # Tier 7
    "S1": {"archetype": "policy", "objective": "rl_ppo"},
    "S2": {"archetype": "classifier", "objective": "cross_entropy"},
    "S3": {"archetype": "policy", "objective": "rl_ppo"},
}


def get_archetype(model_id: str):
    """Return archetype info for a model id."""
    return ARCHETYPES.get(model_id)
