"""Load each agent only when requested, keeping optional dependencies optional."""

from importlib import import_module

_AGENTS = {
    "GreedyRefine": "greedy_refine",
    "DirectAnswer": "direct_answer",
    "FunSearch": "funsearch",
    "AIDE": "aide",
    "ChainOfExperts": "chain_of_experts",
    "ReEvo": "reevo",
    "BestOfN": "bestofn",
    "YieldGreedyRefine": "new_greedy_refine",
    "YieldFunSearch": "new_funsearch",
    "YieldReEvo": "new_reevo",
    "EoHAgent": "eoh_agent",
    "MctsAhd": "mcts_agent",
}
__all__ = list(_AGENTS)


def __getattr__(name):
    if name not in _AGENTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    agent = getattr(import_module(f"agents.{_AGENTS[name]}"), name)
    globals()[name] = agent
    return agent
