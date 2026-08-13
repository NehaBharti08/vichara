"""Graph construction.

```
START -> plan -> {refuse | clarify | act}
act -> {guard | synthesize | halt}
guard -> {approve | execute | halt}
approve -> {execute | observe}          # interrupt() suspends here
execute -> observe -> {reflect | compress | act | synthesize}
reflect -> {plan | act | halt}
synthesize -> halt -> END
```

Two routing decisions worth defending:

**`observe` is the hub, not `act`.** Every path back into the loop passes
through it, which is what gives compression, reflection and completion a
single place to be decided rather than three duplicated conditions.

**Nothing returns to `act` without passing the guard again.** A ceiling
checked once at the top of the loop is a ceiling that a replan can walk
straight past.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from vichara.agent.memory import should_compress
from vichara.agent.nodes.acting import (
    act_node,
    approve_node,
    execute_node,
    guard_node,
    observe_node,
    route_after_act,
    route_after_approve,
    route_after_guard,
)
from vichara.agent.nodes.context import AgentContext
from vichara.agent.nodes.finishing import compress_node, halt_node, synthesize_node
from vichara.agent.nodes.planning import (
    clarify_node,
    plan_node,
    reflect_node,
    refuse_node,
    route_after_plan,
    route_after_reflect,
    should_reflect,
)
from vichara.agent.state import AgentState
from vichara.logging import get_logger

log = get_logger(__name__)


def _bind(
    fn: Callable[[AgentState, AgentContext], AgentState], context: AgentContext
) -> Callable[[AgentState], AgentState]:
    """Bind context so nodes keep LangGraph's single-argument signature.

    The context holds tools, a provider and open file handles -- none of which
    can be checkpointed. Keeping it out of state is what lets a run resume in
    a different process.

    An explicit closure rather than functools.partial: partial erases the
    concrete signature to Callable[..., Any], and LangGraph's add_node
    overloads then match nothing.
    """

    def node(state: AgentState) -> AgentState:
        return fn(state, context)

    return node


def route_after_observe(state: AgentState, context: AgentContext) -> str:
    """The loop's main decision.

    Order matters. A terminal reason set by the guard wins over everything;
    then plan completion; then reflection; then compression. Compression is
    last because digesting the history and then immediately terminating wastes
    a request on a summary nobody reads.
    """
    if state.get("terminal_reason") is not None:
        return "halt"

    plan = state.get("plan")
    step = state.get("step", 0)

    if step >= context.config.budget.max_steps:
        return "synthesize"

    should, _ = should_reflect(state, context)
    if should:
        return "reflect"

    messages: list[Any] = []
    if should_compress(messages, step, context.config.memory):
        return "compress"

    if plan is not None and plan.steps and step >= len(plan.steps):
        # The plan is advisory: finishing it is a signal to try answering, not
        # an instruction to stop. The act node decides whether it actually can.
        return "act"

    return "act"


def build_graph(context: AgentContext, checkpointer: Any | None = None) -> Any:
    """Compile the agent graph."""
    # Annotated Any deliberately. LangGraph's `add_node` overloads are bounded
    # on TypedDictLikeV1 | TypedDictLikeV2 | DataclassLike | BaseModel, and
    # mypy --strict does not resolve any of them against a `total=False`
    # TypedDict -- every add_node call below reports "no overload variant
    # matches". The runtime is fine; this is a stub limitation. Containing it
    # in one annotated line is more honest than twelve scattered ignores,
    # and it is narrow: only graph wiring loses checking, not the nodes.
    graph: Any = StateGraph(AgentState)

    graph.add_node("plan", _bind(plan_node, context))
    graph.add_node("refuse", _bind(refuse_node, context))
    graph.add_node("clarify", _bind(clarify_node, context))
    graph.add_node("act", _bind(act_node, context))
    graph.add_node("guard", _bind(guard_node, context))
    graph.add_node("approve", _bind(approve_node, context))
    graph.add_node("execute", _bind(execute_node, context))
    graph.add_node("observe", _bind(observe_node, context))
    graph.add_node("reflect", _bind(reflect_node, context))
    graph.add_node("compress", _bind(compress_node, context))
    graph.add_node("synthesize", _bind(synthesize_node, context))
    graph.add_node("halt", _bind(halt_node, context))

    graph.add_edge(START, "plan")

    graph.add_conditional_edges(
        "plan",
        route_after_plan,
        {"act": "act", "refuse": "refuse", "clarify": "clarify", "halt": "halt"},
    )
    graph.add_edge("refuse", "halt")
    graph.add_edge("clarify", "halt")

    graph.add_conditional_edges(
        "act",
        route_after_act,
        {"guard": "guard", "synthesize": "synthesize", "halt": "halt"},
    )
    graph.add_conditional_edges(
        "guard",
        route_after_guard,
        {
            "approve": "approve",
            "execute": "execute",
            "synthesize": "synthesize",
            "halt": "halt",
        },
    )
    graph.add_conditional_edges(
        "approve",
        route_after_approve,
        {"execute": "execute", "observe": "observe"},
    )
    graph.add_edge("execute", "observe")

    def observe_router(state: AgentState) -> str:
        return route_after_observe(state, context)

    graph.add_conditional_edges(
        "observe",
        observe_router,
        {
            "act": "act",
            "reflect": "reflect",
            "compress": "compress",
            "synthesize": "synthesize",
            "halt": "halt",
        },
    )
    graph.add_edge("compress", "act")
    graph.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {"plan": "plan", "act": "act", "halt": "halt"},
    )

    graph.add_edge("synthesize", "halt")
    graph.add_edge("halt", END)

    return graph.compile(checkpointer=checkpointer)
