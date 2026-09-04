---
title: Adaptive topology
---

# Adaptive topology

A [`Colony`][ant_ai.a2a.colony.Colony] wires collaboration edges once, with `collab()`. An
**adaptive topology** recomputes them at runtime: which agents can reach which is decided per round
instead of at construction.

The mechanism deliberately keeps two things apart:

| | Decided by |
| --- | --- |
| **Reachability** — who is in my address book this round | the topology layer |
| **Selection** — whom I actually talk to | the agent: a tool call under visibility, `Envelope.recipients` under delivery |

There is no router agent. Each participant publishes its own natural-language **query** ("what I
need") and **key** ("what I offer"), and the matcher is a mechanical cosine comparison over those
self-descriptions — both sides of every comparison are text an agent wrote about itself. The
AgentCard stays the channel: it is the *static* half of a participant's advertisement, and the
descriptors are the *dynamic* half. What changes each round is which cards are in scope.

## Quick start

```python
from ant_ai.a2a import Colony
from ant_ai.embeddings.backends.sentence_transformer import SentenceTransformerEmbedder
from ant_ai.topology.builtins import DyTopo

colony = Colony()
colony.agent("architect", agent=architect, workflow=wf, card=card_a)
colony.agent("developer", agent=developer, workflow=wf, card=card_d)
colony.agent("reviewer", agent=reviewer, workflow=wf, card=card_r)

colony.collab("architect", "developer", mutual=True)  # still the round-0 seed

colony.topology(DyTopo(embedder=SentenceTransformerEmbedder()))

async for event in colony.ensemble().stream("Build a CSV parser"):
    ...
```

`SentenceTransformerEmbedder` needs the optional extra:

```bash
pip install 'ant-ai[topology]'
```

A colony with no `topology()` call behaves exactly as before: `collab()` edges are the static
topology, materialised as peer tools.

## How a round runs

```mermaid
flowchart LR
    A["1. Act<br/>scheduler picks who runs"] --> B["2. Record<br/>lineage + consumption"]
    B --> C["3. Plan<br/>run the stages in order"]
    C --> D["4. Halt?<br/>ask the halt policy"]
    D --> E["5. Materialise<br/>rebind tools / fill inboxes"]
    E -.->|next round| A
```

A decided topology governs the **next** round: it is what participants will act under. Round 0 is
seeded from the colony's declared `collab()` edges.

Halting is asked **after** the stages, not before. Repairing an early termination reroutes a
submit back to its issuer and un-terminates it, which only means anything if the run has not
already ended — checking halting first would make the single most consequential detector unable
to act.

## Core concepts

A strategy is an ordered list of **stages**, each transforming the plan for the next round.

| Concept | Description |
| --- | --- |
| [`Stage`][ant_ai.topology.plan.Stage] | One transform of the next round's plan. The extension point for a routing or repair algorithm. |
| [`RoundPlan`][ant_ai.topology.plan.RoundPlan] | What the next round will look like: turns, scores, links, notices, findings. Stages return a new one rather than mutating. |
| [`RunContext`][ant_ai.topology.plan.RunContext] | The state of the run right now — read-only, so everything a stage writes goes in the plan it returns. |
| [`Detector`][ant_ai.topology.heal.Detector] | Finds one structural failure pattern. Hosted by the `Heal` stage, which owns applying corrections. |
| [`TopologyMaterialiser`][ant_ai.topology.materialise.TopologyMaterialiser] | Turns a plan into reality: `VisibilityMaterialiser` rebinds peer tools, `DeliveryMaterialiser` routes messages. |
| [`Scheduler`][ant_ai.topology.schedule.Scheduler] | Who activates on a tick. `RoundScheduler` is the synchronous barrier; `BufferScheduler` activates only agents whose inbox changed. |
| [`Halt`][ant_ai.topology.halt.Halt] | Who may end a run, and not before which round. |
| [`InteractionGraph`][ant_ai.topology.graph.InteractionGraph] | The run record: activations, messages, and both granted and exercised edges. |
| [`TopologyStrategy`][ant_ai.topology.strategy.TopologyStrategy] | A published method's hyperparameters plus how they assemble, via one hook: `build()`. |
| [`Ensemble`][ant_ai.topology.runtime.Ensemble] | The round loop. |

The record answers *what happened*; a strategy says *what it means*. `InteractionGraph` will tell
you which messages nothing consumed; whether that is a failure, and after how long, is a published
method's claim and lives with it in `ant_ai.topology.builtins`.

### Edge direction

`Link(src, dst)` is **information flow**: `src` offers, `dst` needs.

- Delivery pushes `src`'s message into `dst`'s inbox — direct.
- Visibility gives **`dst`** a tool that calls **`src`**, because `dst` is the one that needs what
  `src` offers. The tool binding is the *reverse* of the arrow.

`Colony.collab(source, target)` means *source may call target*, so a colony records its declared
edges as `Link(src=target, dst=source)`.

### Addressing

Under delivery, the two halves of that split meet in one rule:

```
delivery = selection ∩ reachability
```

**Selection** is [`Envelope.recipients`][ant_ai.topology.participant.Envelope] — whom the sender
addressed, in its own words, elicited in the same single pass as everything else
(`TurnPayload.messages`). **Reachability** is `plan.links`. Each side has a default, and the
defaults are what let two very different published methods run through one materialiser:

- A sender that addressed **nobody** is routed by the links — a matcher-driven run, unchanged.
- A round **no stage wrote links for** has no opinion on reachability, so an addressed message goes
  where it was addressed. That is a method whose agents name their own correspondents, with no
  matcher under it at all.
- A message addressed to somebody unreachable is **not** delivered. It stays a generated event that
  reached no one — which is the failure `OrphanedEvent` reports, and silently widening reachability
  to whoever was named would erase the pathology instead of surfacing it.

A turn therefore carries `outputs`, a list, not a public/private pair: one activation splitting work
five ways sends five different messages, not one broadcast five agents happen to read. `public` and
`private` remain as accessors and as constructor shorthand for the common case.

### Reactions

A participant also says what it *did* with each message it was handed — `consume`, `wait`,
`discard`, or `reroute` to someone better placed — keyed by the `[eN]` tag the brief gave it:

```python
TurnPayload(reactions={"e1": "wait"}, reroute={"e2": ["reviewer"]})
```

Anything unmentioned counts as consumed. The reaction is what the `delivers` edge records, so the
record says what happened rather than what the scheduler assumed, and three things follow from it:
a waited message stays in the buffer and stays outstanding, lineage is attributed from what was
*consumed* rather than from everything delivered, and `ExcessiveRerouting` counts a message being
bounced whether the bouncing was an agent's decision or a supervisor's.

## Using a strategy

Strategies live in `ant_ai.topology.builtins`, one module per paper.

```python
from ant_ai.topology import Halt
from ant_ai.topology.builtins import DigToHeal, DyTopo, chain, mesh, star

colony.topology(mesh(["architect", "developer", "reviewer"]))
colony.topology(DyTopo(embedder=embedder, tau=0.35, k_in=3))
```

Hyperparameters are validated fields, so `DyTopo(tau=2.0)` fails at construction rather than
producing a quietly meaningless run, and `strategy.provenance()` reports them without a
hand-maintained dict.

### Composing

Two strategies layer with `|`. Stages concatenate; for every other component the right-hand side
wins, but **only** where it set that component explicitly — so composing never reverts a setting
to a default:

```python
strategy = DyTopo(embedder=embedder) | DigToHeal()
```

That yields DyTopo's `Semantic` and `TopK` stages followed by DIG's `Heal`, DIG's scheduler and
materialiser, and a provenance record naming both halves. Neither strategy knows the other exists.

## Adding a strategy

A new method answers up to four questions, and overrides only the ones its paper actually changes:

| Question | Seam |
| --- | --- |
| What changes reachability? | a `Stage` |
| What counts as broken? | a `Detector`, hosted by `Heal` |
| Who acts when? | a `Scheduler` |
| Who says stop? | a `Halt` |

Then one module under `builtins/`, with a `TopologyStrategy` whose single hook assembles them:

```python
class MyMethod(TopologyStrategy):
    name = "mine"
    citation = "arXiv:..."

    threshold: float = Field(0.5, ge=0.0, le=1.0)  # validated, recorded in provenance

    def build(self) -> Pipeline:
        return Pipeline(stages=[MyStage(threshold=self.threshold)])
```

Nothing in the core changes, and the strategy registers itself by name:

```python
strategy = TopologyStrategy.create("mine", threshold=0.7)
```

A stage is one async method. It holds no participant handles, performs no I/O on the run and
cannot deliver anything, so it can be replayed against a recorded trace with no agents running:

```python
class Decay(BaseModel):
    """Halve the weight of every edge that survived from last round."""

    async def apply(self, plan: RoundPlan, ctx: RunContext) -> RoundPlan:
        previous = {(l.src, l.dst) for l in ctx.graph.links(ctx.round)}
        return plan.with_links(
            tuple(
                l.model_copy(update={"weight": l.weight / 2})
                if (l.src, l.dst) in previous
                else l
                for l in plan.links
            )
        )
```

Scoring and sparsifying are separate stages because nearly every routing method is those two
steps. `RandomTopology` reuses DyTopo's own `TopK` unchanged, so a random control holds sparsity
constant by construction rather than by careful reimplementation.


## Halting

Who may end a run is a **method** choice, not a framework constant. Left implicit it silently
distorts comparisons: with agent-declared halting, the first agent to finish its own piece ends
everyone's round, so two conditions run for different numbers of rounds and their token totals stop
being comparable.

One class with three knobs, set on a strategy's `Pipeline`:

```python
Halt()  # anyone may end the run (default)
Halt(unanimous=True)  # everyone must agree
Halt(deciders={"integrator"}, min_rounds=3)  # only the manager, and not before round 3
Halt.never()  # pin the round budget
```

Two independent constraints in one rule rather than two objects nested. `min_rounds` exists
because a weak completion signal — docstring examples passing, say — otherwise ends a run at
round 1 on false confidence and the topology never gets a second round to rewire.

## Self-healing

A [`Detector`][ant_ai.topology.heal.Detector] finds one structural failure pattern; a
[`Heal`][ant_ai.topology.heal.Heal] is the stage that runs a set of them and applies what they
prescribe. The split is the point: what varies between healing methods is the *detectors*, not the
mechanics of rewriting a message.

```python
from ant_ai.topology.builtins import DigToHeal, dig_detectors

colony.topology(DyTopo(embedder=embedder) | DigToHeal())  # routing plus repair
colony.topology(
    strategy, detectors=dig_detectors()[:2]
)  # a one-off subset, or your own
```

`DigToHeal` implements [arXiv:2603.00309](https://arxiv.org/abs/2603.00309). Its seven detectors:

| | Pattern | Fires when | Correction |
| --- | --- | --- | --- |
| `ET` | Early Termination | a submit lands while work is unconsumed | inject what is outstanding, reroute the submit back to its issuer — which **un-terminates** it, so the run continues |
| `MC` | Missing Completion | work is exhausted and nobody submitted | emit "the work is done, somebody call it" |
| `OE` | Orphaned Event | a message was routed to nobody | inject status, reroute to its generator |
| `DL` | Deadlock | work is pending and nobody activated | emit a broadcast to restart activity |
| `ER` | Excessive Rerouting | one message rerouted past a threshold, never consumed | inject that fact into the payload |
| `CLA` | Cross-Lineage Aggregation | one activation consumed messages with disjoint ancestry | inject the lineage into each |
| `RSP` | Repeated Subproblem | two problem-reducing activations consumed the same input | tell both they may be duplicating |

Writing a new one is a single async method over the graph:

```python
class Starvation(Detector):
    pattern: str = "STARVE"

    async def detect(self, graph, ctx):
        idle = [n for n in ctx.names if not graph.in_neighbours(n, ctx.round)]
        return [
            self.finding(
                ctx,
                f"{n} heard from nobody",
                Intervention(kind="emit", content="...", recipients=(n,)),
            )
            for n in idle
        ]
```

A detector holds no participant handles, performs no I/O and cannot deliver anything, so it can be
developed and falsified against a recorded trace with no agents running. Every finding also reaches
the caller as a [`HealingEvent`][ant_ai.core.events.HealingEvent] — healing that leaves no trace on
the stream is indistinguishable from a run that never needed it.

### Why the scheduler matters here

Four of the seven detectors are unreachable under a synchronous barrier. If every agent acts every
round then `V_A(t)` is never empty and `Deadlock` is dead code; an agent nobody routed to still
burns a turn, so `OrphanedEvent` never manifests. `BufferScheduler` activates only participants
whose inbox changed, which makes an empty activation set a legitimate state a detector can see.
`DigToHeal` selects it by default.

## Comparing strategies

There is no benchmark harness in the library. What the library gives you is the
*comparability*: every strategy runs through the same `Ensemble`, leaves the same
`InteractionGraph` to measure, and records its own `provenance()`. The loop over conditions is
yours, and it is short:

```python
from ant_ai.topology import Ensemble
from ant_ai.topology.builtins import Baseline, DigToHeal, DyTopo

dytopo = DyTopo(embedder=embedder, max_rounds=6)

for label, strategy in {
    "no method": Baseline(max_rounds=6),
    "dytopo": dytopo,
    "dytopo + healing": dytopo | DigToHeal(),
}.items():
    ensemble = Ensemble(
        participants=build_participants(),  # fresh per run — agents carry state
        pipeline=strategy.pipeline(),
        provenance=strategy.provenance(),
    )
    answer = await ensemble.ainvoke(task)
    print(label, score(answer), len(ensemble.graph.rounds()), ensemble.findings)
```

Building participants fresh for every run matters: agents carry conversation state and peer
bindings, so reusing them leaks one condition's history into the next.

`examples/_bench.py` is a fuller version of that loop — repeats, structural metrics off the
graph, findings per pattern, a markdown table — kept in `examples/` rather than shipped, because
which metrics matter and how to aggregate them are a researcher's call, not the framework's.
`examples/dig_healing.py` uses it, offline.

## Explainability

Every routing decision reaches the caller as a
[`TopologyEvent`][ant_ai.core.events.TopologyEvent], one per round, whose links carry the score and a
human-readable reason.

```python
async for event in ensemble.stream(task):
    match event:
        case TopologyEvent():
            for link in event.links:
                print(event.round, link.src, "->", link.dst, link.reason)
```

The `InteractionGraph` records reachability the policy *granted* and calls the agent actually
*made*, separately. Their difference is signal:

```python
ensemble.graph.unused_visibility(round=2)  # peers reachable but never called
ensemble.graph.to_mermaid()  # per-round diagram
```

The graph is plain pydantic, so a whole run round-trips through `model_dump_json()` and can be
analysed — or replayed into a supervisor — with no agents running.

### Watching one happen

`examples/dig_in_action/` draws the graph while the run is still going: activations appear as they
start, the time axis grows, and a repair shows up at the moment the supervisor makes it. It is a
live rebuild of the figure from the [DIG page](https://happyeureka.github.io/dig/), fed by
`Ensemble.stream` over SSE, and it runs offline with no model behind it.

```bash
uv run python -m examples.dig_in_action
```

The projection there takes an `InteractionGraph`, not a run, so the same page draws a recorded
trace with nothing running — and pointing it at a colony of real agents is a change of one factory
function.

## Limitations

- **Workflows and structured turns.** `Workflow.stream` takes no response schema, so a
  workflow-driven participant answers with one plain public message: no query/key descriptors, no
  addressed messages, no declared reactions and nothing ever submitted. Every strategy built on
  those degrades to something that runs and does nothing — a matcher scoring unchanging AgentCard
  text, or a detector that never sees a symptom. `ensemble()` therefore decides for you: it invokes
  agents directly when the pipeline reads any of it (`Pipeline.needs_structured_turns`), and runs
  the workflow when it does not. Pass `use_workflows=True` to override that; a matcher whose
  fallback ends up total warns once.
- **Remote peers cannot be rebound.** A2A has no operation for attaching a tool to an agent in
  another process, so `A2AParticipant` adapts under `DeliveryMaterialiser`; under visibility its
  reachability stays as the colony wired it, and the materialiser reports it as unbindable rather
  than pretending. Asking for both at once — `ensemble(local=False)` with a deciding stage and a
  visibility materialiser — is a topology that constrains nothing, and warns at build time.
- **Synchronous rounds.** Both schedulers still advance on a round barrier, so `BufferScheduler`
  gives event-driven *activation* but not event-driven *timing*: an agent acts at the next barrier
  rather than the instant its buffer changes, and turns that share a round start together. That
  costs latency realism, not detectability, and it is the last structural difference from a runtime
  whose agents fire the moment their mailbox changes.
- **Detection is per round.** `Heal` runs its detectors once per round, after the stages. A method
  that hooks activation-complete, event-delivered and idle separately would see the same conditions
  at a finer grain; the queries themselves are unchanged by that.
- **Message-level healing needs delivery mode.** DIG detects when an event is generated, before it
  is delivered. Under `VisibilityMaterialiser` a peer call collapses generation, delivery and
  activation into one synchronous tool call, so there is nothing to inspect in between;
  [`SupervisorHook`][ant_ai.topology.heal.SupervisorHook] applies the two rewrites a tool call
  can express (`inject`, `reroute`) at that boundary, and `drop`/`emit` stay with the round loop.
- **Unaddressed output still needs a topology.** A message that names no recipients is delivered by
  the links, so a run with neither addressing nor a routing stage moves nothing. Structural
  accounting settles a turn's outputs together (see `InteractionGraph.siblings`), which is what
  keeps the copy a turn leaves for the record from being reported as an orphan when its addressed
  sibling was delivered.
- **Stage order is load-bearing.** A sparsifier must follow the scorer whose `plan.scores` it
  reads. That is the honest cost of composing by concatenation: it is visible in the list, but
  nothing type-checks it.
