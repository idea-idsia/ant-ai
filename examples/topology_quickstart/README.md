# Topology quickstart

The smallest adaptive-topology run: three agents, one strategy string, one report.

```bash
uv run python -m examples.topology_quickstart --dry-run   # no LLM, no API key
uv run python -m examples.topology_quickstart             # needs an LLM
```

`--dry-run` builds and validates the ensemble, prints what it assembled, and
stops before any model call. It is the fastest way to see what a strategy string
resolves to:

```
pipeline    : Semantic -> TopK -> Heal
scheduler   : BufferScheduler
materialiser: DeliveryMaterialiser
seed links  : [('developer', 'architect'), ('architect', 'developer'), ...]
provenance  : dytopo|dig
```

## What it shows

**One string selects a strategy.** `colony.evolve("dytopo|dig")` layers
semantic rewiring (DyTopo) under structural repair (DIG). Equivalent to
`DyTopo() | DigToHeal()`, and the names are what an ablation sweep varies.

**`collab()` edges still matter.** They seed round 0 and remain the standing
topology for any round no stage rewires.

**The configuration is checked before it runs.** `ensemble()` raises
`TopologyConfigurationError` on a combination that cannot work — a matcher with
nothing to match on, a topology that can reach nobody — rather than running to
completion and producing an artefact. Try it:

```bash
# E001: a workflow-driven turn carries no descriptors for the matcher to read
uv run python -c "
from examples.topology_quickstart.__main__ import build
build('gpt-4o-mini').ensemble(use_workflows=True)
"
```

**The report says whether it did anything.** A topology that never moved and a
detector firing every round are both invisible in the answer string:

```
strategy       : dytopo|dig
rounds         : 4  (reviewer submitted)
messages       : 30 generated, 21 delivered, 21 consumed, 9 outstanding
links/round    : r1=6, r2=4, r3=5
findings       : MCx1
```

## Next

- [Adaptive topology](../../docs/docs/multi-agent/topology.md) — the guide
- [Topology architecture](../../docs/docs/architecture/topology.md) — the design
- [`dig_in_action/`](../dig_in_action) — the same machinery drawn live as the DIG figure
