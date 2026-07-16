# BloomBot Roadmap: Phases 6 to 12

Living document, same convention as the project knowledge file: targets are estimates, reviewed and refined at the start of each phase. Detailed task breakdowns with done criteria are produced at phase start; this file holds scope, order, and exit criteria. Last updated: 2026-07-15.

## Status

Phases 1 to 5 are complete: RAG API (ChromaDB + OpenAI embeddings, FastAPI), production hardening (tests, validation, rate limiting), CI/CD (GitHub Actions, branch protection), monitoring and dual evaluation (retrieval metrics + LLM-as-judge), and a deployed Streamlit chat UI. This roadmap supersedes the original Phases 6 to 8.

## Phase mapping (original plan → this roadmap)

| Original phase | Disposition |
|---|---|
| Phase 6: Inventory-Aware Upgrade | Now Phase 8. One task was started early under the old numbering; it is parked until Phase 8 opens. |
| Phase 7: Image Generation | Rescoped. A minimal build-sheet render ships in Phase 8; the full iterative image feature is Phase 12. |
| Phase 8: Final Polish | Merged into Phase 11. |

## Sequencing rationale

1. Retrieval quality first: the dual evaluation framework already measures it, so every change ships with a before/after delta, and it directly addresses the known failure mode where fluent generation masked a retrieval miss (q13).
2. Multi-turn before composition: the composer should be designed session-aware ("make it cheaper", "swap the roses") rather than retrofitted.
3. Within the retrieval phase: preprocessing, then metadata filtering, then hybrid search, then reranking. Filtering consumes the structured constraints that preprocessing extracts, and reranking is measured on top of the improved candidate set.
4. Image work is split: a cheap spec-driven render lands in Phase 8, where the demo and the pilot benefit most; the full editing and evaluation feature waits until the core repo is complete (Phase 12).
5. The order also optimizes stopping points: if work halts early, the most substantial capabilities are already finished, documented, and deployed.

## Standing rules (every feature)

- One feature per branch → PR → merge. Tests included. CI green before merge.
- Any retrieval-affecting feature runs the evaluation suite before and after; the delta is recorded alongside the decision row in `docs/engineering-decisions-log.md`.
- Deploy after merge. README section and phase learning doc at phase close.
- Documentation describes only behavior that exists and numbers that were measured. Synthetic traffic is always labeled as synthetic and is never reported as user data.
- External data informs attribute distributions and query vocabulary only. No third-party text or images are copied into the repo, the catalog, or the vector store, except openly licensed sources used with attribution.

## Data and feedback design principles

- Feedback is an event stream with a fixed schema, agnostic to its source. Event fields: request_id, session_id, variant_id, source (ui, simulator, pilot), query, retrieved ids and scores, response, rating, latency, tokens, timestamp. The UI thumbs, the Phase 11 simulator, and pilot users all write the same event through the same ingestion path, so real data arriving requires no pipeline changes.
- Anything tuned on synthetic feedback is re-validated on real feedback before conclusions are drawn, since LLM raters carry systematic biases.

## Pilot plan

- Tier 1 (during Phases 10 to 11): 15 to 30 recruited users on the deployed app. Signals: thumbs, a short survey, 100+ real queries. Used for failure-mode discovery and the Phase 10 tuning iteration, not for statistical A/B claims, which need per-arm sample sizes in the hundreds of sessions.
- Tier 2 (after Phase 12): small fulfillment pilot, 5 to 10 real orders with local delivery, validating that recommendations translate to the physical product and that assembly instructions are followable when building real orders. Logistics and findings tracked outside the repo.

## Phase 6: Retrieval Quality (~2.5 weeks, target Jul 16 to Aug 1)

Goal: measurably better retrieval on a corpus large enough to make retrieval non-trivial.

Scope:

1. Coverage instrumentation: pytest-cov wired into CI, baseline percentage recorded. Enforcement gate deferred to Phase 11.
2. Catalog expansion to roughly 200 to 300 generated bouquets with structured metadata (price, flowers, colors, occasions, availability flag). Attribute distributions grounded in a manual survey of 20 to 30 real listings (price bands, stem counts, size tiers, occasion tags, naming conventions). No listing text or photos are copied.
3. Evaluation set expansion: ground truth relabeled against the new catalog and grown to roughly 40 to 50 queries, including exact-name and multi-constraint queries. Query phrasing informed by vocabulary found in public flower-shop reviews, written fresh, never republished. New baseline recorded; prior metrics retired as not comparable.
4. Query preprocessing: classification (on-topic, occasion, intent), cleanup, structured constraint extraction (budget, colors, flower includes and excludes, occasion), optional expansion.
5. Metadata filtering: extracted constraints applied as vector store filters alongside semantic search.
6. Hybrid search: BM25 keyword retrieval fused with dense retrieval. Fusion method decided at task start.
7. Reranking: cross-encoder or LLM-scored reranking over the fused candidate set. Approach decided at task start.

Expected metric direction (hypotheses to verify, not promises): filtering and reranking should lift precision (0.41 is the weak baseline number); hybrid search should lift recall on exact product names; preprocessing should lift both on messy and multi-intent queries.

Exit criteria: items 4 to 7 merged and deployed, each with an eval delta row in the decisions log; go or no-go recorded on the backlogged domain-knowledge collection. README v6.

## Phase 7: Multi-Turn Conversation (~1 week, target Aug 3 to Aug 8)

Goal: users can refine a request ("cheaper", "no roses, add lilies") without restating it.

Scope:

1. API contract: the request carries message history; the API stays stateless. Pre-decision: history-in-request rather than server-side sessions, to avoid session store infrastructure. Revisit at phase start.
2. Query condensation: history plus new message produce a standalone query that feeds the Phase 6 preprocessing pipeline unchanged.
3. History windowing and truncation to bound token cost.
4. UI: send session history with each request.
5. Evaluation: 10 to 15 scripted multi-turn scenarios added to the eval suite.

Exit criteria: multi-turn scenarios pass, deployed UI supports refinement. README v7.

## Phase 8: Inventory-Aware Composition + Build Sheet Render (firm 2 weeks, target Aug 10 to Aug 21)

Goal: compose bouquets from individual-flower inventory under constraints instead of retrieving only fixed products, and give the customer both a rendering of the result and instructions to assemble it. Absorbs the inventory task started early under the old numbering.

Scope:

1. Day-one architecture decision, recorded in the decisions log: agentic tool-use loop (check stock → compose → validate → revise) versus single-pass constrained generation. This decision controls how the capability is described everywhere.
2. Inventory data model: flowers with stock levels, price, color, seasonality. Kept coherent with the Phase 6 catalog schema.
3. Constraint-aware composer: budget, availability, occasion fit, pairing rules. Build sheet output: specific flowers, quantities, and assembly instructions.
4. Assembly instructions: ordered steps a non-florist can follow, generated from the build sheet spec and returned as a structured field. Programmatic checks: steps reference only build sheet flowers, every build sheet flower appears in the steps, ordering is sane. Clarity optionally scored by reusing the LLM-as-judge harness with an instruction rubric.
5. Programmatic constraint evaluation: composed outputs checked automatically against budget, stock, and occasion constraints, plus the instruction checks above.
6. Retrieval path and composition path coexist (quick recommendation versus custom composition).
7. Minimal render: one generated image per build sheet, produced from the spec, cached keyed on the spec, no editing. Provider decided at task start; integration kept swappable per the model-independence constraint.

Note: this is the fullest phase. If the composer overruns, item 4 is the designated slip item and moves to a fast-follow PR; nothing downstream depends on it.

Exit criteria: composition endpoint live, constraint and instruction checks passing, build sheet with steps and render visible in the UI, architecture decision documented. README v8.

## Phase 9: Guardrails (~1 week, target Aug 24 to Aug 28)

Goal: the system fails safely and cheaply.

Scope:

1. Off-topic filtering with graceful rejection, reusing the Phase 6 query classifier.
2. Grounding check: recommended or composed items, including flowers referenced in assembly instructions, must exist in the catalog or inventory; violations are blocked and logged.
3. PII redaction on inbound text before logging and LLM calls. Mechanism decided at task start.
4. Cost controls: per-request token budgets; response cache with multi-turn-aware keys.
5. Adversarial and edge-case tests for each guardrail.

Exit criteria: guardrails active on all request paths, tests merged, deployed. README v9.

## Phase 10: Observability + Feedback (~1 week, target Aug 31 to Sep 4)

Goal: every request is traceable, quality regressions surface automatically, and feedback is captured through a source-agnostic pipeline and used at least once.

Scope:

1. LangSmith tracing across preprocessing, retrieval, reranking, generation, and guardrails.
2. Scheduled evaluation in CI (cron): full retrieval eval against the live pipeline; the workflow alerts when recall or F1 drops below thresholds.
3. Feedback event schema and ingestion endpoint per the design principles above; UI thumbs up and down wired in as the first producer, logged with request IDs.
4. Feedback analysis script: correlate ratings with retrieval scores and latency; run one documented tuning iteration from the findings (Tier 1 pilot signals when available, otherwise ratings generated during eval runs).

Exit criteria: traces visible end to end, scheduled eval running with thresholds, one feedback-driven change shipped and documented. README v10.

## Phase 11: Simulator + Experimentation Harness + Final Polish (~1.5 weeks, target Sep 7 to Sep 16)

Goal: realistic labeled synthetic traffic, controlled comparison of pipeline variants, repo fully interview-ready.

Scope:

1. LLM user simulator: persona-driven single and multi-turn sessions against the public API (depends on Phase 7), writing feedback events through the Phase 10 ingestion path with source set to simulator.
2. Config-driven variants: variant registry in configuration; two prompt or retrieval configurations compared offline against the same eval set, metrics reported side by side.
3. Variant routing in the API: deterministic assignment by hashing session_id so a session stays in one arm; exposure logging; per-arm metric computation with a significance test in the report. Simulator traffic demonstrates the mechanism end to end.
4. Coverage pushed toward 90 percent or higher; the CI coverage gate from Phase 6 switched on at the achieved level.
5. Final pass: README, architecture diagram, learning docs consistency, deployment verification.

Exit criteria: simulator producing labeled synthetic sessions, variant comparison documented with results, coverage gate active, docs consistent. README v11.

## Phase 12: Iterative Image Feature (~1 to 1.5 weeks, target Sep 17 to Sep 25)

Goal: the build sheet is the source of truth for the visual; customers can refine the design and trust that the image reflects the spec.

Scope:

1. Spec-driven editing: an edit request ("make it more pink") modifies the build sheet spec, which is re-rendered. Consistency is maintained at the spec level; pixel-level continuity between renders is best effort and depends on the provider's editing endpoints (decided at task start).
2. VLM faithfulness evaluation: a vision model judges whether the rendered image shows the flowers and colors the build sheet lists; results tracked in the eval suite.
3. Image cost controls: the Phase 8 render cache extended to edit chains; per-session render budget.
4. UI: edit flow in the chat interface.

Out of scope, tracked outside the repo: drag-and-drop editing, compositing renders into uploaded photos of venues or spaces.

Exit criteria: edit flow live, faithfulness eval reporting, render budget enforced. README v12.

## Backlog (not scheduled)

- Domain-knowledge retrieval collection: a second vector store collection of flower domain knowledge (symbolism, seasonality, care, assembly techniques) built from openly licensed sources used with attribution (e.g., Wikipedia, CC BY-SA), queried alongside the product collection. Estimated 3 to 4 days including eval additions. Go or no-go decided at Phase 6 close.
