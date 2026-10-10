# Prompt evaluation golden set

These small, hand-authored examples are starter evaluation data for the router
and advisor prompt experiments. Router cases are synthetic classification
examples. Advisor inputs use passages retrieved from this project's local
Qdrant index, with the retrieved text and source metadata copied into each
record. `GOLDEN-*` customer IDs are placeholders, not real customers.

Files ending in `train.jsonl` are for prompt iteration/optimization. Files
ending in `holdout.jsonl` should be reserved for comparing a candidate against
the baseline after tuning. Do not tune on holdout cases and then report those
same cases as independent validation.

Each JSONL record has:

- `case_id`: stable example identifier.
- `inputs`: model inputs. Router cases provide `query_text`; advisor cases also
  provide `context` and `retrieved_passages` in the shapes used by the app.
- `outputs`: target output (`{"intent": ...}` for router, answer text for
  advisor).
- `expectations`: criteria for task-specific evaluation scorers. These are
  metadata for evaluation, not text to include in the model prompt.

The advisor cases store the top four Qdrant passages per query, matching the
application's current retrieval limit. The query text, passage text, citation
ID, source URL, publication, page, section, and retrieval score were captured
from the local index. Re-run retrieval after rebuilding the index: passage
ranks, scores, and citation IDs may change. These snapshots preserve the
evidence shown to the model for this evaluation run.

The `advisor_*_context_labels.jsonl` files provide hand-reviewed
`relevant_context_ids`, `required_evidence`, and per-passage
`evidence_by_context` annotations for context precision/recall. Context
precision is P@4: the fraction of retrieved passage IDs labeled relevant.
Context recall is the fraction of required evidence facets represented across
the relevant retrieved passages. Some retrieved passages are deliberately
labeled irrelevant to make P@4 discriminative. These are human-authored
judgments and should be reviewed when the data or source index changes.

Use exact-match accuracy for the router as an initial metric. Advisor
groundedness and answer relevancy are scored by a Gemini judge against the
retrieved passages and reference answer; context precision/recall use the
hand-reviewed labels above. The target answers are examples of acceptable
responses, not the only valid wording. Judge scores are approximate, can vary,
and cost API calls. Inspect outputs, judge rationales, and scorer failures
manually; this small set is a smoke-test seed, not sufficient evidence for
production or regulatory claims.

The advisor passages come from FCA publications listed in the project source
catalog. Review the source PDFs and current FCA material before relying on any
interpretation; this starter dataset is not legal or compliance approval. Keep
any customer data out of this dataset.
