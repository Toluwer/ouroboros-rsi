# Evolution Log

Append-only record of every improvement attempt, written by the system
itself. The daemon appends one row per cycle; humans do not edit history
rows. Rejected cycles are recorded and rolled back — the failures are part
of the record.

Scoring: fixed 250-question CommonsenseQA validation subset, direct-answer
prompt, greedy decoding. A cycle is accepted only if it beats the incumbent
by more than 0.004 accuracy (constitution: `accept_margin`).

| cycle | action | held-out acc | outcome | checkpoint |
|---|---|---|---|---|
| 0 | phase 0 (operator SFT on CommonsenseQA train) | 0.364 | accepted | baseline checkpoint `ouroboros-v0` |
