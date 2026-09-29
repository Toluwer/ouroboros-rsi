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
| 1 | self-improve (sample 0 + rationalize 0, train 50 steps) | 0.300 | rolled back | incumbent kept |
| 2 | self-improve (sample 0 + rationalize 0, train 50 steps) | 0.217 | rolled back | incumbent kept |
| 3 | self-improve (sample 0 + rationalize 0, train 50 steps) | 0.317 | rolled back | incumbent kept |
| 4 | self-improve (sample 0 + rationalize 0, train 50 steps) | 0.267 | rolled back | incumbent kept |
| 5 | self-train (verified 6 + hard-mined 6, train 50 steps) | 0.333 | rolled back | incumbent kept |
| 6 | self-train (verified 12 + hard-mined 0, train 50 steps) | 0.300 | rolled back | incumbent kept |
| 7 | self-train (verified 44 + hard-mined 16, train 180 steps) | 0.340 | rolled back | incumbent kept |
| 8 | self-train (verified 120 + hard-mined 46, train 150 steps) | 0.324 | rolled back | incumbent kept |
| 9 | self-train (verified 125 + hard-mined 40, train 120 steps) | 0.396 | accepted | `ouroboros-v2` (+0.032) |
| 10 | self-train (verified 90 + hard-mined 23, train 120 steps) | 0.380 | rolled back | incumbent kept |
| 11 | self-train (verified 132 + hard-mined 37, train 120 steps) | 0.348 | rolled back | incumbent kept |
| 12 | self-train (verified 139 + hard-mined 37, train 120 steps) | 0.388 | rolled back | incumbent kept |
