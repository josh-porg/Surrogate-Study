# Reference material (small, committed)

## math796_hw1/
Copied 2026-10-03 from `OneDrive - University of Kansas/MATLAB/Autumn 2025/Math 796/HW_1/` (author's own work).
- `optimization_test_functions/<category>/*.m` — 32 two-dimensional test functions from the Surjanovic & Bingham
  Virtual Library of Simulation Experiments (https://www.sfu.ca/~ssurjano/), grouped by the VLSE categories:
  many_local_minima (9), bowl_shaped (5), plate_shaped (5), valley_shaped (4), steep_ridges_and_drops (3), other (6).
- `HW1_script.m` — the driver comparing fminunc, patternsearch, ga, particleswarm, simulannealbnd, surrogateopt and
  ga/SA + fminunc hybrids. Report: `Math 796 .../MATH_796_HW_1.pdf`.

These are the "optimization test surfaces" the paper's canonical suite starts from (ported to Python in `code/`).
Caveats on the HW1 results themselves are logged in docs/BUGS.md (H-01..H-04).

Not copied, by request: the MATLAB DMD / flow-field data in `OneDrive/research` and `OneDrive/MATLAB/DMD` (large).
