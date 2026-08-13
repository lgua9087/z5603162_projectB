# AI use, attribution, and final review notes

## Attribution

Codex performed the implementation, automated testing, artifact generation,
report composition, and visual QA in this build. The student supplied the zID
project folder and scope, directed the app-first order, prohibited execution of
the Week 2 sample, constrained writes to this directory, and deferred
deployment. These notes intentionally distinguish those directions from work
performed by the assistant.

The assistant also consulted the Project B brief, the provided context and data
access helper, the Week 8 VADER compound-score example, and the official VADER
paper cited in the report. No other student's project was inspected or copied.

## Where AI helped

- Mapped the required investor journey into five app views and a precomputed
  artifact architecture.
- Implemented and cross-checked the three data streams, four portfolio methods,
  finance lexicon, coverage-aware fusion, and report exhibits.
- Turned important assumptions into executable validations and regression tests.
- Added cross-sectional IC/spread diagnostics, combined-fund risk contribution
  and concentration measures, and limited annual/cost/fee/parameter robustness.
- Found and corrected a daily-rebalancing mismatch, a sentiment coverage timing
  error, and a report-image layout error during iterative audits.
- Generated a transparent negative-result interpretation instead of claiming
  that the sentiment extension added alpha.

## Remaining limitations

- Three live years are short and include one unusually large crypto cycle.
- VADER scores headline tone, not truth or realised market impact; no labelled
  finance validation set is available.
- Primary reported net returns deduct estimated turnover costs but not management
  fees; a 0.0%/0.5%/1.0% fee sensitivity is reported separately. The prototype
  still omits taxes, market impact, capacity, and detailed order execution.
- The allocation lab is a historical simulation, not a suitability assessment.
- The app has passed headless tests but has not yet been deployed or checked in
  a public browser environment.
- The DOCX passed structural checks but could not be visually rendered because
  LibreOffice was unavailable; the matching PDF was fully rendered and checked.

## Evidence and reproducibility

The final automated state is 23/23 pipeline validations, 26 passing tests, clean
Ruff checks on authored Python, and a passing hand-in script. Exact result tables
are under `results/tables/`; app artifacts are under `results/data/`; figures are
under `results/figures/`.

## Student actions before submission

1. Review the economic argument and rewrite any sentence that is not in your own
   voice or that you cannot explain.
2. Manually verify at least one return, one portfolio rebalance/turnover value,
   and one headline's next-trading-day lag; add the evidence to these logs.
3. Open `report.docx` in Word, update cross-reference fields if prompted, and
   visually compare it with `report.pdf`.
4. Run the app locally and inspect every view at desktop and narrow widths.
5. Confirm the final public GitHub and Streamlit URLs remain accessible before
   submitting the report and repository.
