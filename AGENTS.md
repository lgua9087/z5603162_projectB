# MarketReady Funds agent instructions

These are the canonical working instructions for `z5603162_projectB`. Read
`PROJECT_BRIEF.md`, `context/DATA_GUIDE.md`, and `context/project_context.md`
before changing the analysis. Work only inside this project directory.

## Product and scope

Build MarketReady Funds: an educational Streamlit product that lets a
financially literate self-directed investor compare systematic funds, inspect a
fund fact sheet, simulate an allocation, and view equity-news sentiment. Part B
covers portfolio and sentiment modelling plus implementation. Deployment is a
separate, later browser stage.

The supplied equity, crypto, and headline datasets are the only raw inputs.
Load them through `src/data_access.py`; never copy or commit raw data. Commit
derived, app-readable artifacts under `results/`.

## Modelling rules

- Compute adjusted-close simple returns within ticker. Use each asset family's
  native calendar for its standalone funds and the equity calendar for combined
  funds.
- Run monthly walk-forward backtests. Targets at date `t` may use observations
  only through `t-1`; the initial 252-equity-day or 365-crypto-day estimation
  window must finish before the first live return.
- Keep portfolios long-only and fully invested. Apply the documented per-asset
  caps and 50% mean / 10% covariance shrinkage consistently.
- Let holdings drift with returns between rebalances. Compute turnover against
  actual pre-trade weights and deduct 5 basis points per unit of realised
  turnover on a rebalance date.
- Use 252 periods for equity and combined annualisation and 365 for crypto. The
  reported Sharpe ratio assumes a 0% risk-free rate.
- Score original headline text without stripping case, punctuation, negation,
  boosters, or contrast words. Use VADER 3.3.2 plus the reviewed finance
  extension and record both model distributions.
- Align weekend/non-trading-day headlines forward to the next equity trading
  day. Lag both sentiment tone and coverage by at least one trading day before
  a position can use them. Never turn missing news into evidence of neutral
  sentiment without saying so.
- Treat the coverage-aware sentiment tilt as a tested research extension, not a
  promised source of alpha. Validate its cross-sectional ranking with short-
  horizon IC and high-minus-low spreads before characterising signal strength.
- For combined funds, report equity/crypto capital weights, covariance-based
  family risk contributions, and at least one concentration measure. Distinguish
  risk allocation from capital allocation in investor-facing explanations.

## Code and artifact conventions

- Keep pure, reusable functions in `src/`; runnable orchestration belongs in
  `scripts/`; tests belong in `tests/`.
- `scripts/run_part_b.py` is the single model build entrypoint. It must write the
  exact required filenames and fail if validation checks fail.
- `streamlit_app.py` and `src/app_*` may read only precomputed CSV/JSON artifacts.
  The app must not download raw data, import VADER, or run an optimiser.
- Preserve stable identifiers (`fund_id`, `family`, `method`, ISO dates) in data
  artifacts; reserve presentation labels and percentage formatting for the app
  or report layer.
- Keep deploy dependencies in `requirements.txt`; put reproduction, testing,
  and document-build packages in `requirements-dev.txt`.
- Use clear type hints, short functions, vectorised pandas where reasonable, and
  deterministic ordering. Do not suppress failed checks or silently repair
  invalid weights.

## Verification gates

Before declaring the project complete:

1. Rebuild with the repository Python environment.
2. Require every row of `results/tables/validation_results.csv` to pass.
3. Run the smoke and AppTest suite with pytest and lint authored Python with
   Ruff.
4. Run `scripts/check_handin.py` and remove generated cache clutter.
5. Confirm every required exhibit exists, has a self-contained caption, and is
   interpreted in the report.
6. Render every PDF page and visually inspect it. If Word rendering is not
   available, disclose that limitation and perform structural DOCX checks.
7. Do not initialise, publish, or deploy the repository unless explicitly asked.

## AI transparency and human review

Record substantial prompts, outputs, risks, and corrections in `ai/`. Attribute
assistant work honestly; do not imply that the student independently found an
issue when an automated audit found it. The student remains responsible for
reviewing the economic argument, checking citations, and expressing the final
submission in their own words.
