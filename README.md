# MarketReady Funds

MarketReady Funds is a Project B educational prototype for FINS5545. It turns
the supplied equity, crypto, and headline data into 12 separately investable
systematic fund simulations, a coverage-aware sector sentiment analytic, and a
decision-focused Streamlit interface.

The build is complete and tested. The app is live on Streamlit Community Cloud
and deployed from the public `lgua9087/z5603162_projectB` GitHub repository.

## Product experience

The app supports five linked views:

1. **Fund marketplace** — filter and compare all 12 out-of-sample products.
2. **Fund fact sheet** — inspect growth, drawdown, metrics, and latest targets.
3. **Allocation lab** — combine up to five funds, simulate a normalized mix,
   and inspect equity/crypto, sector, concentration, and holding look-through.
4. **News sentiment** — read sector tone beside ticker-news coverage and the
   before/after fusion result.
5. **Method and risks** — review the information clock, assumptions, and limits.

URL query parameters retain the selected view, and every investor-facing table
or simulated path can be downloaded. The deployed app remains lightweight:
it reads verified artifacts from `results/` and never downloads raw data, scores
VADER, or solves a portfolio.

## Reproduce and test

Run these commands from this standalone repository root on macOS/Linux:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python scripts/run_part_b.py
.venv/bin/python scripts/build_report.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
.venv/bin/python -m ruff check streamlit_app.py src scripts tests
.venv/bin/python scripts/check_handin.py
```

To inspect this project app locally after the artifacts exist:

```bash
.venv/bin/python -m streamlit run streamlit_app.py
```

The Week 2 sample app was used only as a static design reference; it was not
executed during this build.

## Method in brief

- Source sample: 1 January 2020 to 31 December 2023.
- Live evidence: 1 January 2021 for crypto and 4 January 2021 for equity and
  combined products, through 31 December 2023.
- Families: equity, crypto, and combined; methods: Equal Weight, Minimum
  Variance, Maximum Sharpe, and Risk Parity.
- Monthly walk-forward targets use only prior observations. Holdings drift
  between rebalances; turnover is measured against pre-trade weights and costs
  5 bp per unit.
- Equity and combined results use 252-period annualisation; crypto uses 365.
  Sharpe ratios use a 0% risk-free rate.
- The sentiment model preserves original headline grammar and compares plain
  VADER 3.3.2 with a reviewed finance lexicon. Tone and coverage are both lagged
  one trading day before the equity Risk Parity tilt uses them.

The highest sample Sharpe is Crypto Minimum Variance (1.03), but its maximum
drawdown is 71.9%. Combined Risk Parity produces a 0.89 Sharpe with a 19.5%
drawdown. Its median crypto capital weight is 7.6%, while crypto contributes
16.7% of covariance risk by construction. The finance extension changes 3.2%
of headline scores. Cross-sectional validation does not show a persuasive
positive sentiment ranking: mean IC is 0.006 at one day and -0.002 at five days,
while the five-day high-minus-low spread is -4.7 bp. Consistently, the
coverage-aware equity tilt lowers Sharpe from 0.722 to 0.695; its maximum
drawdown improves by 0.46 percentage points.

## Repository map

- `streamlit_app.py` — deployment entrypoint.
- `src/` — ETL, features, portfolios, sentiment, fusion, figures, and app helpers.
- `scripts/run_part_b.py` — deterministic end-to-end model build.
- `scripts/build_report.py` — Word/PDF report builder from committed results.
- `results/data/` — app-readable derived artifacts.
- `results/tables/` and `results/figures/` — report evidence.
- `report/report.docx` and `report/report.pdf` — editable source and submission.
- `tests/test_smoke.py` — model invariants and headless tests for all app views.
- `ai/` and `AGENTS.md` — transparent AI workflow evidence and operating rules.

## Verification status

- 23 of 23 pipeline validations pass.
- 25 automated tests pass, including all five Streamlit views and the final
  look-through/default-selection checks.
- All six required figures and the fusion comparison table are present.
- The 17-page PDF has 10 narrative pages, followed by references and appendices;
  every page was rendered and visually inspected.

This is an educational backtest, not investment advice. Reported net returns
deduct estimated turnover costs but not management fees. Historical results are
sample-specific and omit taxes, bid-ask spreads beyond the cost estimate,
market impact, and capacity constraints.

## Live deployment

- Live app: <https://z5603162projectb-zdwhgwh2jhcput7tn6w2ku.streamlit.app/>
- Public GitHub repository: <https://github.com/lgua9087/z5603162_projectB>
- Deployment branch: `main`
- Entrypoint: `streamlit_app.py`

The public repository contains the complete submission project. The app reads
the required precomputed artifacts and the repository excludes raw data,
secrets, and caches.
