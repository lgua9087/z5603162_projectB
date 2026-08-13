# Prompt log — evidence strengthening and commercial interpretation

## What I wanted

Review the existing Project B report, code, and generated results while
preserving the report structure, visual style, core findings, and 10-page
narrative limit. Strengthen sentiment validation, combined-fund risk
diagnostics, robustness, methodological wording, and investor relevance without
inventing results or adding unnecessary methods.

## What the assistant produced

The model pipeline now generates four linked evidence extensions:

1. One- and five-day cross-sectional sentiment Information Coefficients and
   high-minus-low sentiment portfolio spreads with HAC t-statistics.
2. Equity/crypto capital weights, covariance-based family risk contributions,
   effective holdings, and top-five concentration for combined funds.
3. Annual 2021–2023 fund results plus limited 0/5/10 bp trading-cost,
   0%/0.5%/1% management-fee, and two sentiment-parameter sensitivities.
4. Exact estimation-window imputation counts and explicit optimiser-fallback
   disclosure.

The app's allocation lab also exposes look-through holdings, top-five
concentration, and crypto exposure so overlapping funds cannot be mistaken for
independent diversification.

## Evidence and interpretation

Over the matching 2021–2023 live fund period, the mean cross-sectional IC is
0.006 at one day and -0.002 at five days. The five-day high-minus-low sentiment
return is -4.7 bp, and neither horizon gives statistically persuasive positive
evidence. The primary coverage-aware tilt
therefore underperforms the equity Risk Parity base in a sample where its
ranking diagnostic does not support positive predictiveness. Its additional
trading-cost drag is 0.049 percentage points, so turnover worsens—but does not
fully explain—the return shortfall.

Combined Risk Parity has a median crypto capital weight of 7.6% but a fixed
16.7% crypto covariance-risk contribution because ten of the sixty assets are
crypto and asset-level risk contributions are equalised. Combined Maximum
Sharpe is substantially more concentrated and has more variable crypto capital
exposure, supporting a more cautious product interpretation than its Sharpe
rank alone.

The annual tables preserve the pronounced regime dependence across 2021, 2022,
and 2023. The combined Risk Parity conclusion is qualitatively stable across
the limited cost and management-fee assumptions, while all tested sentiment
tilts remain behind the untitled equity Risk Parity base. These are sample
robustness checks, not forecasts.

## What was risky and how it was controlled

- The diagnostics use only lagged signals and future returns at fixed short
  horizons; no parameter was selected to maximise the reported outcome.
- Robustness uses a small set of economically interpretable alternatives rather
  than a broad sweep.
- Primary “net” performance is labelled net of estimated trading costs and
  before management fees.
- The pipeline reports zero mean-filled optimiser cells across 1,129,320
  estimation-window cells and names the single solver fallback event and rule.
- New prose was generated from committed CSV/JSON artifacts used by the app,
  with secondary diagnostics placed in appendices to retain ten narrative pages.

## Human review still required

The student must verify the economic interpretation, citations, and numerical
examples, then revise any wording they cannot explain in their own voice.
Deployment is still outstanding, so the report does not claim that a public
GitHub repository or Streamlit URL exists.
