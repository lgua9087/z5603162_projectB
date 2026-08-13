# Prompt log — model build and critical audit

## What I wanted

Finish Project B after the app-first pass, using the supplied equity, crypto,
and headline data and the Week 8 VADER compound-score example.

## Prompt(s)

The controlling request was to “find all required information [and] finish the
projectB in this run,” with the linked Week 8
`vader_model/04_from_valence_to_compound.py` supplied as a reference.

## What the assistant produced

The assistant built a deterministic pipeline for cleaned native-calendar
returns, 12 monthly walk-forward funds, plain and finance-extended VADER,
10 sector indices, a one-day-lagged coverage-aware equity tilt, six report
figures, report/app artifacts, and 18 explicit validation checks. The portfolio
suite covers three families and four methods with family-specific annualisation,
long-only caps, shrinkage, turnover, and costs.

## What was wrong or risky

Two material implementation problems were found during the assistant's own
audit before finalisation:

1. An early portfolio path applied the target weights every day. That silently
   described daily rebalancing while the report and design called the strategy
   monthly buy-and-hold.
2. An early sentiment signal lagged tone but allowed the coverage scaler to use
   the current date, and then penalised coverage a second time. That mixed the
   information clocks and weakened the signal for the wrong reason.

Optimised expected-return portfolios also carry an economic risk even when the
code is correct: noisy historical means can make Maximum Sharpe targets unstable.

## How it was corrected and checked

The portfolio engine now lets weights drift daily and computes each rebalance's
turnover against actual pre-trade holdings before deducting 5 bp per unit. Both
the rolling tone and rolling coverage are shifted by one equity trading day,
and the coverage scaler is applied once. Regression tests reproduce these
specific failure modes. The pipeline then passed 18 of 18 validations and the
full test suite passed 18 tests.

The outcome was not polished into a positive claim. Crypto Minimum Variance has
the highest Sharpe (1.03) but a 71.9% maximum drawdown. The sentiment tilt lowers
Sharpe from 0.722 to 0.695 while improving maximum drawdown by only 0.46
percentage points and materially increasing turnover.

## Human review still required

The corrections above were made by the assistant's audit, not independently by
the student. Before submission, the student should trace one rebalance and one
Monday-to-Tuesday headline example by hand and record any independent finding
or change here.
