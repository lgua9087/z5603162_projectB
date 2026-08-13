# Prompt log — app-first build

## What I wanted

Inspect the Streamlit sample code under Week 2 without executing it, use it as a
design hint, build the Project B app first, and write only inside
`z5603162_projectB`.

## Prompt(s)

> “Inspect contents under week2 folder, there are streamlit app sample code. do
> not run it, but this will be a hint on how to write the app for projectB. I
> need you to write app first. The deployment will be done in later stage. Once
> you find all required information, finish the projectB in this run. only write
> things under projects/z5603162_projectB directory.”

## What the assistant produced

The assistant read the Week 2 source files and Project B brief statically, then
built a distinct app rather than copying the sample. `streamlit_app.py` provides
a marketplace, 12 consistent fact sheets, an allocation lab, sector sentiment,
a fusion diagnostic, and a method/risk page. `src/app_data.py` validates and
loads committed artifacts; `src/app_charts.py` contains reusable Plotly figures.
The app uses URL-seeded view state, cached reads, downloadable tables, and no
build-time dependency.

## What was wrong or risky

- A deployment app that imported VADER or reran optimisation would be slow and
  fragile on the free tier.
- The initial explanatory copy could have implied zero transaction costs even
  though the pipeline deducts 5 bp per unit of turnover.
- Plotly calls needed the current Streamlit width API rather than a deprecated
  container-width argument.
- A UI can appear complete while one route fails only after state is seeded.

## How it was corrected and checked

The app was separated from the build pipeline and made dependent only on the
verified `results/` CSV/JSON files. The cost disclosure was corrected, Plotly
calls use `width="stretch"`, and headless Streamlit AppTest cases open all five
views. A clean marketplace load completed in about 3.2 seconds on the local
course environment. The Week 2 app itself was never run.

## Human review still required

The student should open the finished app locally before deployment and confirm
that labels, default funds, colors, and customer language match the intended
product positioning.
