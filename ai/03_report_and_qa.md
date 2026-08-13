# Prompt log — report generation and QA

## What I wanted

Turn the verified artifacts into a complete Word-first report and submission
PDF while keeping all work inside the Project B directory.

## Prompt(s)

The report followed the same completion request, the Project B exhibit list,
the repository's Word-report rules, and the requirement to defer deployment.

## What the assistant produced

`scripts/build_report.py` reads committed result artifacts and creates an
editable `report/report.docx` plus a matching `report/report.pdf`. The design
uses an A4 editorial layout, built-in Word styles, numbered captions and
cross-references, explicit table widths, alt text, page fields, and six
self-contained figures. The narrative occupies pages 1–10; references are page
11 and appendices are pages 12–14.

## What was wrong or risky

- The first PDF composition gave report figures fixed width and height, which
  visibly distorted their aspect ratios and pushed later content to extra pages.
- A report could quote stale or hand-entered results if it did not build from
  the exact CSV/JSON artifacts used by the app.
- The first architecture subheading used “deployed” even though deployment was
  explicitly deferred.

## How it was corrected and checked

The PDF builder now reads each source image's dimensions and scales it
proportionally. At that stage, the report was rebuilt to 14 pages and every page was rendered
to PNG and visually inspected for clipping, overflow, table fit, figure
legibility, and pagination. The wording was changed to “A lightweight
deployment architecture.” That build's structural audit found 114 paragraphs,
25 headings, 14 captions, eight tables, and six figures in the DOCX. Numerical
statements are populated from the final result artifacts.

LibreOffice was unavailable in the environment, so the editable DOCX could not
be rendered through Word-compatible software here. Its package structure was
audited, and the PDF generated from the same content model passed full visual
inspection.

## Human review still required

The student must read the report, verify the cited course material and VADER
paper, and revise the interpretation into wording they can explain and defend.
The assistant-generated prose should not be submitted as unreviewed personal
writing.
