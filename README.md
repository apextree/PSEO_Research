# PSEO Earnings and Flow Dashboard

This dashboard visualizes earnings and flow outcomes for Education and Computer Science graduates using US Census Bureau Post-Secondary Employment Outcomes data.

## Running the Dashboard

1. Install the required dependencies:
`pip install -r requirements.txt`
2. Run the application:
`streamlit run app.py`

## Methodology

The data is sourced from the US Census Bureau PSEO earnings files. The following transformations were applied:

* Filtered the raw dataset for CIP 11 (Computer Science) and CIP 13 (Education).
* Utilized 2 digit CIP aggregates (Levels 28, 34, 40, and 46) rather than 4 digit codes. This ensures better data coverage for Master's and Doctoral programs which often face suppression at more granular levels.
* Converted suppressed data values (represented as zeros or negatives in raw files) to null values to ensure accurate mean and percentile calculations.
* Aggregated cohort years to match standard PSEO reporting periods.

## Data Limitations and Interpretation

* Privacy Suppression: The Census Bureau hides data for programs with small graduate counts to protect student identity. Some institutions may show as having no data for specific cohorts or degree levels.
* Rounding: All earnings figures are rounded by the Census Bureau to the nearest hundred dollars.
* Geographic Scope: Outcomes represent where graduates are employed. The data does not account for regional cost of living differences which may impact the perceived value of the earnings listed.
* Inflation: Earnings are presented in nominal dollars based on the reporting period and are not adjusted for current inflation rates.