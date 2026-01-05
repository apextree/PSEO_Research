# PSEO Earnings and Flow Dashboard

This dashboard visualizes earnings and flow outcomes for Education and Computer Science graduates using US Census Bureau Post-Secondary Employment Outcomes data.

## Running the Dashboard

### Method 1: Clone and Run
1. Clone the repository:
`git clone https://github.com/anubhavdhungana/PSEO_Research.git`
2. Install the required dependencies:
`pip install -r requirements.txt`
3. Run the application:
`streamlit run app.py`

### Method 2: Use Streamlit Cloud
1. Access the app at https://pseoresearch-ldtdxel35fbkurmpkwpmpv.streamlit.app/

## Methodology

The data is sourced from the US Census Bureau PSEO earnings files. The following transformations were applied:

* Filtered the raw dataset for CIP 11 (Computer Science) and CIP 13 (Education).
* Utilized 2 digit CIP aggregates (Levels 28, 34, 40, and 46) rather than 4 digit codes. This ensures better data coverage for Master's and Doctoral programs which often face suppression at more granular levels.
* Converted suppressed data values (represented as zeros or negatives in raw files) to null values to ensure accurate mean and percentile calculations.
* Aggregated cohort years to match standard PSEO reporting periods.

## Data Limitations and Interpretation

* Privacy Suppression: The Census Bureau hides data for programs with small graduate counts to protect student identity. Some institutions may show as having no data for specific cohorts or degree levels.
* Due to privacy Suppression, only CIP level 2 could be used for the visualizations as CIP level 4 significantly reduced available data for institutions, degree levels, and cohorts -- especially for flow data.