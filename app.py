import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CONFIGURATION & DATA LOADING ---
st.set_page_config(page_title="PSEO Earnings Dashboard", layout="wide")

# Custom CSS for High Contrast Text
st.markdown("""
<style>
    h1, h2, h3, h4, h5, h6, p, label, .stRadio, .stMultiSelect {
        color: #ffffff;
        font-family: 'Helvetica', sans-serif;
    }
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(mode):
    
    if mode == 'All Cohorts':
        # Load the Foundation file (Agg 28)
        df = pd.read_csv('Earnings_Data/state_foundation_all_years.csv', dtype={'cipcode': str, 'degree_level': str})
        df['Cohort Group'] = 'All Cohorts' # Placeholder
    else:
        # Load the Trends File (Agg 34)
        df = pd.read_csv('Earnings_Data/state_cohort_trends.csv', dtype={'cipcode': str, 'degree_level': str})
    
    # Map Major Codes
    def get_major_family(code):
        if str(code).startswith('13'): return 'Education'
        if str(code).startswith('11'): return 'CompSci / AI'
        return 'Other'
    
    if 'Major Family' not in df.columns:
        df['Major Family'] = df['cipcode'].apply(get_major_family)

    # Map Degree Levels
    degree_map = {
        '01': 'Certificate <1yr', '02': 'Certificate 1-2yrs',
        '03': 'Associates', '05': 'Bachelor\'s',
        '07': 'Master\'s', '17': 'Doctoral'
    }

    df['Degree Label'] = df['degree_level'].astype(str).map(degree_map).fillna(df['degree_level'])
    
    return df[df['Major Family'].isin(['Education', 'CompSci / AI'])]


# --- 2. SIDEBAR FILTERS ---
st.sidebar.header("Controls")
# COHORT SELECTOR
cohort_options = [
    'All Cohorts', 
    '2001-2003', '2004-2006', '2007-2009', 
    '2010-2012', '2013-2015', '2016-2018', '2019-2021'
]
selected_cohort = st.sidebar.selectbox("Select Graduation Cohort", cohort_options)


timeframe_label = st.sidebar.radio(
    "Select Timeframe",
    options=['1 Year Post-Grad', '5 Years Post-Grad', '10 Years Post-Grad'],
    index=0
)
time_code = {'1 Year Post-Grad': 'y1', '5 Years Post-Grad': 'y5', '10 Years Post-Grad': 'y10'}[timeframe_label]

st.sidebar.markdown("---")

available_degrees = ['Certificate <1yr', 'Associates', 'Bachelor\'s', 'Master\'s', 'Doctoral']
selected_degrees = st.sidebar.multiselect(
    "Select Degree Level(s)",
    options=available_degrees,
    default=['Bachelor\'s', 'Master\'s']
)

available_majors = ['Education', 'CompSci / AI']
selected_majors = st.sidebar.multiselect(
    "Select Major(s)",
    options=available_majors,
    default=['Education', 'CompSci / AI']
)

# --- 3. DATA PREPARATION ---

if selected_cohort == 'All Cohorts':
    df = load_data("All Cohorts")
else:
    raw_df = load_data("Trends")
    #filtering for specific bucket
    df = raw_df[raw_df['Cohort Group'] == selected_cohort].copy()

if not selected_degrees or not selected_majors:
    st.error("Please select at least one Degree Level and one Major.")
    st.stop()

mask = (df['Degree Label'].isin(selected_degrees)) & (df['Major Family'].isin(selected_majors))
filtered_df = df[mask]

cols_to_use = [f'{time_code}_p25_earnings', f'{time_code}_p50_earnings', f'{time_code}_p75_earnings']
agg_df = filtered_df.groupby(['Major Family', 'Degree Label'])[cols_to_use].mean().reset_index()

plot_data = agg_df.melt(
    id_vars=['Major Family', 'Degree Label'],
    value_vars=cols_to_use,
    var_name='Metric',
    value_name='Earnings'
)

metric_map = {
    f'{time_code}_p25_earnings': '25th Percentile (Entry)', 
    f'{time_code}_p50_earnings': '50th Percentile (Median)', 
    f'{time_code}_p75_earnings': '75th Percentile (Top Tier)'
}
plot_data['Percentile'] = plot_data['Metric'].map(metric_map)



# --- 4. VISUALIZATION ---
st.title("Earnings Comparison Dashboard")
st.markdown(f"**Viewing:** {timeframe_label} Outcomes")

facet_col_arg = "Major Family" if len(selected_majors) > 1 else None

fig = px.bar(
    plot_data,
    x="Degree Label",
    y="Earnings",
    color="Percentile",
    facet_col=facet_col_arg,
    barmode="group",
    category_orders={
        "Degree Label": available_degrees, 
        "Percentile": ['25th Percentile (Entry)', '50th Percentile (Median)', '75th Percentile (Top Tier)']
    },
    color_discrete_map={
        '25th Percentile (Entry)': '#A8DADC',      
        '50th Percentile (Median)': '#457B9D',     
        '75th Percentile (Top Tier)': '#1D3557'    
    },
    height=800
)

# --- 5. UNIFIED STYLING (The Fix) ---

# FIX A: Force Consistent Bar Text (100k format)
# This ensures both Left and Right bars show "$50k" instead of "$50000"
fig.update_traces(
    texttemplate='%{y:$.3s}', # "$.2s" means Dollar + 2 sig digits + "SI" suffix (k/M)
    textposition='inside'    # Puts the number on top of the bar
)

# FIX B: Target ALL Axes (Plural)
# update_xaxes applies to xaxis, xaxis2, xaxis3...
fig.update_xaxes(
    title_text="",        # Kill the "Degree Label" on BOTH sides
    showline=False,        # Black line at bottom of BOTH sides
    tickfont=dict(size=14, color='black'), # Make "Bachelor's" visible on BOTH sides
    matches='x'           # Force them to share the same ticks
)

# FIX C: Target ALL Y-Axes
fig.update_yaxes(
    title_text="Annual Earnings ($)", 
    title_font=dict(size=14, color="black", weight="bold"),
    showgrid=True,
    gridcolor="#888888",
    gridwidth=1,
    dtick=20000,
    tickformat="$,.0f",
    tickfont=dict(size=12, color="black", weight="bold"),    
    matches='y'           # Force them to share the same scale
)



# Standard Layout Cleanup
fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="black", family="Helvetica"), # Global Font Black
    yaxis2 = dict(title=""),


    legend=dict(
        orientation="h",
        yanchor="bottom", y=-0.25,
        xanchor="center", x=0.5,
        title="",
        font=dict(color="black")
    ),


    margin=dict(t=60, l=50, r=50, b=100)
)

# Clean up Top Labels (Facet Strips)
fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font=dict(size=20, color="black", weight="bold")))

# --- 6. CONDITIONAL DIVIDER ---
if len(selected_majors) > 1:
    fig.add_shape(
        type="line",
        x0=0.5, y0=0, x1=0.5, y1=1,
        xref="paper", yref="paper",
        line=dict(color="black", width=1, dash="dot")
    )

st.plotly_chart(fig, use_container_width=True)

# --- 7. SUMMARY TABLE (Fixed Index Error) ---
st.markdown("### Detailed Data")
display_table = agg_df.copy()
display_table = display_table.rename(columns=metric_map)
st.dataframe(
    display_table.style.format("${:,.0f}", subset=list(metric_map.values())), 
    use_container_width=True
)
