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


tab1, tab2 = st.tabs(["Earnings Analysis", "Industry Flows"])

with tab1:
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


#=================================
# HANDLE AND VISUALIZE FLOW DATA =
#=================================

@st.cache_data
def load_flow_data():
    """
    Loads the Industry Flow dataset (PSEOF).
    
    Purpose:
        This function handles the large dataset tracking where graduates move 
        (the 'Target' industry) after leaving their degree program (the 'Source').
    
    Methodology:
        - Maps raw CIP codes to 'Education' or 'CS/AI' families.
        - Maps numeric Degree codes (e.g., '05') to readable labels (e.g., "Bachelor's").
        - Used specifically for the 'Career Pathways' visualization.
    """
    
    df = pd.read_csv('Flow_Data/pseo_double_sankey_ready.csv', dtype={'cipcode': str, 'degree_level': str})
    
    # Map Major Names (Source Nodes)
    df['Major Name'] = df['cipcode'].apply(lambda x: 'Education' if str(x).startswith('13') else 'CompSci / AI')
    
    # Map Degree Labels
    degree_map = {
        '01': 'Certificate <1yr', '02': 'Certificate 1-2yrs',
        '03': 'Associates', '05': 'Bachelor\'s',
        '07': 'Master\'s', '17': 'Doctoral'
    }
    df['Degree Label'] = df['degree_level'].astype(str).map(degree_map).fillna(df['degree_level'])

    return df


### VISUALIZE FLOW DATA ###

with tab2:
    st.header("Industry & Geographic Pathways")
    
    # 1. LOAD DATA
    flow_df = load_flow_data()

    # 2. TAB CONTROLS
    col1, col2 = st.columns([1, 2])
    with col1:
        # Source State Filter (Where they graduated)
        state_list = sorted(flow_df['source_state_name'].dropna().unique().tolist())
        selected_state = st.selectbox("Select Origin State", ["All States"] + state_list)
    
    with col2:
        # THE TOGGLE: "Double Sankey"
        # If checked, we show the second leg of the journey (Industry -> Location)
        show_geo = st.checkbox("Show Geographic Destination (Double Sankey)", value=False)

    # 3. WARNING LOGIC (37.5% Representation)
    if selected_cohort != "All Cohorts" and time_code in ['y5', 'y10']:
        st.warning(
            f"⚠️ **Data Representation Warning:** You are viewing {timeframe_label} outcomes for the {selected_cohort} cohort. "
            "Due to privacy suppression in longitudinal data, approximately **37.5%** of graduates are represented here. "
            "For the most complete view, switch Cohort to 'All Cohorts'."
        )

    # 4. FILTERING
    # A. Global Filters (Degree & Major)
    mask_flow = (flow_df['Degree Label'].isin(selected_degrees)) & \
                (flow_df['Major Name'].isin(selected_majors))
    
    # B. State Filter
    if selected_state != "All States":
        mask_flow &= (flow_df['source_state_name'] == selected_state)
    
    # C. Cohort Filter
    if selected_cohort != "All Cohorts":
        mask_flow &= (flow_df['grad_cohort'] == selected_cohort)

    filtered_flow = flow_df[mask_flow].copy()
    
    # 5. SELECT FLOW COLUMN (y1_flow, y5_flow, y10_flow)
    flow_col = f'{time_code}_flow'

    # 6. SANKEY LOGIC
    if filtered_flow.empty or filtered_flow[flow_col].sum() == 0:
        st.info(f"No flow data available for **{selected_state}** with current filters.")
    else:
        # --- TOGGLE LOGIC ---
        if not show_geo:
            # Single Sankey: Show ONLY Major -> Industry
            # We filter for rows where the Source is one of our Majors
            sankey_data = filtered_flow[filtered_flow['source_node'].isin(['Education', 'CompSci / AI'])].copy()
        else:
            # Double Sankey: Show Everything (Major -> Industry -> Geo)
            # The CSV already contains both legs, so we use the whole filtered set
            sankey_data = filtered_flow.copy()

        # Aggregate flows (Group by Source -> Target)
        agg_flow = sankey_data.groupby(['source_node', 'target_node'])[flow_col].sum().reset_index()
        
        # Remove zero flows
        agg_flow = agg_flow[agg_flow[flow_col] > 0]
        agg_flow = agg_flow.sort_values(flow_col, ascending=False)

        # Build Nodes & Links
        all_nodes = list(agg_flow['source_node'].unique()) + list(agg_flow['target_node'].unique())
        all_nodes = list(set(all_nodes)) # Deduplicate
        node_map = {label: i for i, label in enumerate(all_nodes)}

        source_ids = agg_flow['source_node'].map(node_map)
        target_ids = agg_flow['target_node'].map(node_map)

        # Plotly Figure
        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15, thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color="#1D3557" # Professional Dark Blue
            ),
            link=dict(
                source=source_ids,
                target=target_ids,
                value=agg_flow[flow_col],
                color="rgba(69, 123, 157, 0.4)" # Semi-transparent Blue
            )
        )])

        fig_sankey.update_layout(
            title_text=f"Workforce Pathways: {selected_state}",
            height=700 if show_geo else 500, # Taller if showing geography
            font=dict(size=12, color="black"),
            margin=dict(t=40, l=10, r=10, b=10)
        )

        st.plotly_chart(fig_sankey, use_container_width=True)
        
        # 7. METRICS TABLE
        with st.expander("📊 View Data Table"):
            st.dataframe(agg_flow.rename(columns={flow_col: "Graduates"}), use_container_width=True)