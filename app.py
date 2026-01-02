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
    st.header("🎓 Graduate Flow Analysis")
    
    # 1. LOAD DATA
    flow_df = load_flow_data()

    # 2. CONTROLS CONTAINER
    # We use a container to organize the filters and the new buttons clearly
    with st.container():
        col_filters, col_view = st.columns([1, 2])
        
        with col_filters:
            # Source State Filter
            state_list = sorted(flow_df['source_state_name'].dropna().unique().tolist())
            selected_state = st.selectbox("Select Origin State", ["All States"] + state_list)
            
            # Unit Toggle (Raw vs %)
            # User Request: "Toggle between raw number and percentage"
            unit_mode = st.radio("Display Units", ["Count (N)", "Percentage (%)"], horizontal=True)

        with col_view:
            st.markdown("**Destinations to Visualize:**")
            # User Request: "Two buttons... both clickable and unclickable"
            # We use columns to place them side-by-side like buttons
            b_col1, b_col2 = st.columns(2)
            with b_col1:
                show_industry = st.checkbox("Industry Sectors", value=True)
            with b_col2:
                show_geo = st.checkbox("Geography", value=False)

    st.divider()

    # 3. FILTERING LOGIC
    # A. Global Filters
    mask_flow = (flow_df['Degree Label'].isin(selected_degrees)) & \
                (flow_df['Major Name'].isin(selected_majors))
    
    # B. State Filter
    if selected_state != "All States":
        mask_flow &= (flow_df['source_state_name'] == selected_state)
    
    # C. Cohort Filter
    mask_flow &= (flow_df['cohort_label'] == selected_cohort)
    if selected_cohort != "All Cohorts" and time_code in ['y5', 'y10']:
        st.caption(f"⚠️ Note: Data representation for {selected_cohort} in {timeframe_label} is approx. 37.5% due to privacy suppression.")

    filtered_flow = flow_df[mask_flow].copy()
    
    # D. Select Timeframe Column
    flow_col = f'{time_code}_flow'

    # 4. VIEW LOGIC (The 4-Way Switch)
    if filtered_flow.empty or filtered_flow[flow_col].sum() == 0:
        st.info("No data available for the selected filters.")
    else:
        # We determine Source -> Target based on which buttons are checked
        
        # SCENARIO A: BOTH (Double Sankey)
        # Major -> Industry -> Geography
        if show_industry and show_geo:
            # We use the pre-calculated legs from the CSV
            sankey_data = filtered_flow.copy() # Use all rows (Leg 1 + Leg 2)
            # Leg 1: Major -> Industry (source_node=Major, target_node=Industry)
            # Leg 2: Industry -> Geo (source_node=Industry, target_node=Geo)
        
        # SCENARIO B: INDUSTRY ONLY
        # Major -> Industry
        elif show_industry and not show_geo:
            # Filter for Leg 1 only (where Source is a Major)
            sankey_data = filtered_flow[filtered_flow['source_node'].isin(['Education', 'CompSci / AI'])].copy()
        
        # SCENARIO C: GEOGRAPHY ONLY
        # Major -> Geography
        elif show_geo and not show_industry:
            # We need to construct this view because the CSV splits them.
            # We take Leg 2 (Industry->Geo) because it preserves the 'Major Name' column.
            # We ignore the 'Industry' node and map directly Major -> Geo.
            sankey_data = filtered_flow[~filtered_flow['source_node'].isin(['Education', 'CompSci / AI'])].copy()
            
            # OVERRIDE: Set source to Major, target is already Geo
            sankey_data['source_node'] = sankey_data['Major Name']
            # target_node is already the Geography (State or Division)
        
        # SCENARIO D: NEITHER (Employment Status)
        # Major -> Employed vs NO or very lerss earnings observed
        else:
            # 1. Separate the NME (No Earnings) rows
            # This assumes your updated cleanup script labeled the target as "No or very less earnings observed"
            nme_label = "No or very less earnings observed"
            nme_data = filtered_flow[filtered_flow['target_node'] == nme_label].copy()
            
            # 2. Aggregate all Industry flows into a single "Employed" category
            # We filter for 'Leg 1' flows (Major -> Industry) that are NOT the NME label
            emp_data = filtered_flow[
                (filtered_flow['source_node'].isin(['Education', 'CompSci / AI'])) & 
                (filtered_flow['target_node'] != nme_label)
            ].copy()
            
            if not emp_data.empty:
                # Group by source to collapse all industries into one flow
                emp_data = emp_data.groupby(['source_node', 'Major Name', 'cohort_label', 'Degree Label'])[flow_col].sum().reset_index()
                emp_data['target_node'] = "Employed (Earnings Observed)"
            
            # Combine them for the Sankey view
            sankey_data = pd.concat([nme_data, emp_data], ignore_index=True)

        # 5. AGGREGATION
        agg_flow = sankey_data.groupby(['source_node', 'target_node'])[flow_col].sum().reset_index()
        agg_flow = agg_flow[agg_flow[flow_col] > 0] # Remove zero flows

        # 6. PERCENTAGE CALCULATION
        # Calculate total flow per Source Node to compute %
        source_totals = agg_flow.groupby('source_node')[flow_col].sum().to_dict()
        
        def get_label(row):
            count = row[flow_col]
            total = source_totals.get(row['source_node'], count)
            pct = (count / total * 100) if total > 0 else 0
            if unit_mode == "Percentage (%)":
                return f"{pct:.1f}%"
            return f"{int(count):,}"

        # 7. SANKEY DIAGRAM CONSTRUCTION
        all_nodes = list(agg_flow['source_node'].unique()) + list(agg_flow['target_node'].unique())
        all_nodes = list(set(all_nodes)) # Deduplicate
        node_map = {label: i for i, label in enumerate(all_nodes)}

        source_ids = agg_flow['source_node'].map(node_map)
        target_ids = agg_flow['target_node'].map(node_map)
        
        # Calculate custom labels for links
        link_labels = agg_flow.apply(get_label, axis=1)
        # Calculate custom values (if % mode, the width should still be based on count to keep proportions, 
        # but the hover/label changes. OR we can make width % too. Let's keep width as count for stability.)

        fig_sankey = go.Figure(data=[go.Sankey(
            valueformat = ".1f%" if unit_mode == "Percentage (%)" else ",.0f",
            node=dict(
                pad=20, thickness=20,
                line=dict(color="black", width=0.5),
                label=all_nodes,
                color="black" # High contrast node text
            ),
            link=dict(
                source=source_ids,
                target=target_ids,
                value=agg_flow[flow_col],
                label=link_labels, # This puts the number/pct on the tooltip
                color="rgba(200, 200, 200, 0.5)" # Light gray for high contrast against white
            )
        )])

        # 8. HIGH CONTRAST STYLING (White Background)
        fig_sankey.update_layout(
            title_text=f"Flow Analysis: {selected_state}",
            height=600,
            font=dict(size=12, color="black", family="Arial"), # Strict black font
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=50, l=10, r=10, b=10)
        )
        
        # Force the numbers to appear ON the diagram nodes/links?
        # Plotly Sankey doesn't easily support static text on links, but we can enable node values.
        fig_sankey.update_traces(node_align="justify")

        st.plotly_chart(fig_sankey, use_container_width=True)
        
        # 9. SUMMARY TABLE
        with st.expander("📄 View Underlying Data"):
            display_df = agg_flow.rename(columns={flow_col: "Count"})
            display_df['Percentage'] = display_df.apply(lambda x: f"{(x['Count'] / source_totals[x['source_node']]*100):.1f}%", axis=1)
            st.dataframe(display_df, use_container_width=True)