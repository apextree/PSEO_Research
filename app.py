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

    # 2. CONTROLS CONTAINER (Clean Standard UI)
    with st.container():
        col_filters, col_view = st.columns([1, 1], gap="large")
        
        with col_filters:
            # Source State Filter
            state_list = sorted(flow_df['source_state_name'].dropna().unique().tolist())
            selected_state = st.selectbox("Select Origin State", ["All States"] + state_list)
            
            # Unit Toggle
            unit_mode = st.radio("Display Units", ["Count (N)", "Percentage (%)"], horizontal=True, label_visibility="collapsed")

        with col_view:
            st.markdown("##### Destinations to Visualize")
            # REPLACED "Pills" with standard Multiselect to remove the weird artifact
            view_selection = st.pills(
                "Select Flows:",
                options=["Industry Sectors", "Geography"],
                selection_mode="multi",
                default=["Industry Sectors"], 
                label_visibility="collapsed"
            )
            
            show_industry = "Industry Sectors" in view_selection
            show_geo = "Geography" in view_selection

    st.divider()

    # 3. FILTERING LOGIC
    mask_flow = (flow_df['Degree Label'].isin(selected_degrees)) & \
                (flow_df['Major Name'].isin(selected_majors))
    
    if selected_state != "All States":
        mask_flow &= (flow_df['source_state_name'] == selected_state)
    
    mask_flow &= (flow_df['cohort_label'] == selected_cohort)
    
    filtered_flow = flow_df[mask_flow].copy()
    flow_col = f'{time_code}_flow'

    # 4. VIEW LOGIC
    if filtered_flow.empty:
        st.info("No data available for the selected filters.")
    else:
        # SCENARIO A: BOTH
        if show_industry and show_geo:
            sankey_data = filtered_flow.copy() 
        
        # SCENARIO B: INDUSTRY ONLY
        elif show_industry and not show_geo:
            sankey_data = filtered_flow[filtered_flow['source_node'].isin(['Education', 'CompSci / AI'])].copy()
        
        # SCENARIO C: GEOGRAPHY ONLY
        elif show_geo and not show_industry:
            sankey_data = filtered_flow[~filtered_flow['source_node'].isin(['Education', 'CompSci / AI'])].copy()
            sankey_data['source_node'] = sankey_data['Major Name']
        
        # SCENARIO D: STATUS VIEW
        else:
            nme_label = "No or very less earnings observed"
            summary_rows = []
            cohort_groups = filtered_flow.groupby(['Major Name', 'cohort_label', 'degree_level', 'institution'])
            
            for (major, cohort, deg, inst), group in cohort_groups:
                emp_rows = group[group['source_node'] == major]
                emp_count = emp_rows[flow_col].sum()
                
                nme_col_name = f'{time_code}_grads_nme'
                if nme_col_name in group.columns:
                    nme_count = group[nme_col_name].max() 
                else:
                    nme_count = 0
                
                if emp_count > 0:
                    summary_rows.append({'source_node': major, 'target_node': 'Employed (Earnings Observed)', flow_col: emp_count})
                if nme_count > 0:
                    summary_rows.append({'source_node': major, 'target_node': nme_label, flow_col: nme_count})

            sankey_data = pd.DataFrame(summary_rows)

        # 5. AGGREGATION
        if sankey_data.empty:
             st.warning("No flow data found for this selection.")
             st.stop()

        agg_flow = sankey_data.groupby(['source_node', 'target_node'])[flow_col].sum().reset_index()
        agg_flow = agg_flow[agg_flow[flow_col] > 0] 

        # 6. NODE & COLOR LOGIC
        # Calculate totals for labels
        node_totals = {}
        for _, row in agg_flow.iterrows():
            node_totals[row['source_node']] = node_totals.get(row['source_node'], 0) + row[flow_col]
            node_totals[row['target_node']] = node_totals.get(row['target_node'], 0) + row[flow_col]

        # Calculate Grand Total for Percentage Calculation
        # We assume the unique 'source_nodes' at depth 0 represent the full population
        # A simple approximation is the sum of all values in the flow (divided by number of steps if needed)
        # For labeling, we will use the node's value relative to the ENTIRE displayed flow sum (to show relative share)
        total_flow_volume = agg_flow[flow_col].sum()

        all_nodes = list(agg_flow['source_node'].unique()) + list(agg_flow['target_node'].unique())
        all_nodes = list(set(all_nodes)) 
        node_map = {label: i for i, label in enumerate(all_nodes)}

        # COLOR PALETTE (Vibrant for Nodes)
        palette = [
            "#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", 
            "#1d3557", "#457b9d", "#a8dadc", "#e63946", "#6d597a"
        ]
        node_colors = [palette[i % len(palette)] for i in range(len(all_nodes))]
        
        # LINK COLORS (Grey with Opacity for cleanliness)
        link_color_static = "rgba(200, 200, 200, 0.3)"
        link_colors = [link_color_static] * len(agg_flow)

        # GENERATE LABELS (Fixing the Percentage Missing Issue)
        node_labels = []
        for n in all_nodes:
            val = node_totals.get(n, 0)
            if unit_mode == "Percentage (%)":
                # Calculate % relative to the total flow in this specific view
                # Note: For Sankey, Total In = Total Out. 
                # We normalize by the max flow of a single stage to keep it logical, 
                # or just simple % of total node volume if it's a distribution.
                # Here we just show the node's count formatted as % of the current view's total
                # This is a heuristic, but visually helpful.
                
                # Better approach: Just show the count, but if user wants %, 
                # we show % of the PRIMARY SOURCE (e.g. % of all graduates)
                
                # Find the 'Degree/Major' source nodes to get the true denominator
                # For now, we will simply format the number if it's 0-100, 
                # BUT since val is a raw count, we need to calculate the % manually.
                
                # Heuristic: Find the max node value (likely the source) and use that as denominator
                max_node_val = max(node_totals.values())
                pct = (val / max_node_val) * 100 if max_node_val > 0 else 0
                label_str = f"<b>{n}</b><br>{pct:.1f}%"
            else:
                label_str = f"<b>{n}</b><br>{int(val):,}"
            node_labels.append(label_str)

        # 7. SANKEY DIAGRAM
        fig_sankey = go.Figure(data=[go.Sankey(
            valueformat = ".1f%" if unit_mode == "Percentage (%)" else ",.0f",
            
            # Global Font Settings
            textfont = dict(color="black", size=10, family="Arial"),
            
            node=dict(
                pad=15, 
                thickness=15, 
                line=dict(color="white", width=0.5),
                label=node_labels, # Updated Labels
                color=node_colors, # COLORED NODES
            ),
            link=dict(
                source=agg_flow['source_node'].map(node_map),
                target=agg_flow['target_node'].map(node_map),
                value=agg_flow[flow_col],
                color=link_colors, # GREY LINKS
            )
        )])

        fig_sankey.update_layout(
            title=dict(
                text=f"<b>Graduate Flows: {selected_state}</b>",
                font=dict(color="black", size=16), # Force Solid Black & Larger
                x=0.5,
                xanchor='center'
            ),
            height=900,
            font=dict(size=10, color="black"),
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin=dict(t=80, l=10, r=10, b=50), # Increased top margin for title visibility
            hovermode="x",
        )
        
        # 8. CONTAINMENT BOX (CSS)
        st.markdown(
            """
            <style>
            .sankey-container {
                border: 1px solid #d0d0d0;
                border-radius: 8px; /* Standard rounded corners, not a pill */
                padding: 15px;
                background-color: #ffffff;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            </style>
            """, unsafe_allow_html=True
        )

        # RENDER CHART (Clean, no CSS wrapper to avoid the "bulge")
        st.plotly_chart(
            fig_sankey, 
            use_container_width=True,
            config={'displayModeBar': False} # Removes the floating toolbar on hover
        )
        
        # 8. DATA TABLE
        with st.expander("📄 View Underlying Data"):
            # ... (Keep your existing data table code here)
            display_df = agg_flow.rename(columns={flow_col: "Count"})
            # Safe percentage calculation
            display_df['Percentage'] = display_df.apply(lambda x: f"{(x['Count'] / node_totals.get(x['source_node'], 1)*100):.1f}%", axis=1)
            st.dataframe(display_df, use_container_width=True)