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
        if str(code).startswith('11'): return 'CompSci'
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
    
    return df[df['Major Family'].isin(['Education', 'CompSci'])]


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

available_majors = ['Education', 'CompSci']
selected_majors = st.sidebar.multiselect(
    "Select Major(s)",
    options=available_majors,
    default=['Education', 'CompSci']
)


# --- 3. DATA PREPARATION ---

# Institution Filter in Sidebar
try:
    # Use a faster way to get unique institutions
    inst_names = pd.read_csv('Earnings_Data/pseoe_institution.csv', usecols=['institution'])
    inst_options = ["National Average"] + sorted(inst_names['institution'].unique().tolist())
except:
    inst_options = ["National Average"]

selected_inst = st.sidebar.selectbox("Select Institution", inst_options)

# LOGIC: Switch between State Aggregate and Specific Institution
if selected_inst == "National Average":
    if selected_cohort == 'All Cohorts':
        df = load_data("All Cohorts")
    else:
        raw_df = load_data("Trends")
        df = raw_df[raw_df['Cohort Group'] == selected_cohort].copy()
else:
    # Load the institution-specific file
    df_inst = pd.read_csv('Earnings_Data/pseoe_institution.csv', dtype={'cipcode': str, 'degree_level': str, 'grad_cohort': str})
    df = df_inst[df_inst['institution'] == selected_inst].copy()
    
    # Map Major Family
    df['Major Family'] = df['cipcode'].apply(lambda x: 'Education' if str(x).startswith('13') else 'CompSci')
    
    # Map Degrees
    degree_map = {
        '01': "Certificate <1yr", '1': "Certificate <1yr",
        '03': "Associates", '3': "Associates",
        '05': "Bachelor's", '5': "Bachelor's",
        '07': "Master's", '7': "Master's",
        '17': "Doctoral", '18': "Doctoral"
    }
    df['Degree Label'] = df['degree_level'].map(degree_map).fillna(df['degree_level'])

    # Cohort Filtering
    if selected_cohort == 'All Cohorts':
        # Handles both '0' and '0000' representations for "All Cohorts"
        df = df[df['grad_cohort'].isin(['0', '0000'])]
    else:
        cohort_map_rev = {
            '2001-2003': '2001', '2004-2006': '2004', '2007-2009': '2007',
            '2010-2012': '2010', '2013-2015': '2013', '2016-2018': '2016', '2019-2021': '2019'
        }
        target_year = cohort_map_rev.get(selected_cohort)
        df = df[df['grad_cohort'] == target_year]

# --- ENSURE DATA EXISTS BEFORE PROCEEDING ---
if not selected_degrees or not selected_majors:
    st.error("Please select at least one Degree Level and one Major.")
    st.stop()

# Final Masking
mask = (df['Degree Label'].isin(selected_degrees)) & (df['Major Family'].isin(selected_majors))
filtered_df = df[mask]

# If still empty, give user feedback on why
if filtered_df.empty:
    st.warning(f"No data found for {selected_inst} with these filters. Check if {selected_cohort} or the selected Degree Levels exist for this school.")
    st.stop()

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

    # Force Consistent Bar Text (100k format)
    fig.update_traces(
        texttemplate='%{y:$.3s}', # "$.2s" means Dollar + 2 sig digits + "SI" suffix (k/M)
        textposition='inside'    # Puts the number on top of the bar
    )

  
    fig.update_xaxes(
        title_text="",        # Kill the "Degree Label" on BOTH sides
        showline=False,        # Black line at bottom of BOTH sides
        tickfont=dict(size=14, color='black'), # Make "Bachelor's" visible on BOTH sides
        matches='x'           # Force them to share the same ticks
    )


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

    # --- 7. SUMMARY TABLE
    st.markdown("### Detailed Data")
    display_table = agg_df.copy()
    display_table = display_table.rename(columns=metric_map)
    st.dataframe(
        display_table.style.format("${:,.0f}", subset=list(metric_map.values())), 
        use_container_width=True
    )

    # ==========================================
    # --- 8. Earnigngs Growth Over Time ---
    # ==========================================
    st.divider()
    st.subheader("Earnings Visualized Over Time Post Graduation")
    st.markdown("Visualize the trajectory of earnings across the 1, 5, and 10-year milestones.")

    # A separate selector for this graph only
    growth_perc_label = st.radio(
        "Select Earnings Percentile",
        options=["25th Percentile (Entry)", "50th Percentile (Median)", "75th Percentile (Top Tier)"],
        index=1,
        horizontal=True,
        key="growth_percentile_selector" # Unique key to prevent conflict
    )

    # Map the selected label to the data column suffix
    perc_suffix_map = {
        "25th Percentile (Entry)": "p25",
        "50th Percentile (Median)": "p50",
        "75th Percentile (Top Tier)": "p75"
    }
    g_suffix = perc_suffix_map[growth_perc_label]

    # Identify the growth columns (y1, y5, y10)
    growth_cols = [f'y1_{g_suffix}_earnings', f'y5_{g_suffix}_earnings', f'y10_{g_suffix}_earnings']
    
    # Check if data is available for the selected institution/cohort
    if all(c in filtered_df.columns for c in growth_cols):
        # Aggregate data (useful if multiple CIP codes exist for the same major family)
        growth_plot_df = filtered_df.groupby(['Major Family', 'Degree Label'])[growth_cols].mean().reset_index()

        # Melt data for plotting (Transforming columns into a 'Time' category)
        growth_melted = growth_plot_df.melt(
            id_vars=['Major Family', 'Degree Label'],
            value_vars=growth_cols,
            var_name='Raw_Time',
            value_name='Earnings'
        )

        # Map raw column names to Y-axis labels
        time_display_map = {
            f'y1_{g_suffix}_earnings': '1 Year After',
            f'y5_{g_suffix}_earnings': '5 Years After',
            f'y10_{g_suffix}_earnings': '10 Years After'
        }

        growth_melted['Time Milestone'] = growth_melted['Raw_Time'].map(time_display_map)
        growth_melted['Earnings Label'] = growth_melted['Earnings'].apply(
            lambda x: f"${x/1000:.1f}k" if pd.notnull(x) else ""
        )

        # Create the Slope Graph
        fig_slope = px.line(
            growth_melted,
            x="Time Milestone",
            y="Earnings",
            color="Degree Label",
            text="Earnings Label",
            facet_col="Major Family" if len(selected_majors) > 1 else None,
            markers=True,
            category_orders={"Time Milestone": ["1 Year After", "5 Years After", "10 Years After"]},
            color_discrete_sequence=['#457B9D', '#1D3557', '#A8DADC', '#E63946'],
            height=600
        )

        # High-Contrast Styling
        fig_slope.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="black", family="Helvetica"),
            margin=dict(t=60, l=50, r=50, b=100),
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.25,
                xanchor="center", x=0.5,
                title="",
                font=dict(color="black")
            )
        )

        fig_slope.update_xaxes(
            title_text="Annual Earnings ($)",
            title_font=dict(size=14, color="black", weight="bold"),
            showgrid=True,
            gridcolor="#888888",
            tickformat="$,.0f",
            tickfont=dict(color="black")
        )

        fig_slope.update_yaxes(
            title_text="",
            tickfont=dict(color="black")
        )

        # Clean up labels at the top of the chart
        fig_slope.for_each_annotation(lambda a: a.update(
            text=a.text.split("=")[-1], 
            font=dict(size=16, color="black", weight="bold")
        ))

        # --- ADDING THE EDUCATION CEILING REFERENCE LINE ---
        
        # Find Max earnings for Education at Year 10
        edu_10yr_data = growth_melted[
            (growth_melted['Major Family'] == 'Education') & 
            (growth_melted['Time Milestone'] == '10 Years After')
        ]
        
        if not edu_10yr_data.empty:
            # We take the max in case there are multiple degree levels selected (e.g., Bachelors and Masters)
            edu_ceiling_val = edu_10yr_data['Earnings'].max()
            
            # Add the horizontal line to the figure
            fig_slope.add_hline(
                y=edu_ceiling_val, 
                line_dash="dot", 
                line_color="rgba(230, 57, 70, 0.5)", # Semi-opaque red (High contrast)
                line_width=2,
                annotation_text=f" Education Ceiling: ${edu_ceiling_val/1000:.1f}k", 
                annotation_position="top right",
                annotation_x=2,
                annotation_font=dict(size=10, color="red", family="Arial Black")

            )
        
        # --- END OF REFERENCE LINE LOGIC ---

        st.plotly_chart(fig_slope, use_container_width=True)
    else:
        st.info("The current dataset does not contain sufficient growth data for this specific selection.")
    

    # ==========================================
    # --- 9. EARNINGS CHANGE ANALYSIS ---
    # ==========================================
    st.divider()
    st.subheader("Growth Rate in Earnings Over Time Post Graduation")
    st.markdown("""
    This graph shows the growth rate in earnings between career milestones. 
    It tracks the step-up from Year 1 to Year 5, and the subsequent step-up from Year 5 to Year 10.
    """)

    # Local Percentile Selector
    growth_perc_choice = st.radio(
        "Select Percentile for Growth Calculation:",
        options=["25th Percentile (Entry)", "50th Percentile (Median)", "75th Percentile (Top Tier)"],
        index=1,
        horizontal=True,
        key="growth_delta_selector" 
    )

    g_suffix = {"25th Percentile (Entry)": "p25", "50th Percentile (Median)": "p50", "75th Percentile (Top Tier)": "p75"}[growth_perc_choice]

    change_cols = [f'y1_{g_suffix}_earnings', f'y5_{g_suffix}_earnings', f'y10_{g_suffix}_earnings']
    
    if all(c in filtered_df.columns for c in change_cols):
        change_base = filtered_df.groupby(['Major Family', 'Degree Label'])[change_cols].mean().reset_index()

        # Toggle for Normalization
        view_mode = st.radio(
            "View Mode:",
            options=["Absolute ($)", "Normalized (%)"],
            index=0,
            horizontal=True,
            key="growth_view_mode"
        )

        # Calculate Absolute Deltas
        change_base['Y1-Y5 Δ'] = change_base[f'y5_{g_suffix}_earnings'] - change_base[f'y1_{g_suffix}_earnings']
        change_base['Y5-Y10 Δ'] = change_base[f'y10_{g_suffix}_earnings'] - change_base[f'y5_{g_suffix}_earnings']

        # Calculate Normalized (%) Growth
        # Growth = ((New - Old) / Old) * 100
        change_base['Y1-Y5 %'] = (change_base['Y1-Y5 Δ'] / change_base[f'y1_{g_suffix}_earnings']) * 100
        change_base['Y5-Y10 %'] = (change_base['Y5-Y10 Δ'] / change_base[f'y5_{g_suffix}_earnings']) * 100

        # Choose which columns to melt based on toggle
        if view_mode == "Absolute ($)":
            val_vars = ['Y1-Y5 Δ', 'Y5-Y10 Δ']
            y_axis_label = "Total Earnings Increase ($)"
            # Label format: +$12.5k
            label_func = lambda x: f"+${x/1000:.1f}k" if pd.notnull(x) else ""
        else:
            val_vars = ['Y1-Y5 %', 'Y5-Y10 %']
            y_axis_label = "Percentage Growth (%)"
            # Label format: +15.2%
            label_func = lambda x: f"+{x:.1f}%" if pd.notnull(x) else ""

        change_melted = change_base.melt(
            id_vars=['Major Family', 'Degree Label'],
            value_vars=val_vars,
            var_name='Growth Period',
            value_name='Display Value'
        )

        change_melted['Label'] = change_melted['Display Value'].apply(label_func)
        
        # Build Figure
        fig_change = px.line(
            change_melted,
            x="Growth Period",
            y="Display Value",
            color="Degree Label",
            text="Label",
            facet_col="Major Family" if len(selected_majors) > 1 else None,
            facet_col_spacing=0.05, # Increased spacing to prevent X-axis clashing
            markers=True,
            color_discrete_sequence=['#457B9D', '#1D3557', '#A8DADC', '#E63946'],
            height=600
        )

        # Stagger labels and apply styling
        stagger_positions = ["top left", "top right", "bottom center"]
        for i, trace in enumerate(fig_change.data):
            # Prevents labels for different degrees from stacking on top of each other
            trace.textposition = stagger_positions[i % len(stagger_positions)]
            trace.textfont = dict(size=8, color="black", family="Arial Black", weight="normal")
            trace.mode = 'lines+markers+text'
            trace.marker = dict(size=10)
            trace.cliponaxis = False

        # Consolidated Layout Styling
        fig_change.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="black", family="Helvetica"),
            # Margin buffer prevents "top left/right" labels from going off-screen
            margin=dict(t=100, l=80, r=80, b=120), 
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.4,
                xanchor="center", x=0.5,
                title="",
                font=dict(color="black")
            ),
            uniformtext=dict(mode=False) # Prevents Plotly from hiding "clashing" labels
        )

        # Consolidated Y-Axis Styling (Uses 'Display Value' for range)
        tick_fmt = "$,.0f" if view_mode == "Absolute ($)" else ".1f%"
        
        fig_change.update_yaxes(
            title_text=y_axis_label,
            title_font=dict(size=14, color="black"),
            gridcolor="#EEEEEE",
            tickformat=tick_fmt,
            tickfont=dict(color="black"),
            # Dynamic range based on Display Value with 50% headroom
            range=[
                change_melted['Display Value'].min() * 0.5, 
                change_melted['Display Value'].max() * 1.5
            ]
        )

        fig_change.update_layout(
            yaxis2 = dict(title=""),
        )
        fig_change.update_xaxes(
            title_text="",
            tickfont=dict(color="black", size=11, weight="bold")
        )

        # Clean up facet labels at the top
        fig_change.for_each_annotation(lambda a: a.update(
            text=a.text.split("=")[-1], 
            font=dict(size=14, color="black")
        ))

        st.plotly_chart(fig_change, use_container_width=True)

    # ==========================================
    # --- 10. LONGITUDINAL HISTORICAL TRENDS ---
    # ==========================================
    st.divider()
    st.subheader("Historical Trends")
    st.markdown("Track the specific performance of degrees across cohorts and post-graduation milestones.")

    # CHART-SPECIFIC CONTROLS
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        h_perc_choice = st.radio(
            "Select Earnings Percentile:",
            options=["25th Percentile (Entry)", "50th Percentile (Median)", "75th Percentile (Top Tier)"],
            index=1, horizontal=True, key="h_trend_perc"
        )
        h_suffix = {"25th Percentile (Entry)": "p25", "50th Percentile (Median)": "p50", "75th Percentile (Top Tier)": "p75"}[h_perc_choice]

    with h_col2:
        h_years_selected = st.multiselect(
            "Select Post-Graduation Timeframe(s):",
            options=["1 Year Post-Grad", "5 Years Post-Grad", "10 Years Post-Grad"],
            default=["1 Year Post-Grad", "5 Years Post-Grad"], key="h_trend_years"
        )
    
    year_map = {"1 Year Post-Grad": "y1", "5 Years Post-Grad": "y5", "10 Years Post-Grad": "y10"}
    selected_prefixes = [year_map[y] for y in h_years_selected]

    if not h_years_selected:
        st.warning("Please select at least one timeframe to visualize.")
    else:
        # DATA LOADING & CLEANING
        if selected_inst == "National Average":
            h_raw = load_data("Trends")
            h_raw['Cohort Label'] = h_raw['Cohort Group']
        else:
            df_inst_all = pd.read_csv('Earnings_Data/pseoe_institution.csv', dtype={'cipcode': str, 'degree_level': str, 'grad_cohort': str})
            h_raw = df_inst_all[(df_inst_all['institution'] == selected_inst) & (df_inst_all['agg_level_pseo'] == 46)].copy()
            h_raw['Major Family'] = h_raw['cipcode'].apply(lambda x: 'Education' if str(x).startswith('13') else 'CompSci')
            deg_map_std = {'01':"Certificate <1yr",'03':"Associates",'05':"Bachelor's",'07':"Master's",'17':"Doctoral"}
            h_raw['Degree Label'] = h_raw['degree_level'].astype(str).str.zfill(2).map(deg_map_std)
            cohort_ui = {'2001':'01-03','2004':'04-06','2007':'07-09','2010':'10-12','2013':'13-15','2016':'16-18','2019':'19-21'}
            h_raw['Cohort Label'] = h_raw['grad_cohort'].map(cohort_ui)

        # TRANSFORMATION (One line per year per degree)
        target_cols = [f"{pre}_{h_suffix}_earnings" for pre in selected_prefixes]
        h_mask = (h_raw['Major Family'].isin(selected_majors)) & (h_raw['Degree Label'].isin(selected_degrees))
        h_filtered = h_raw[h_mask].copy()

        # Melt the timeframe columns into a single dimension
        h_melted = h_filtered.melt(
            id_vars=['Cohort Label', 'Degree Label', 'Major Family'],
            value_vars=target_cols,
            var_name='Time_Col',
            value_name='Earnings_Val'
        )
        
        # Add readable year label
        h_melted['Year_Label'] = h_melted['Time_Col'].apply(
            lambda x: 'Yr 10' if 'y10' in x else ('Yr 5' if 'y5' in x else 'Yr 1')
        )
        
        # CREATE THE UNIQUE LINE IDENTIFIER
        h_melted['Line_ID'] = h_melted['Degree Label'] + " (" + h_melted['Year_Label'] + ")"

        # CUSTOM COLOR MAPPING (Degree-based shades)
        # 1yr = Darkest, 5yr = Medium, 10yr = Lightest
        color_map_custom = {
            "Bachelor's (Yr 1)": "#003049", "Bachelor's (Yr 5)": "#335c67", "Bachelor's (Yr 10)": "#8ecae6",
            "Master's (Yr 1)": "#780000", "Master's (Yr 5)": "#c1121f", "Master's (Yr 10)": "#ffb3c1",
            "Doctoral (Yr 1)": "#132a13", "Doctoral (Yr 5)": "#31572c", "Doctoral (Yr 10)": "#90a955",
            "Certificate <1yr (Yr 1)": "#ffb703", "Certificate <1yr (Yr 5)": "#fb8500", "Certificate <1yr (Yr 10)": "#ffea00",
            "Associates (Yr 1)": "#3c096c", "Associates (Yr 5)": "#7b2cbf", "Associates (Yr 10)": "#c77dff"
        }

        h_final = h_melted.groupby(['Cohort Label', 'Line_ID', 'Major Family', 'Degree Label'])['Earnings_Val'].mean().reset_index()
        h_final = h_final.dropna(subset=['Earnings_Val'])

        # VISUALIZATION
        fig_hist = px.line(
            h_final,
            x="Cohort Label",
            y="Earnings_Val",
            color="Line_ID",
            text=h_final['Earnings_Val'].apply(lambda x: f"${x/1000:.1f}k"),
            facet_col="Major Family",
            category_orders={
                "Major Family": ["CompSci", "Education"],
                "Cohort Label": ["01-03", "04-06", "07-09", "10-12", "13-15", "16-18", "19-21"]
            },
            markers=True,
            color_discrete_map=color_map_custom,
            height=650
        )

        # Styling for contrast and legibility
        fig_hist.update_traces(
            textposition='top center',
            textfont=dict(size=9, color="black", family="Arial Black"),
            line=dict(width=3),
            cliponaxis=False
        )

        fig_hist.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=100, l=80, r=80, b=150),
            # LEGEND STYLE
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.5, xanchor="center", x=0.5,
                title="", font=dict(color="black", size=11, weight="bold"),
                bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1
            )
        )

        # AXIS STYLE (Force Black Text)
        fig_hist.update_yaxes(
            title="Annual Earnings ($)", title_font=dict(color="black", size=14, weight="bold"),
            gridcolor="#DDDDDD", tickformat="$,.0f", tickfont=dict(color="black", size=12, weight="bold"),
        )
        
        fig_hist.update_xaxes(
            title="", tickfont=dict(color="black", size=12, weight="bold"),
            showline=True, linecolor="black"
        )
        fig_hist.update_layout(
            yaxis2 = dict(title="")
        )

        # FACET HEADERS (Major Name)
        fig_hist.for_each_annotation(lambda a: a.update(
            text=a.text.split("=")[-1], 
            font=dict(size=18, color="black", weight="bold")
        ))

        st.plotly_chart(fig_hist, use_container_width=True)

    # ==========================================
    # --- 11. TOP 10 SCHOOLS BY MAJOR ---
    # ==========================================
    st.divider()
    st.header("Top 10 Performing Institutions")
    st.markdown("Comparing the highest earners across specific degrees and milestones. Reads only the cohort data from the sidebar.")

    # LOCAL CONTROLS (Shared by both figures)
    top_col1, top_col2, top_col3 = st.columns(3)
    
    with top_col1:
        top_perc = st.radio(
            "Select Earnings Percentile:",
            options=["25th Percentile (Entry)", "50th Percentile (Median)", "75th Percentile (Top Tier)"],
            index=1, horizontal=True, key="top_perc_selector"
        )
        t_suffix = {"25th Percentile (Entry)": "p25", "50th Percentile (Median)": "p50", "75th Percentile (Top Tier)": "p75"}[top_perc]

    with top_col2:
        top_years = st.radio(
            "Years After Graduation:",
            options=["1 Year", "5 Years", "10 Years"],
            index=0, horizontal=True, key="top_years_selector"
        )
        t_pre = {"1 Year": "y1", "5 Years": "y5", "10 Years": "y10"}[top_years]

    with top_col3:
        top_degree = st.selectbox(
            "Select Degree Level:",
            options=available_degrees,
            index=2, # Default to Bachelor's
            key="top_degree_selector"
        )

    # DATA PROCESSING
    df_top_raw = pd.read_csv('Earnings_Data/pseoe_institution.csv', dtype={'cipcode': str, 'degree_level': str, 'grad_cohort': str})
    
    # Map Major and Degree
    df_top_raw['Major Family'] = df_top_raw['cipcode'].apply(lambda x: 'Education' if str(x).startswith('13') else ('CompSci' if str(x).startswith('11') else 'Other'))
    deg_map_top = {'01':"Certificate <1yr", '02':"Certificate 1-2yrs", '03':"Associates", '05':"Bachelor's", '07':"Master's", '17':"Doctoral"}
    df_top_raw['Degree Label'] = df_top_raw['degree_level'].astype(str).str.zfill(2).map(deg_map_top)

    # Filter by Cohort (Sidebar)
    if selected_cohort == 'All Cohorts':
        df_top = df_top_raw[(df_top_raw['grad_cohort'].isin(['0', '0000'])) & (df_top_raw['agg_level_pseo'] == 40)].copy()
    else:
        cohort_map_top = {'2001-2003': '2001', '2004-2006': '2004', '2007-2009': '2007', '2010-2012': '2010', '2013-2015': '2013', '2016-2018': '2016', '2019-2021': '2019'}
        target_yr = cohort_map_top.get(selected_cohort)
        df_top = df_top_raw[(df_top_raw['grad_cohort'] == target_yr) & (df_top_raw['agg_level_pseo'] == 46)].copy()

    # Define the metric
    t_metric = f"{t_pre}_{t_suffix}_earnings"

    # HELPER FUNCTION FOR RENDERING CHARTS
    def render_top_10_chart(major_name, color_hex, key_suffix):
        # Filter for specific major and degree
        chart_data = df_top[(df_top['Major Family'] == major_name) & (df_top['Degree Label'] == top_degree)].copy()
        chart_data = chart_data.dropna(subset=[t_metric])
        
        # Group by institution to collapse multiple sub-program blocks into one
        # We take the mean (average) of earnings across sub-programs for that school
        chart_data = chart_data.groupby('institution')[t_metric].mean().reset_index()

        # Now sort and take exactly Top 10 UNIQUE universities
        chart_data = chart_data.sort_values(by=t_metric, ascending=False).head(10)

        if chart_data.empty:
            st.info(f"No data available for Top 10 in {major_name} with the selected filters.")
            return

        st.subheader(f"Top 10 Schools: {major_name}")
        
        # Use 'institution' and the metric for the bar chart
        fig = px.bar(
            chart_data,
            x="institution",
            y=t_metric,
            text=chart_data[t_metric].apply(lambda x: f"${x/1000:.1f}k"),
            height=500,
            color_discrete_sequence=[color_hex]
        )

        # Styling: Ensure Pure Black Ticks and Labels
        fig.update_traces(
            textposition='inside',
            textfont=dict(size=13, color="white", family="Arial Black"),
            marker_line_color='black',
            marker_line_width=1.5
        )

        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=40, l=60, r=60, b=150),
            xaxis=dict(
                categoryorder='total descending',
                tickangle=45,
                tickfont=dict(color="black", size=11, weight="bold"), # Pure Black
                linecolor="black",
                title=""
            ),
            yaxis=dict(
                title="Annual Earnings ($)",
                title_font=dict(color="black", size=14, weight="bold"), # Pure Black
                gridcolor="#EEEEEE",
                tickformat="$,.0f",
                tickfont=dict(color="black", size=12, weight="bold"), # Pure Black
                linecolor="black"
            )
        )
        
        st.plotly_chart(fig, use_container_width=True, key=f"top10_chart_{key_suffix}")

    # RENDER THE TWO FIGURES
    if "CompSci" in selected_majors:
        render_top_10_chart("CompSci", "#1D3557", "cs")
        st.write("") # Spacer

    if "Education" in selected_majors:
        render_top_10_chart("Education", "#E63946", "edu")



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
    df['Major Name'] = df['cipcode'].apply(lambda x: 'Education' if str(x).startswith('13') else 'CompSci')
    
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
    
    # LOAD DATA
    flow_df = load_flow_data()

    # CONTROLS CONTAINER (Clean Standard UI)
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

    # FILTERING LOGIC
    mask_flow = (flow_df['Degree Label'].isin(selected_degrees)) & \
                (flow_df['Major Name'].isin(selected_majors))
    
    if selected_state != "All States":
        mask_flow &= (flow_df['source_state_name'] == selected_state)
    
    mask_flow &= (flow_df['cohort_label'] == selected_cohort)
    
    filtered_flow = flow_df[mask_flow].copy()
    flow_col = f'{time_code}_flow'

    # VIEW LOGIC
    if filtered_flow.empty:
        st.info("No data available for the selected filters.")
    else:
        # SCENARIO A: BOTH
        if show_industry and show_geo:
            sankey_data = filtered_flow.copy() 
        
        # SCENARIO B: INDUSTRY ONLY
        elif show_industry and not show_geo:
            sankey_data = filtered_flow[filtered_flow['source_node'].isin(['Education', 'CompSci'])].copy()
        
        # SCENARIO C: GEOGRAPHY ONLY
        elif show_geo and not show_industry:
            sankey_data = filtered_flow[~filtered_flow['source_node'].isin(['Education', 'CompSci'])].copy()
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

        # AGGREGATION & COLORING
        if sankey_data.empty:
             st.warning("No flow data found for this selection.")
             st.stop()

        agg_flow = sankey_data.groupby(['source_node', 'target_node'])[flow_col].sum().reset_index()
        agg_flow = agg_flow[agg_flow[flow_col] > 0] 


        # Calculate totals for sizing
        node_totals = {}
        for _, row in agg_flow.iterrows():
            node_totals[row['source_node']] = node_totals.get(row['source_node'], 0) + row[flow_col]
            node_totals[row['target_node']] = node_totals.get(row['target_node'], 0) + row[flow_col]

        # Calculate Grand Total for Percentage (Denominator)
        major_nodes = ['Education', 'CompSci']
        major_flows = agg_flow[agg_flow['source_node'].isin(major_nodes)]
        total_cohort_count = major_flows[flow_col].sum()
        if total_cohort_count == 0:
            total_cohort_count = agg_flow[flow_col].sum()

        # Identify "Levels" (Columns in the diagram)
        sources = set(agg_flow['source_node'])
        targets = set(agg_flow['target_node'])
        
        # Level 0: Roots (Start Points -> Majors)
        level_0 = list(sources - targets) 
        
        # Level 1: Intermediates (Industries in double-flow, or empty in single)
        level_1 = list(sources & targets) 
        
        # Level 2: Leaves (End Points -> Geos, NME, or Industries in single flow)
        level_2 = list(targets - sources) 
        
        # Sort Each Level DESCENDING by Magnitude
        def get_magnitude(n): return node_totals.get(n, 0)
        
        level_0.sort(key=get_magnitude, reverse=True)
        level_1.sort(key=get_magnitude, reverse=True)
        level_2.sort(key=get_magnitude, reverse=True)
        
        # Combine in strict order: Left -> Middle -> Right
        # This ensures Index 0 is Top-Left, and forces the layout engine to respect the order
        all_nodes = level_0 + level_1 + level_2
        
        # Create Map
        node_map = {label: i for i, label in enumerate(all_nodes)}

        # COLOR PALETTE
        palette = [
            "#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", 
            "#1d3557", "#457b9d", "#a8dadc", "#e63946", "#6d597a"
        ]
        node_colors = [palette[i % len(palette)] for i in range(len(all_nodes))]
        
        # LINK COLORS (Grey with Opacity)
        link_color_static = "rgba(200, 200, 200, 0.3)"
        link_colors = [link_color_static] * len(agg_flow)

        # GENERATE CORRECT LABELS
        node_labels = []
        for n in all_nodes:
            # Determine Value (Outgoing for Sources, Incoming for Targets)
            if n in level_0:
                val = agg_flow[agg_flow['source_node'] == n][flow_col].sum()
            elif n in level_2:
                val = agg_flow[agg_flow['target_node'] == n][flow_col].sum()
            else:
                # Intermediates: Max of In/Out
                v_in = agg_flow[agg_flow['target_node'] == n][flow_col].sum()
                v_out = agg_flow[agg_flow['source_node'] == n][flow_col].sum()
                val = max(v_in, v_out)
            
            if unit_mode == "Percentage (%)":
                pct = (val / total_cohort_count) * 100
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

        # RENDER CHART
        st.plotly_chart(
            fig_sankey, 
            use_container_width=True,
            config={'displayModeBar': False}
        )
        
        # # DATA TABLE
        # with st.expander("📄 View Underlying Data"):
        #     display_df = agg_flow.rename(columns={flow_col: "Count"})
        #     display_df['Percentage'] = display_df.apply(lambda x: f"{(x['Count'] / node_totals.get(x['source_node'], 1)*100):.1f}%", axis=1)
        #     st.dataframe(display_df, use_container_width=True)