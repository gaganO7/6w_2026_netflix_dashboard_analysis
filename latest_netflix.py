import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

# Set visualization flag to use Streamlit's built-in charts only
VISUALIZATION_AVAILABLE = True

# Page configuration
st.set_page_config(
    page_title="Netflix Content Analysis Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("Netflix Content Data Analysis")
st.markdown("---")

# Executive Summary
st.header("Executive Summary")
st.markdown("""
This comprehensive data science project analyzes Netflix's content catalog, one of the world's leading streaming
entertainment platforms. The analysis aims to uncover valuable insights about content trends, catalog composition,
genre popularity, and geographic distribution across Movies and TV Shows on the platform.

**Key Objectives:**
- Understand the split between Movies and TV Shows on the platform
- Analyze how the catalog has grown over time by release year and date added
- Examine genre and country distribution to identify content strategy patterns
- Investigate content ratings and duration to understand audience targeting
- Explore director and country contribution patterns
- Identify catalog gaps and content strategy opportunities

**Expected Deliverables:**
- Interactive visualizations showcasing the Netflix content landscape
- Statistical analysis of key catalog composition metrics
- Actionable insights for content strategists and platform stakeholders
- A cleaned, analysis-ready dataset for further modeling
""")

st.markdown("---")

# Project Description
st.header("Project Description")

st.subheader("Problem Statement")
st.markdown("""
The streaming industry is highly competitive and dynamic, with platforms constantly seeking ways to improve their
content offerings, catalog strategy, and subscriber satisfaction. This project leverages Netflix's comprehensive
content dataset to provide data-driven insights that can help:

- **Content Strategists**: Optimize catalog composition, genre mix, and acquisition strategy
- **Platform Investors**: Identify content investment opportunities and market gaps
- **Market Researchers**: Understand catalog trends and geographic content patterns
- **Content Analysts**: Track how the catalog has evolved year over year
""")

st.subheader("Dataset Overview")
st.markdown("""
The Netflix dataset contains comprehensive information about titles on the platform including:

**Core Title Information:**
- Show ID, title, and content type (Movie or TV Show)
- Director and cast information
- Short plot description

**Catalog Metadata:**
- Country of production
- Date the title was added to Netflix
- Original release year

**Classification Data:**
- Content rating (e.g., PG-13, TV-MA)
- Duration (minutes for movies, seasons for TV shows)
- Primary genre classification

**Geographic Data:**
- Country-wise production and distribution information
- Regional content analysis capabilities
""")

st.subheader("Research Context: What Prior Studies Found on This Catalog")
st.markdown("""
This dataset is the well-known Netflix titles catalog (8,807 records) that has been studied in several published
data-analytics write-ups. Reviewing that prior work gives us a set of expectations to test against our own cleaned
data, rather than analyzing the catalog completely from scratch. Recurring findings across that research include:

- **Genre concentration**: Drama and Comedy consistently come out as the two largest genres in the catalog
- **Post-2015 growth**: The number of titles released per year rose sharply after 2015, coinciding with Netflix's
  well-documented shift toward original productions
- **Mature-audience skew**: A majority of titles carry mature or teen ratings (TV-MA, TV-14) rather than
  family-friendly ratings
- **Regional diversification**: While the United States dominates the country field, representation from Asian
  and European countries has grown, reflecting Netflix's international expansion strategy
- **Duration/type consistency checks**: Prior cleaning work flags and corrects any rows where a "Movie" carries a
  season-based duration or a "TV Show" carries a minutes-based duration

Throughout the **Data Visualization & Insights** section below, each relevant chart includes a short callout
comparing what this specific dataset actually shows against these prior findings, so you can see at a glance
whether the pattern holds here too.
""")

st.markdown("---")

# Data Overview Section
st.header("Data Overview")

# Load and display the dataset
@st.cache_data
def load_data():
    """Load the Netflix dataset with caching"""
    try:
        df = pd.read_csv('netflix.csv')
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_data
def prepare_dataframe_for_display(df, max_string_length=100):
    """Prepare dataframe/series for display by truncating long strings and handling object dtypes to avoid Arrow serialization issues"""
    if df is None:
        return df

    # Handle Series by converting to DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if df.empty:
        return df

    try:
        df_display = df.copy()
        # Fixing Datatype issues
        for col in df_display.columns:
            col_dtype = df_display[col].dtype

            if col_dtype == 'object' or col_dtype == 'O' or pd.api.types.is_object_dtype(df_display[col]):

                df_display[col] = df_display[col].astype(str)

                df_display[col] = df_display[col].replace(['nan', 'None', 'NaN', '<NA>', 'null'], '')

                mask = df_display[col].str.len() > max_string_length
                if mask.any():
                    df_display.loc[mask, col] = df_display.loc[mask, col].str[:max_string_length] + '...'

            elif pd.api.types.is_numeric_dtype(df_display[col]):
                if df_display[col].dtype == 'object':
                    df_display[col] = pd.to_numeric(df_display[col], errors='coerce')

            elif df_display[col].dtype.name == 'mixed':
                df_display[col] = df_display[col].astype(str)

        for col in df_display.select_dtypes(include=['object']).columns:
            df_display[col] = df_display[col].astype(str)

        for col in df_display.columns:
            if df_display[col].dtype == 'object' or str(df_display[col].dtype).startswith('object'):
                df_display[col] = df_display[col].astype(str)

        return df_display

    except Exception as e:
        # if there are issues just convert column to string to avoid full error state
        st.warning(f"DataFrame conversion warning: {str(e)}. Using fallback conversion.")
        try:
            df_fallback = df.copy()
            for col in df_fallback.columns:
                df_fallback[col] = df_fallback[col].astype(str)
            return df_fallback
        except:
            return pd.DataFrame(columns=[str(col) for col in df.columns]).astype(str)

def safe_dataframe_display(df, use_container_width=True, **kwargs):
    if df is not None and not df.empty:
        prepared_df = prepare_dataframe_for_display(df)
        return st.dataframe(prepared_df, use_container_width=use_container_width, **kwargs)
    else:
        return st.info("No data to display")

# Load the data
df = load_data()

if df is not None:
    # Basic dataset information
    st.subheader("Dataset Basic Information")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        movies_count = (df['type'] == 'Movie').sum() if 'type' in df.columns else 0
        st.metric("Movies", f"{movies_count:,}")

    # Create tabs for different data overview sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Column Info",
        "Missing Values",
        "Sample Data",
        "Statistics",
        "Categorical Data",
        "Data Quality"
    ])

    with tab1:
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column Name': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        col_info_display = prepare_dataframe_for_display(col_info)
        st.dataframe(col_info_display, use_container_width=True)

        # Data types summary
        st.subheader("Data Types Summary")
        dtype_summary = df.dtypes.value_counts()
        dtype_summary_df = pd.DataFrame({
            'Data Type': dtype_summary.index.astype(str),
            'Count': dtype_summary.values
        })
        dtype_summary_display = prepare_dataframe_for_display(dtype_summary_df)
        st.dataframe(dtype_summary_display, use_container_width=True)

        # Column details
        st.subheader("Detailed Column Descriptions")
        st.markdown("""
        **Column Descriptions:**
        - **show_id**: Unique identifier for the title
        - **type**: Content type — Movie or TV Show
        - **title**: Title of the movie or TV show
        - **director**: Director(s) of the title
        - **cast**: Main cast members
        - **country**: Country (or countries) of production
        - **date_added**: Date the title was added to Netflix
        - **release_year**: Year the title was originally released
        - **rating**: Content/age rating (e.g., PG-13, TV-MA)
        - **duration**: Duration — minutes for movies, seasons for TV shows
        - **primary_genre**: Primary genre/category the title is listed under
        - **description**: Short plot summary
        """)

    with tab2:
        st.subheader("Missing Values Analysis")
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]

        if not missing_data.empty:
            st.write("Columns with missing values:")
            missing_df = pd.DataFrame({
                'Column': missing_data.index,
                'Missing Count': missing_data.values,
                'Missing Percentage': (missing_data.values / len(df) * 100).round(2)
            })
            missing_df_display = prepare_dataframe_for_display(missing_df)
            st.dataframe(missing_df_display, use_container_width=True)

            # Bar chart for missing values
            st.subheader("Missing Values Visualization")
            st.bar_chart(missing_df.set_index('Column')['Missing Percentage'])
        else:
            st.success("No missing values found in the dataset!")

    with tab3:
        st.subheader("Sample Data")
        st.write("First 10 rows of the dataset:")
        df_display = prepare_dataframe_for_display(df.head(10))
        st.dataframe(df_display, use_container_width=True)

        st.subheader("Random Sample")
        st.write("10 random rows from the dataset:")
        df_sample = prepare_dataframe_for_display(df.sample(10))
        st.dataframe(df_sample, use_container_width=True)

        st.subheader("Last 5 Rows")
        st.write("Last 5 rows of the dataset:")
        df_tail = prepare_dataframe_for_display(df.tail(5))
        st.dataframe(df_tail, use_container_width=True)

    with tab4:
        st.subheader("Statistical Summary")

        # Statistical summary for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            st.write("**Numerical Columns Statistics:**")
            num_stats = df[numerical_cols].describe()
            num_stats_display = prepare_dataframe_for_display(num_stats)
            st.dataframe(num_stats_display, use_container_width=True)
        else:
            st.info("No numerical columns found in the dataset.")

        # Statistical summary for object columns
        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            st.write("**Text/Object Columns Statistics:**")
            object_stats = df[object_cols].describe()
            object_stats_display = prepare_dataframe_for_display(object_stats)
            st.dataframe(object_stats_display, use_container_width=True)

    with tab5:
        st.subheader("Categorical Data Analysis")
        categorical_cols = ['type', 'rating', 'primary_genre', 'country']

        for col in categorical_cols:
            if col in df.columns:
                unique_count = df[col].nunique()
                st.write(f"**{col}**: {unique_count} unique values")

                if unique_count <= 20:
                    st.write("All unique values:")
                    value_counts = df[col].value_counts()
                    value_counts_df = prepare_dataframe_for_display(value_counts.reset_index())
                    st.dataframe(value_counts_df, use_container_width=True)
                else:
                    st.write("Top 10 most frequent values:")
                    value_counts = df[col].value_counts().head(10)
                    value_counts_df = prepare_dataframe_for_display(value_counts.reset_index())
                    st.dataframe(value_counts_df, use_container_width=True)
                st.write("---")

    with tab6:
        st.subheader("Data Quality Assessment")

        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        st.metric("Duplicate Rows", duplicate_count)

        # Check for potential issues in specific columns
        issues = []

        if 'date_added' in df.columns:
            date_issues = df['date_added'].isna().sum()
            if date_issues > 0:
                issues.append(f"date_added column has {date_issues} missing values")

        if 'director' in df.columns:
            director_unknown = (df['director'] == 'Unknown').sum()
            if director_unknown > 0:
                issues.append(f"director column has {director_unknown} 'Unknown' entries")

        if 'country' in df.columns:
            country_unknown = (df['country'] == 'Unknown').sum()
            if country_unknown > 0:
                issues.append(f"country column has {country_unknown} 'Unknown' entries")

        st.subheader("Data Quality Issues")
        if issues:
            st.write("**Potential data quality issues identified:**")
            for issue in issues:
                st.write(f"- {issue}")
        else:
            st.success("No major data quality issues detected!")

        # Additional quality checks
        st.subheader("Additional Quality Metrics")
        col1, col2, col3 = st.columns(3)

        with col1:
            total_nulls = df.isnull().sum().sum()
            st.metric("Total Null Values", total_nulls)

        with col2:
            completeness = ((df.size - total_nulls) / df.size * 100)
            st.metric("Data Completeness", f"{completeness:.2f}%")

        with col3:
            unique_rows = df.shape[0] - duplicate_count
            st.metric("Unique Rows", unique_rows)

else:
    st.error("Could not load the dataset. Please ensure 'netflix.csv' is in the correct directory.")

st.markdown("---")

# Data Cleaning Section
st.header("Data Cleaning & Preprocessing")

if df is not None:
    st.markdown("""
    This section focuses on extracting useful date/duration features, standardizing column names, and handling
    missing values. Data cleaning is crucial for ensuring the quality and reliability of our analysis.
    **Note**: as requested, the `date_added` column is dropped after we extract the `added_year` and `month_added`
    features from it, since the raw date string itself is not useful for aggregate visualizations.
    """)

    # Function to extract date-based features before dropping date_added
    @st.cache_data
    def extract_date_features(df):
        """Extract added_year and month_added from date_added, then drop date_added"""
        df_dated = df.copy()

        if 'date_added' in df_dated.columns:
            parsed_dates = pd.to_datetime(df_dated['date_added'], errors='coerce')

            df_dated['added_year'] = parsed_dates.dt.year
            df_dated['added_month_num'] = parsed_dates.dt.month
            df_dated['month_added'] = parsed_dates.dt.strftime('%b')

            # Fill rows where date parsing failed (missing date_added) with 'Unknown'/median year
            if df_dated['added_year'].isnull().sum() > 0:
                median_year = df_dated['added_year'].median()
                df_dated['added_year'] = df_dated['added_year'].fillna(median_year)
            df_dated['added_year'] = df_dated['added_year'].astype(int)
            df_dated['added_month_num'] = df_dated['added_month_num'].fillna(0).astype(int)
            df_dated['month_added'] = df_dated['month_added'].fillna('Unknown')

            # Drop the original date_added column as requested
            df_dated = df_dated.drop(columns=['date_added'])

        return df_dated

    # Function to extract numeric duration and standardize genre column name
    @st.cache_data
    def clean_data(df):
        """Clean the dataset: extract date/duration features, standardize columns, handle missing values"""
        df_cleaned = df.copy()

        # Step 1: Extract date features and drop date_added
        df_cleaned = extract_date_features(df_cleaned)

        # Step 2: Extract numeric duration and a duration unit flag
        if 'duration' in df_cleaned.columns:
            df_cleaned['duration'] = df_cleaned['duration'].astype(str)
            df_cleaned['duration_num'] = df_cleaned['duration'].str.extract(r'(\d+)').astype(float)
            df_cleaned['duration_unit'] = np.where(
                df_cleaned['duration'].str.contains('Season', case=False, na=False),
                'Seasons', 'Minutes'
            )
            # Fill any missing duration_num with the median for that content type
            if 'type' in df_cleaned.columns and df_cleaned['duration_num'].isnull().sum() > 0:
                df_cleaned['duration_num'] = df_cleaned.groupby('type')['duration_num'].transform(
                    lambda x: x.fillna(x.median())
                )
            df_cleaned['duration_num'] = df_cleaned['duration_num'].fillna(df_cleaned['duration_num'].median())

        # Step 3: Standardize the genre column name to 'listed_in' for consistency with the visualization spec
        if 'primary_genre' in df_cleaned.columns:
            df_cleaned = df_cleaned.rename(columns={'primary_genre': 'listed_in'})

        # Step 4: Clean categorical columns
        categorical_cols = ['director', 'cast', 'country', 'rating', 'listed_in']
        for col in categorical_cols:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].fillna('Unknown')
                df_cleaned[col] = df_cleaned[col].replace('', 'Unknown')

        # Step 5: Handle any remaining missing values in any other columns
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().sum() > 0:
                if df_cleaned[col].dtype == 'object':
                    df_cleaned[col] = df_cleaned[col].fillna('Unknown')
                else:
                    median_val = df_cleaned[col].median()
                    if pd.isna(median_val):
                        df_cleaned[col] = df_cleaned[col].fillna(0)
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna(median_val)

        return df_cleaned

    # Apply data cleaning (this happens outside tabs so it's applied globally)
    df_cleaned = clean_data(df)

    # Tabs for data cleaning tasks
    clean_tab1, clean_tab2, clean_tab3 = st.tabs([
        "Date & Duration Processing",
        "Missing Values Treatment",
        "Duration Outlier Treatment"
    ])

    with clean_tab1:
        st.subheader("Date Feature Extraction & Column Drop")

        st.markdown("""
        **Problem Identified**: The raw `date_added` column is a date string (e.g., `2021-09-25`) that isn't
        directly useful for aggregate trend charts. We extract the year and month it was added to Netflix into
        two new columns, then drop the original `date_added` column as requested.
        """)

        st.write("**Transformations Applied:**")
        st.markdown("""
        - **added_year**: Extracted from `date_added` — used for the "Content added by year" trend
        - **month_added**: Extracted from `date_added` (month abbreviation) — used for the monthly addition trend
        - **date_added**: Dropped from the cleaned dataset after feature extraction
        - **duration_num**: Numeric portion of `duration` extracted (minutes for Movies, seasons for TV Shows)
        - **listed_in**: `primary_genre` renamed to `listed_in` for consistency with genre-based visualizations
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Before Cleaning — Columns:**")
            st.write(list(df.columns))
        with col2:
            st.write("**After Cleaning — Columns:**")
            st.write(list(df_cleaned.columns))

        if 'date_added' in df.columns:
            st.success(f"`date_added` successfully dropped. `added_year` and `month_added` retained in its place.")
            st.write("**Sample of extracted date features:**")
            sample_cols = [c for c in ['title', 'added_year', 'month_added'] if c in df_cleaned.columns]
            st.dataframe(prepare_dataframe_for_display(df_cleaned[sample_cols].head(10)), use_container_width=True)

    with clean_tab2:
        st.subheader("Missing Values Treatment")

        st.write("**Data Cleaning Applied:**")
        st.markdown("""
        The following cleaning operations have been applied:
        - **Date Columns**: `date_added` parsed into `added_year`/`month_added`, missing dates filled with the
          median year and 'Unknown' month, then `date_added` dropped
        - **Duration Column**: Numeric value extracted into `duration_num`, missing values filled with the median
          by content type (Movie/TV Show)
        - **Genre Column**: `primary_genre` renamed to `listed_in`, missing values set to 'Unknown'
        - **Categorical Columns**: Missing values set to 'Unknown' (director, cast, country, rating, listed_in)
        - **All Remaining Columns**: Any remaining missing values handled appropriately (numeric with median,
          text with 'Unknown')
        """)

        # Missing values analysis after cleaning
        st.write("**Missing Values After Cleaning:**")
        missing_summary = df_cleaned.isnull().sum().sort_values(ascending=False)
        missing_summary = missing_summary[missing_summary > 0]

        if not missing_summary.empty:
            missing_df = pd.DataFrame({
                'Column': missing_summary.index,
                'Missing Count': missing_summary.values,
                'Missing Percentage': (missing_summary.values / len(df_cleaned) * 100).round(2)
            })
            missing_df_display = prepare_dataframe_for_display(missing_df)
            st.dataframe(missing_df_display, use_container_width=True)
        else:
            st.success("No missing values remain after comprehensive cleaning!")

        # Show before/after comparison
        st.subheader("Before vs After Cleaning Comparison")
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Original Dataset:**")
            original_nulls = df.isnull().sum().sum()
            st.metric("Total Missing Values", f"{original_nulls:,}")

        with col2:
            st.write("**Cleaned Dataset:**")
            cleaned_nulls = df_cleaned.isnull().sum().sum()
            st.metric("Total Missing Values", f"{cleaned_nulls:,}")
            reduction_pct = ((original_nulls - cleaned_nulls) / original_nulls * 100) if original_nulls > 0 else 0
            st.metric("Reduction", f"{reduction_pct:.1f}%")

        # Show cleaning results
        st.subheader("Data Cleaning Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Original Records", f"{df.shape[0]:,}")
        with col2:
            st.metric("Cleaned Records", f"{df_cleaned.shape[0]:,}")
        with col3:
            rows_removed = df.shape[0] - df_cleaned.shape[0]
            st.metric("Rows Removed", rows_removed)

    with clean_tab3:
        st.subheader("Duration Outlier Detection & Treatment")

        st.markdown("""
        We check for outliers in `duration_num` for **Movies** (minutes), since this is the clearest continuous
        numeric field in the dataset. TV Show duration is measured in seasons and is not comparable on the same
        scale, so it is excluded from this outlier check.
        """)

        if 'duration_num' in df_cleaned.columns and 'type' in df_cleaned.columns:
            movie_durations = df_cleaned.loc[df_cleaned['type'] == 'Movie', 'duration_num'].dropna()

            if len(movie_durations) > 0:
                Q1 = movie_durations.quantile(0.25)
                Q3 = movie_durations.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers_mask = (movie_durations < lower_bound) | (movie_durations > upper_bound)
                outliers_count = int(outliers_mask.sum())
                outliers_percentage = (outliers_count / len(movie_durations)) * 100

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Outliers", outliers_count)
                with col2:
                    st.metric("Outlier %", f"{outliers_percentage:.2f}%")
                with col3:
                    st.metric("IQR", f"{IQR:.2f}")
                with col4:
                    st.metric("Median", f"{movie_durations.median():.2f}")

                st.write("**Movie Duration Distribution (minutes):**")
                bins = pd.cut(movie_durations, bins=20)
                bin_counts = bins.value_counts().sort_index()
                bin_labels = [f"{interval.left:.0f}-{interval.right:.0f}" for interval in bin_counts.index]
                chart_data = pd.DataFrame({
                    'Bin Range': bin_labels,
                    'Count': bin_counts.values
                }).set_index('Bin Range')
                st.bar_chart(chart_data)

                st.info(
                    f"Movie durations typically range from **{lower_bound:.0f}** to **{upper_bound:.0f}** minutes "
                    f"under the IQR method. Values outside this range are flagged as outliers but are **kept** in "
                    f"the dataset since very short or very long films are legitimate content, not data errors."
                )
            else:
                st.info("No movie duration data available for outlier detection")
        else:
            st.info("No numerical duration columns available for outlier detection")

    # Type / Duration Consistency Check (matches the validation step used in prior research on this dataset)
    st.subheader("Type & Duration Consistency Check")
    st.markdown("""
    Prior analyses of this dataset flag rows where the `duration` unit doesn't match the `type` label — e.g., a
    "Movie" recorded in seasons, or a "TV Show" recorded in minutes. We run the same check here.
    """)
    if 'duration_unit' in df_cleaned.columns and 'type' in df_cleaned.columns:
        mismatch_mask = (
            ((df_cleaned['type'] == 'Movie') & (df_cleaned['duration_unit'] == 'Seasons')) |
            ((df_cleaned['type'] == 'TV Show') & (df_cleaned['duration_unit'] == 'Minutes'))
        )
        mismatch_count = int(mismatch_mask.sum())
        if mismatch_count > 0:
            st.warning(f"Found {mismatch_count} row(s) where `type` and `duration` unit don't match. These are kept but worth reviewing before modeling.")
            st.dataframe(
                prepare_dataframe_for_display(df_cleaned.loc[mismatch_mask, ['title', 'type', 'duration']].head(10)),
                use_container_width=True
            )
        else:
            st.success("No type/duration mismatches found — every 'Movie' is measured in minutes and every 'TV Show' in seasons, consistent with prior cleaning work on this dataset.")
    else:
        st.info("Duration unit column not available for this check.")

    st.markdown("---")

    # Final dataset summary
    st.subheader("Final Processed Dataset Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Final Records", f"{df_cleaned.shape[0]:,}")
    with col2:
        st.metric("Final Columns", df_cleaned.shape[1])
    with col3:
        remaining_nulls = df_cleaned.isnull().sum().sum()
        st.metric("Remaining Nulls", remaining_nulls)

    # Show column summary
    st.subheader("Column Summary")
    col_summary = pd.DataFrame({
        'Column': df_cleaned.columns,
        'Data Type': df_cleaned.dtypes.astype(str),
        'Non-Null Count': df_cleaned.count(),
        'Null Count': df_cleaned.isnull().sum(),
        'Unique Values': df_cleaned.nunique()
    })
    col_summary_display = prepare_dataframe_for_display(col_summary)
    st.dataframe(col_summary_display, use_container_width=True)

else:
    st.error("Could not load the dataset for cleaning operations.")

st.markdown("---")

# Data Visualization Section
st.header("Data Visualization & Insights")

if df is not None:
    # Use cleaned data for visualizations
    df_viz = df_cleaned if 'df_cleaned' in locals() else df

    # Clear any previous cache to avoid rendering issues
    st.cache_data.clear()

    # Helper: explode a comma-separated column into individual values
    def explode_comma_column(series):
        """Split a comma-separated string column into a flat list of trimmed individual values"""
        all_values = []
        for entry in series.dropna():
            if isinstance(entry, str) and entry != 'Unknown' and entry != '':
                all_values.extend([v.strip() for v in entry.split(',') if v.strip()])
        return all_values

    # Sidebar Filters with Form to prevent auto-rerun
    st.sidebar.header("Data Filters")
    st.sidebar.markdown("Use these filters to explore specific segments of the catalog:")
    st.sidebar.markdown("**Tip:** Set your desired filters and click 'Apply Filters' to update visualizations.")
    st.sidebar.markdown("---")

    # Initialize session state for filters if not exists
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = False

    if 'applied_types' not in st.session_state:
        st.session_state.applied_types = []
    if 'applied_genres' not in st.session_state:
        st.session_state.applied_genres = []
    if 'applied_countries' not in st.session_state:
        st.session_state.applied_countries = []
    if 'applied_ratings' not in st.session_state:
        st.session_state.applied_ratings = []
    if 'applied_year_range' not in st.session_state:
        st.session_state.applied_year_range = None

    # Initialize filtered dataframe
    df_filtered = df_viz.copy()

    # Use form to prevent auto-rerun on filter changes
    with st.sidebar.form("filters_form"):
        # Content Type Filter
        st.subheader("Content Filters")

        if 'type' in df_viz.columns:
            content_types = sorted(df_viz['type'].dropna().unique().tolist())
            selected_types = st.multiselect(
                "Select Content Type",
                content_types,
                help="Filter by Movie or TV Show"
            )
        else:
            selected_types = []

        # Genre Filter (Multi-select)
        if 'listed_in' in df_viz.columns:
            genres = sorted(df_viz['listed_in'].dropna().unique().tolist())
            selected_genres = st.multiselect(
                "Select Genres",
                genres,
                help="Filter by primary genre"
            )
        else:
            selected_genres = []

        st.markdown("---")

        # Geographic Filters Section
        st.subheader("Geographic Filters")

        if 'country' in df_viz.columns:
            all_countries = sorted(set(explode_comma_column(df_viz['country'])))
            selected_countries = st.multiselect(
                "Select Countries",
                all_countries,
                help="Filter by country of production (multiple selection allowed)"
            )
        else:
            selected_countries = []

        st.markdown("---")

        # Rating & Release Filters Section
        st.subheader("Rating & Release Filters")

        if 'rating' in df_viz.columns:
            ratings = sorted(df_viz['rating'].dropna().unique().tolist())
            selected_ratings = st.multiselect(
                "Select Content Ratings",
                ratings,
                help="Filter by content rating (e.g., PG-13, TV-MA)"
            )
        else:
            selected_ratings = []

        if 'release_year' in df_viz.columns:
            year_data = df_viz['release_year'].dropna()
            if len(year_data) > 0:
                min_year = int(year_data.min())
                max_year = int(year_data.max())
                year_range = st.slider(
                    "Release Year Range",
                    min_value=min_year,
                    max_value=max_year,
                    value=(min_year, max_year),
                    step=1,
                    help="Filter titles by original release year"
                )
            else:
                year_range = (1925, 2021)
        else:
            year_range = (1925, 2021)

        # Filter Control Buttons
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            apply_filters = st.form_submit_button("Apply Filters", type="primary", use_container_width=True)

        with col2:
            reset_filters = st.form_submit_button("Reset Filters", use_container_width=True)

    # Handle reset filters
    if reset_filters:
        for key in list(st.session_state.keys()):
            if key.startswith('applied_'):
                del st.session_state[key]
        st.session_state.filters_applied = False
        st.rerun()

    # Apply filters only when form is submitted
    if apply_filters:
        st.session_state.filters_applied = True
        st.session_state.applied_types = selected_types
        st.session_state.applied_genres = selected_genres
        st.session_state.applied_countries = selected_countries
        st.session_state.applied_ratings = selected_ratings
        st.session_state.applied_year_range = year_range

    # Only filter data if filters have been applied
    if st.session_state.filters_applied:
        if st.session_state.applied_types:
            df_filtered = df_filtered[df_filtered['type'].isin(st.session_state.applied_types)]

        if st.session_state.applied_genres:
            df_filtered = df_filtered[df_filtered['listed_in'].isin(st.session_state.applied_genres)]

        if st.session_state.applied_countries:
            mask = df_filtered['country'].apply(
                lambda x: any(c in str(x) for c in st.session_state.applied_countries) if pd.notna(x) else False
            )
            df_filtered = df_filtered[mask]

        if st.session_state.applied_ratings:
            df_filtered = df_filtered[df_filtered['rating'].isin(st.session_state.applied_ratings)]

        if st.session_state.applied_year_range and 'release_year' in df_filtered.columns:
            df_filtered = df_filtered[
            (df_filtered['release_year'] >= st.session_state.applied_year_range[0]) &
            (df_filtered['release_year'] <= st.session_state.applied_year_range[1])
            ]

    # Display filter summary in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Current Status")

    if st.session_state.filters_applied:
        st.sidebar.metric("Titles Showing", f"{len(df_filtered):,}")
        st.sidebar.metric("Total Available", f"{len(df_viz):,}")
        filter_percentage = (len(df_filtered) / len(df_viz) * 100) if len(df_viz) > 0 else 0
        st.sidebar.metric("Percentage Shown", f"{filter_percentage:.1f}%")

        active_filter_count = sum([
            bool(st.session_state.applied_types),
            bool(st.session_state.applied_genres),
            bool(st.session_state.applied_countries),
            bool(st.session_state.applied_ratings),
        ])
        if active_filter_count > 0:
            st.sidebar.success(f"{active_filter_count} filter(s) active")
        else:
            st.sidebar.info("No filters applied")
    else:
        st.sidebar.metric("Total Available", f"{len(df_viz):,}")
        st.sidebar.warning("Filters not applied yet")
        st.sidebar.info("Click 'Apply Filters' to see results")

    # Update df_viz to use filtered data only if filters are applied
    if st.session_state.filters_applied:
        df_viz = df_filtered

    if st.session_state.filters_applied and len(df_filtered) < len(df_cleaned if 'df_cleaned' in locals() else df):
        st.info(f"Showing {len(df_filtered):,} titles based on your filter selection out of {len(df_cleaned if 'df_cleaned' in locals() else df):,} total titles.")

    st.markdown("""
    This section presents comprehensive visualizations to uncover insights from the Netflix content dataset.
    Each plot reveals different aspects of the catalog composition, genre and geographic mix, and release trends.

    **Tip**: Use the filters in the sidebar to explore specific segments of the catalog. Set your desired filters
    and click 'Apply Filters' to update visualizations!
    """)

    st.markdown("---")

    # Dynamic Metrics Dashboard
    st.subheader("Filtered Data Overview")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Titles", f"{len(df_viz):,}")
    with col2:
        if 'type' in df_viz.columns and len(df_viz) > 0:
            movie_pct = (df_viz['type'] == 'Movie').mean() * 100
            st.metric("Movies %", f"{movie_pct:.0f}%")
        else:
            st.metric("Movies %", "N/A")
    with col3:
        if 'listed_in' in df_viz.columns:
            st.metric("Genres", df_viz['listed_in'].nunique())
        else:
            st.metric("Genres", "N/A")
    with col4:
        if 'country' in df_viz.columns:
            unique_countries = len(set(explode_comma_column(df_viz['country'])))
            st.metric("Countries", unique_countries)
        else:
            st.metric("Countries", "N/A")
    with col5:
        if 'release_year' in df_viz.columns and len(df_viz) > 0:
            st.metric("Median Release Year", int(df_viz['release_year'].median()))
        else:
            st.metric("Median Release Year", "N/A")

    st.markdown("---")

    # Check if filtered data is empty
    if len(df_viz) == 0:
        st.error("No titles match your current filter criteria. Please adjust your filters and try again.")
        st.info("**Tip**: Try expanding your filter criteria or reset all filters to see the full dataset.")
        st.stop()

    st.markdown("Explore the dashboard using the tabs below — each tab focuses on one analytical theme so all 15 charts are easier to navigate.")
    st.markdown("")
    tab_overview, tab_trends, tab_genre, tab_geo, tab_rd, tab_directors = st.tabs([
        "Overview",
        "Trends",
        "Genre",
        "Geography",
        "Ratings & Duration",
        "Directors",
    ])

    with tab_overview:
        # =========================================================================
        # Section A: Content Type Distribution
        # =========================================================================
        st.subheader("Content Type Distribution")

        # Chart 1: Movies vs TV Shows (Pie Chart)
        st.write("### Movies vs TV Shows Distribution")
        if 'type' in df_viz.columns:
            type_counts = df_viz['type'].value_counts().reset_index()
            type_counts.columns = ['type', 'count']

            fig1 = px.pie(
                type_counts,
                values='count',
                names='type',
                color='type',
                title="Distribution of Movies vs TV Shows",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            fig1.update_layout(height=500)
            st.plotly_chart(fig1, use_container_width=True)

            st.write("**Key Insights:**")
            top_type = type_counts.iloc[0]
            second_type = type_counts.iloc[1] if len(type_counts) > 1 else None
            total_titles_c1 = type_counts['count'].sum()
            top_pct = top_type['count'] / total_titles_c1 * 100
            st.write(f"- **{top_type['type']}** makes up the larger share of the catalog with {top_type['count']:,} titles ({top_pct:.1f}%)")
            if second_type is not None:
                second_pct = second_type['count'] / total_titles_c1 * 100
                st.write(f"- **{second_type['type']}** accounts for {second_type['count']:,} titles ({second_pct:.1f}%) of the catalog")
            ratio = top_type['count'] / second_type['count'] if second_type is not None and second_type['count'] > 0 else 0
            st.write(f"- The catalog carries roughly **{ratio:.1f}x** more {top_type['type']} titles than {second_type['type'] if second_type is not None else 'the other type'}")
            st.write("- The catalog mix between Movies and TV Shows shapes overall content strategy and acquisition budget allocation")
            st.write(f"- Out of {total_titles_c1:,} titles currently in view, the format split is a key lens for understanding platform positioning")
        else:
            st.error("Column 'type' not found in dataset")

        st.markdown("---")


    with tab_trends:
        # =========================================================================
        # Section B: Release & Addition Trends
        # =========================================================================
        st.subheader("Release & Addition Trends")

        # Chart 2: Number of releases per year (Bar Chart)
        st.write("### Number of Releases per Year")
        if 'release_year' in df_viz.columns:
            year_counts = df_viz['release_year'].value_counts().sort_index().reset_index()
            year_counts.columns = ['release_year', 'count']
            year_counts = year_counts[year_counts['release_year'] >= 1990]

            fig2 = px.bar(
                year_counts,
                x='release_year',
                y='count',
                title="Number of Titles Released per Year (1990 onwards)",
                labels={'release_year': 'Release Year', 'count': 'Number of Titles'},
                color_discrete_sequence=['#E50914']
            )
            fig2.update_layout(height=500)
            st.plotly_chart(fig2, use_container_width=True)

            st.write("**Key Insights:**")
            peak_year = year_counts.loc[year_counts['count'].idxmax()]
            oldest_year = year_counts.loc[year_counts['release_year'].idxmin()] if len(year_counts) > 0 else None
            median_titles_per_year = year_counts['count'].median()
            st.write(f"- **{int(peak_year['release_year'])}** had the most titles released ({int(peak_year['count']):,} titles)")
            st.write("- Content release volume has grown substantially in the last two decades")
            st.write(f"- The median year in this range produced **{median_titles_per_year:.0f} titles**, a useful baseline for spotting standout years")
            if oldest_year is not None:
                st.write(f"- The earliest year shown ({int(oldest_year['release_year'])}) had just {int(oldest_year['count']):,} title(s), highlighting how much smaller the catalog was historically")
            span_years = int(year_counts['release_year'].max() - year_counts['release_year'].min())
            st.write(f"- This chart spans **{span_years} years** of releases, giving a long view of Netflix's content acquisition pace")

            pre_2015 = year_counts.loc[year_counts['release_year'] < 2015, 'count'].sum()
            post_2015 = year_counts.loc[year_counts['release_year'] >= 2015, 'count'].sum()
            years_pre = max((year_counts['release_year'] < 2015).sum(), 1)
            years_post = max((year_counts['release_year'] >= 2015).sum(), 1)
            avg_pre = pre_2015 / years_pre
            avg_post = post_2015 / years_post
            if avg_post > avg_pre:
                st.success(f"**Matches prior research**: average titles released per year rose from ~{avg_pre:.0f}/year before 2015 to ~{avg_post:.0f}/year from 2015 onward, consistent with Netflix's documented push into original productions.")
            else:
                st.info("This filtered slice doesn't show the post-2015 acceleration seen in prior research — try resetting filters to see the full-catalog pattern.")
        else:
            st.error("Column 'release_year' not found in dataset")

        st.markdown("---")

        # Chart 5: Content added by year (Line Chart)
        st.write("### Content Added to Netflix by Year")
        if 'added_year' in df_viz.columns:
            added_year_counts = df_viz['added_year'].value_counts().sort_index().reset_index()
            added_year_counts.columns = ['added_year', 'count']

            fig5 = px.line(
                added_year_counts,
                x='added_year',
                y='count',
                title="Content Added to Netflix by Year",
                labels={'added_year': 'Year Added', 'count': 'Number of Titles'},
                markers=True
            )
            fig5.update_traces(line_color='#E50914')
            fig5.update_layout(height=500)
            st.plotly_chart(fig5, use_container_width=True)

            st.write("**Key Insights:**")
            peak_add_year = added_year_counts.loc[added_year_counts['count'].idxmax()]
            lowest_add_year = added_year_counts.loc[added_year_counts['count'].idxmin()]
            total_added = added_year_counts['count'].sum()
            st.write(f"- **{int(peak_add_year['added_year'])}** saw the most titles added to the platform ({int(peak_add_year['count']):,} titles)")
            st.write("- Catalog growth accelerated sharply in the years leading up to the dataset's most recent snapshot")
            st.write(f"- **{int(lowest_add_year['added_year'])}** had the fewest additions ({int(lowest_add_year['count']):,} titles) among years shown")
            if len(added_year_counts) > 1:
                last_two = added_year_counts.tail(2)
                yoy_change = last_two.iloc[1]['count'] - last_two.iloc[0]['count']
                direction = "up" if yoy_change >= 0 else "down"
                st.write(f"- Year-over-year, the most recent period shown moved **{direction}** by {abs(yoy_change):,.0f} titles versus the prior year")
            st.write(f"- Across all years shown, **{total_added:,.0f} titles** were added to the catalog in this filtered view")
        else:
            st.error("Column 'added_year' not found in dataset")

        st.markdown("---")

        # Chart 12: Monthly content addition trend (Line Chart)
        st.write("### Monthly Content Addition Trend")
        if 'month_added' in df_viz.columns:
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            month_counts = df_viz[df_viz['month_added'] != 'Unknown']['month_added'].value_counts()
            month_counts = month_counts.reindex(month_order).fillna(0).reset_index()
            month_counts.columns = ['month_added', 'count']

            fig12 = px.line(
                month_counts,
                x='month_added',
                y='count',
                title="Monthly Content Addition Trend (All Years Combined)",
                labels={'month_added': 'Month', 'count': 'Number of Titles Added'},
                markers=True
            )
            fig12.update_traces(line_color='#221f1f')
            fig12.update_layout(height=500)
            st.plotly_chart(fig12, use_container_width=True)

            st.write("**Key Insights:**")
            peak_month = month_counts.loc[month_counts['count'].idxmax()]
            low_month = month_counts.loc[month_counts['count'].idxmin()]
            avg_month_count = month_counts['count'].mean()
            st.write(f"- **{peak_month['month_added']}** is the month with the most historical content additions ({int(peak_month['count']):,} titles)")
            st.write("- Monthly patterns can reveal seasonal content release strategy")
            st.write(f"- **{low_month['month_added']}** sees the fewest additions historically ({int(low_month['count']):,} titles)")
            st.write(f"- On average, roughly **{avg_month_count:.0f} titles** are added per month across the year")
            gap = peak_month['count'] - low_month['count']
            st.write(f"- The gap between the busiest and quietest months is **{int(gap):,} titles**, suggesting a deliberate release cadence rather than a flat schedule")
        else:
            st.error("Column 'month_added' not found in dataset")

        st.markdown("---")


    with tab_genre:
        # =========================================================================
        # Section C: Genre Analysis
        # =========================================================================
        st.subheader("Genre Analysis")

        # Chart 3: Top genres (Horizontal Bar Chart)
        st.write("### Top Genres on Netflix")
        if 'listed_in' in df_viz.columns:
            genre_counts = df_viz['listed_in'].value_counts().head(15).reset_index()
            genre_counts.columns = ['listed_in', 'count']

            fig3 = px.bar(
                genre_counts,
                x='count',
                y='listed_in',
                orientation='h',
                color='listed_in',
                title="Top 15 Genres by Number of Titles",
                labels={'count': 'Number of Titles', 'listed_in': 'Genre'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig3.update_layout(height=600, showlegend=False, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig3, use_container_width=True)

            st.write("**Key Insights:**")
            top3_genres_str = ", ".join(genre_counts.head(3)['listed_in'].tolist())
            genre_total = genre_counts['count'].sum()
            top5_share = genre_counts.head(5)['count'].sum() / genre_total * 100
            st.write(f"- **{genre_counts.iloc[0]['listed_in']}** is the most common genre ({genre_counts.iloc[0]['count']:,} titles)")
            st.write("- Genre distribution reflects Netflix's programming priorities across markets")
            st.write(f"- The top 3 genres are **{top3_genres_str}**, together forming the backbone of the catalog")
            st.write(f"- The top 5 genres shown account for **{top5_share:.0f}%** of titles among the top 15, showing how concentrated genre programming is")
            if len(genre_counts) > 1:
                smallest_top15 = genre_counts.iloc[-1]
                st.write(f"- Even within the top 15, **{smallest_top15['listed_in']}** trails with only {smallest_top15['count']:,} titles — a much smaller footprint than the leaders")

            top2_genres = set(genre_counts.head(2)['listed_in'].tolist())
            if 'Dramas' in top2_genres or 'Comedies' in top2_genres:
                st.success("**Matches prior research**: Drama and/or Comedy rank among the top genres here, consistent with published findings on this catalog.")
            else:
                st.info("This filtered slice ranks other genres above Drama/Comedy — try resetting filters to see the full-catalog pattern.")
        else:
            st.error("Column 'listed_in' not found in dataset")

        st.markdown("---")

        # Chart 8: Genre distribution by content type (Stacked Bar Chart)
        st.write("### Genre Distribution by Content Type")
        if 'listed_in' in df_viz.columns and 'type' in df_viz.columns:
            top_genres_list = df_viz['listed_in'].value_counts().head(10).index.tolist()
            genre_type_df = df_viz[df_viz['listed_in'].isin(top_genres_list)]
            genre_type_counts = genre_type_df.groupby(['listed_in', 'type']).size().reset_index(name='count')

            fig8 = px.bar(
                genre_type_counts,
                x='listed_in',
                y='count',
                color='type',
                title="Top 10 Genres Split by Movies vs TV Shows",
                labels={'listed_in': 'Genre', 'count': 'Number of Titles', 'type': 'Content Type'},
                color_discrete_sequence=px.colors.qualitative.Set2,
                barmode='stack'
            )
            fig8.update_layout(height=550, xaxis_tickangle=-40)
            st.plotly_chart(fig8, use_container_width=True)

            st.write("**Key Insights:**")
            genre_totals = genre_type_counts.groupby('listed_in')['count'].sum().sort_values(ascending=False)
            movie_heavy = genre_type_counts[genre_type_counts['type'] == 'Movie'].set_index('listed_in')['count']
            tv_heavy = genre_type_counts[genre_type_counts['type'] == 'TV Show'].set_index('listed_in')['count']
            st.write("- Some genres skew heavily toward Movies while others are TV Show dominated")
            st.write("- This split helps identify genre-specific format strategy on the platform")
            if len(genre_totals) > 0:
                st.write(f"- **{genre_totals.index[0]}** has the largest combined title count across both formats ({int(genre_totals.iloc[0]):,} titles)")
            if len(movie_heavy) > 0:
                top_movie_genre = movie_heavy.idxmax()
                st.write(f"- **{top_movie_genre}** leads in Movie count ({int(movie_heavy.max()):,} Movies) among the top 10 genres")
            if len(tv_heavy) > 0:
                top_tv_genre = tv_heavy.idxmax()
                st.write(f"- **{top_tv_genre}** leads in TV Show count ({int(tv_heavy.max()):,} TV Shows) among the top 10 genres")
        else:
            st.error("Columns 'listed_in' and/or 'type' not found in dataset")

        st.markdown("---")


    with tab_geo:
        # =========================================================================
        # Section D: Geographic Analysis
        # =========================================================================
        st.subheader("Geographic Analysis")

        # Prepare exploded country data used across the geographic charts
        country_exploded_list = explode_comma_column(df_viz['country']) if 'country' in df_viz.columns else []
        country_counts_series = pd.Series(country_exploded_list).value_counts() if country_exploded_list else pd.Series(dtype=int)

        # Chart 4: Top countries (Horizontal Bar Chart)
        st.write("### Top Countries by Number of Titles")
        if len(country_counts_series) > 0:
            country_counts = country_counts_series.head(15).reset_index()
            country_counts.columns = ['country', 'count']

            fig4 = px.bar(
                country_counts,
                x='count',
                y='country',
                orientation='h',
                color='country',
                title="Top 15 Countries by Number of Titles",
                labels={'count': 'Number of Titles', 'country': 'Country'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig4.update_layout(height=600, showlegend=False, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig4, use_container_width=True)

            st.write("**Key Insights:**")
            country_total = country_counts_series.sum()
            top5_country_share = country_counts.head(5)['count'].sum() / country_total * 100
            st.write(f"- **{country_counts.iloc[0]['country']}** produces the most titles on Netflix ({country_counts.iloc[0]['count']:,} titles)")
            st.write("- A handful of countries account for a disproportionate share of the catalog")
            st.write(f"- The top 5 countries alone account for **{top5_country_share:.0f}%** of all country mentions in the dataset")
            if len(country_counts) > 1:
                second_country = country_counts.iloc[1]
                st.write(f"- **{second_country['country']}** is the second-largest producer with {second_country['count']:,} titles")
            st.write(f"- **{len(country_counts_series)}** distinct countries are represented in the current filtered view")

            asia_europe = {'India', 'Japan', 'South Korea', 'China', 'Taiwan', 'Thailand', 'Indonesia',
                           'United Kingdom', 'France', 'Germany', 'Spain', 'Italy', 'Turkey', 'Poland'}
            top15_set = set(country_counts['country'].tolist())
            overlap = top15_set & asia_europe
            if overlap:
                st.success(f"**Matches prior research**: Asian/European countries ({', '.join(sorted(overlap))}) appear in the top 15, reflecting the geographic diversification noted in prior studies.")
            else:
                st.info("This filtered slice shows limited Asian/European representation in the top countries — try resetting filters to see the full-catalog pattern.")
        else:
            st.error("Column 'country' not found in dataset")

        st.markdown("---")

        # Chart 7: Movies vs TV Shows by country (Stacked Bar Chart)
        st.write("### Movies vs TV Shows by Country")
        if 'country' in df_viz.columns and 'type' in df_viz.columns and len(country_counts_series) > 0:
            top_countries_list = country_counts_series.head(10).index.tolist()

            country_type_rows = []
            for _, row in df_viz[['country', 'type']].dropna().iterrows():
                if row['country'] == 'Unknown':
                    continue
                for c in [x.strip() for x in str(row['country']).split(',')]:
                    if c in top_countries_list:
                        country_type_rows.append({'country': c, 'type': row['type']})
            country_type_df = pd.DataFrame(country_type_rows)

            if not country_type_df.empty:
                country_type_counts = country_type_df.groupby(['country', 'type']).size().reset_index(name='count')

                fig7 = px.bar(
                    country_type_counts,
                    x='country',
                    y='count',
                    color='type',
                    title="Top 10 Countries: Movies vs TV Shows",
                    labels={'country': 'Country', 'count': 'Number of Titles', 'type': 'Content Type'},
                    color_discrete_sequence=px.colors.qualitative.Set2,
                    barmode='stack',
                    category_orders={'country': top_countries_list}
                )
                fig7.update_layout(height=550, xaxis_tickangle=-30)
                st.plotly_chart(fig7, use_container_width=True)

                st.write("**Key Insights:**")
                country_type_totals = country_type_counts.groupby('country')['count'].sum().sort_values(ascending=False)
                movie_by_country = country_type_counts[country_type_counts['type'] == 'Movie'].set_index('country')['count']
                tv_by_country = country_type_counts[country_type_counts['type'] == 'TV Show'].set_index('country')['count']
                st.write("- Content format mix (Movies vs TV Shows) varies notably by country of production")
                st.write("- Some markets contribute mostly Movies, others contribute mostly TV Shows")
                if len(country_type_totals) > 0:
                    st.write(f"- **{country_type_totals.index[0]}** has the highest combined title count among the top 10 countries ({int(country_type_totals.iloc[0]):,} titles)")
                if len(movie_by_country) > 0:
                    st.write(f"- **{movie_by_country.idxmax()}** contributes the most Movies ({int(movie_by_country.max()):,}) among these countries")
                if len(tv_by_country) > 0:
                    st.write(f"- **{tv_by_country.idxmax()}** contributes the most TV Shows ({int(tv_by_country.max()):,}) among these countries")
            else:
                st.info("No country/type data available for this chart")
        else:
            st.error("Columns 'country' and/or 'type' not found in dataset")

        st.markdown("---")

        # Chart 11: World content distribution (Choropleth Map)
        st.write("### World Content Distribution")
        if len(country_counts_series) > 0:
            world_country_df = country_counts_series.reset_index()
            world_country_df.columns = ['country', 'count']

            fig11 = px.choropleth(
                world_country_df,
                locations='country',
                locationmode='country names',
                color='count',
                hover_name='country',
                title="Netflix Content Distribution Across the World",
                color_continuous_scale='reds'
            )
            fig11.update_layout(height=550)
            st.plotly_chart(fig11, use_container_width=True)

            st.write("**Key Insights:**")
            countries_with_content = len(world_country_df)
            single_title_countries = (world_country_df['count'] == 1).sum()
            top3_world_share = world_country_df.head(3)['count'].sum() / world_country_df['count'].sum() * 100
            st.write(f"- **{world_country_df.iloc[0]['country']}** has the deepest catalog presence globally ({int(world_country_df.iloc[0]['count']):,} titles)")
            st.write("- The map highlights where Netflix's content production and licensing footprint is concentrated")
            st.write(f"- **{countries_with_content}** countries are represented somewhere in the catalog in this filtered view")
            st.write(f"- The top 3 countries by title count make up **{top3_world_share:.0f}%** of all country mentions, showing a concentrated global footprint")
            if single_title_countries > 0:
                st.write(f"- **{single_title_countries}** countries have just a single title, pointing to long-tail markets with minimal content presence")
        else:
            st.error("Column 'country' not found in dataset")

        st.markdown("---")


    with tab_rd:
        # =========================================================================
        # Section E: Content Ratings & Duration
        # =========================================================================
        st.subheader("Content Ratings & Duration Analysis")

        # --- Rating Explanation Block ---
        st.markdown("""
        ### Understanding the `rating` Column
        The `rating` column records the **content/age rating** assigned to each title — the same kind of label
        you'd see on a DVD box or a cable TV guide (e.g., `PG-13`, `TV-MA`). It's one of the most useful columns
        in the dataset because it tells us *who a title is meant for*, not just what it's about.

        **Why Netflix uses content ratings:**
        - **Audience guidance**: Ratings tell viewers up front whether a title contains violence, language, or
          sexual content before they press play
        - **Parental controls & kid profiles**: Netflix's Kids profiles and parental PIN locks are built directly
          on top of these rating codes — a household can block anything above `TV-14`, for example
        - **Regulatory compliance**: Streaming platforms operate under different content-classification rules in
          different countries, so a standardized rating field is needed for licensing and legal compliance
        - **Content discovery & personalization**: Ratings feed into recommendation and search filtering, letting
          users browse "family-friendly" or "mature" sections of the catalog
        - **Catalog strategy signal**: For an analyst, the rating mix reveals *who* Netflix is programming
          for — a catalog skewed toward `TV-MA`/`TV-14` signals a primarily adult/teen audience strategy
        """)

        rating_legend = {
            'TV-MA': 'Mature Audience Only — may include graphic violence, explicit sexual content, or strong language',
            'TV-14': 'Parents Strongly Cautioned — may be unsuitable for children under 14',
            'TV-PG': 'Parental Guidance Suggested — may contain material unsuitable for younger children',
            'TV-G': 'General Audience — suitable for all ages',
            'TV-Y': 'All Children — suitable for children of all ages',
            'TV-Y7': 'Directed to Older Children — suitable for ages 7 and up',
            'TV-Y7-FV': 'Directed to Older Children (Fantasy Violence) — like TV-Y7, with more intense fantasy violence',
            'R': 'Restricted — viewers under 17 require an accompanying parent or guardian',
            'PG-13': 'Parents Strongly Cautioned — some material may be inappropriate for children under 13',
            'PG': 'Parental Guidance Suggested — some material may not be suitable for children',
            'G': 'General Audiences — suitable for all ages',
            'NC-17': 'Adults Only — no one 17 and under admitted',
            'NR': 'Not Rated — no official rating was provided',
            'UR': 'Unrated — an unrated version of a title',
        }
        with st.expander("**Rating Legend** — what each code in this dataset means"):
            present_ratings = df_viz['rating'].dropna().unique().tolist() if 'rating' in df_viz.columns else []
            legend_rows = [
                {'Rating': r, 'Meaning': rating_legend.get(r, 'Rating code not in standard legend')}
                for r in sorted(present_ratings)
            ]
            if legend_rows:
                st.dataframe(prepare_dataframe_for_display(pd.DataFrame(legend_rows)), use_container_width=True, hide_index=True)
            else:
                st.info("No rating values available in the current filtered selection.")

        st.markdown("---")

        # Chart 6: Content ratings distribution (Bar Chart)
        st.write("### Content Ratings Distribution")
        if 'rating' in df_viz.columns:
            rating_counts = df_viz['rating'].value_counts().reset_index()
            rating_counts.columns = ['rating', 'count']

            fig6 = px.bar(
                rating_counts,
                x='rating',
                y='count',
                color='rating',
                title="Content Ratings Distribution",
                labels={'rating': 'Content Rating', 'count': 'Number of Titles'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig6.update_layout(height=500, showlegend=False, xaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig6, use_container_width=True)

            st.write("**Key Insights:**")
            second_rating = rating_counts.iloc[1] if len(rating_counts) > 1 else None
            rare_ratings = rating_counts[rating_counts['count'] <= 10]
            st.write(f"- **{rating_counts.iloc[0]['rating']}** is the most common content rating ({rating_counts.iloc[0]['count']:,} titles)")
            st.write("- Rating distribution shows the platform primarily targets mature/teen audiences")
            if second_rating is not None:
                st.write(f"- **{second_rating['rating']}** is the second most common rating ({second_rating['count']:,} titles)")
            st.write(f"- **{len(rating_counts)}** distinct rating codes appear in the current filtered view")
            if len(rare_ratings) > 0:
                rare_list = ", ".join(rare_ratings['rating'].tolist())
                st.write(f"- Rarely-used ratings ({rare_list}) each have 10 or fewer titles, marking niche corners of the catalog")

            mature_ratings = ['TV-MA', 'TV-14', 'R', 'NC-17']
            mature_share = rating_counts.loc[rating_counts['rating'].isin(mature_ratings), 'count'].sum() / rating_counts['count'].sum() * 100
            if mature_share >= 50:
                st.success(f"**Matches prior research**: mature/teen ratings (TV-MA, TV-14, R, NC-17) make up {mature_share:.0f}% of titles here, confirming the adult-skewing catalog reported in prior studies.")
            else:
                st.info(f"In this filtered slice, mature/teen ratings only make up {mature_share:.0f}% — try resetting filters to see the full-catalog pattern.")
        else:
            st.error("Column 'rating' not found in dataset")

        st.markdown("---")

        # Chart 13: Content Rating by Type (Heatmap)
        st.write("### Content Rating by Type (Movies vs TV Shows)")
        if 'rating' in df_viz.columns and 'type' in df_viz.columns:
            rating_type_ct = pd.crosstab(df_viz['rating'], df_viz['type'])
            rating_type_ct = rating_type_ct.loc[df_viz['rating'].value_counts().index]

            fig13 = px.imshow(
                rating_type_ct,
                text_auto=True,
                aspect='auto',
                title="Content Rating by Type — Heatmap",
                labels={'x': 'Content Type', 'y': 'Content Rating', 'color': 'Number of Titles'},
                color_continuous_scale='reds'
            )
            fig13.update_layout(height=550)
            st.plotly_chart(fig13, use_container_width=True)

            st.write("**Key Insights:**")
            row_totals = rating_type_ct.sum(axis=1).sort_values(ascending=False)
            st.write("- Darker cells show which rating + format combinations are most common on the platform")
            if 'TV-MA' in rating_type_ct.index and 'TV Show' in rating_type_ct.columns and 'Movie' in rating_type_ct.columns:
                st.write(f"- **TV-MA**: {int(rating_type_ct.loc['TV-MA', 'Movie']):,} Movies vs {int(rating_type_ct.loc['TV-MA', 'TV Show']):,} TV Shows")
            st.write("- This view helps content teams see whether mature content leans toward Movies or TV Shows")
            if len(row_totals) > 0:
                st.write(f"- **{row_totals.index[0]}** is the single busiest rating/format combination overall, with {int(row_totals.iloc[0]):,} titles across both formats")
            movie_only_ratings = [r for r in rating_type_ct.index if rating_type_ct.loc[r].get('TV Show', 0) == 0 and rating_type_ct.loc[r].get('Movie', 0) > 0]
            if movie_only_ratings:
                st.write(f"- Rating(s) **{', '.join(movie_only_ratings)}** appear exclusively on Movies in this filtered view, with no matching TV Shows")
        else:
            st.error("Columns 'rating' and/or 'type' not found in dataset")

        st.markdown("---")

        # Chart 10: Duration distribution (Histogram)
        st.write("### Duration Distribution")
        if 'duration_num' in df_viz.columns and 'type' in df_viz.columns:
            fig10 = px.histogram(
                df_viz,
                x='duration_num',
                color='type',
                title="Duration Distribution (Minutes for Movies, Seasons for TV Shows)",
                labels={'duration_num': 'Duration', 'count': 'Number of Titles'},
                color_discrete_sequence=px.colors.qualitative.Set2,
                barmode='overlay',
                opacity=0.7,
                nbins=40
            )
            fig10.update_layout(height=500)
            st.plotly_chart(fig10, use_container_width=True)

            st.write("**Key Insights:**")
            movie_median = df_viz.loc[df_viz['type'] == 'Movie', 'duration_num'].median()
            tv_median = df_viz.loc[df_viz['type'] == 'TV Show', 'duration_num'].median()
            movie_max = df_viz.loc[df_viz['type'] == 'Movie', 'duration_num'].max()
            tv_max = df_viz.loc[df_viz['type'] == 'TV Show', 'duration_num'].max()
            st.write(f"- Median Movie duration is **{movie_median:.0f} minutes**")
            st.write(f"- Median TV Show length is **{tv_median:.0f} season(s)**")
            st.write(f"- The longest Movie in this filtered view runs **{movie_max:.0f} minutes**")
            st.write(f"- The longest-running TV Show has **{tv_max:.0f} season(s)**")
            st.write("- Movies cluster tightly around typical feature-length runtimes, while TV Shows are heavily skewed toward just 1-2 seasons")
        else:
            st.error("Column 'duration_num' and/or 'type' not found in dataset")

        st.markdown("---")

        # Chart 14: Movie Duration by Rating (Box Plot)
        st.write("### Movie Duration by Content Rating")
        if 'duration_num' in df_viz.columns and 'rating' in df_viz.columns and 'type' in df_viz.columns:
            movies_only = df_viz[df_viz['type'] == 'Movie']
            if len(movies_only) > 0:
                top_ratings_for_box = movies_only['rating'].value_counts().head(8).index.tolist()
                box_data = movies_only[movies_only['rating'].isin(top_ratings_for_box)]

                fig14 = px.box(
                    box_data,
                    x='rating',
                    y='duration_num',
                    color='rating',
                    title="Movie Duration (Minutes) by Content Rating",
                    labels={'rating': 'Content Rating', 'duration_num': 'Duration (Minutes)'},
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig14.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig14, use_container_width=True)

                st.write("**Key Insights:**")
                median_by_rating = box_data.groupby('rating')['duration_num'].median().sort_values(ascending=False)
                overall_movie_median = box_data['duration_num'].median()
                st.write(f"- **{median_by_rating.index[0]}** movies run longest on average (median {median_by_rating.iloc[0]:.0f} min)")
                st.write("- Comparing runtime across ratings can reveal whether mature-rated films tend to run longer or shorter than family titles")
                st.write(f"- **{median_by_rating.index[-1]}** movies run shortest on average (median {median_by_rating.iloc[-1]:.0f} min)")
                st.write(f"- The overall median across these {len(top_ratings_for_box)} ratings is **{overall_movie_median:.0f} minutes**, a useful baseline for comparison")
                spread = median_by_rating.iloc[0] - median_by_rating.iloc[-1]
                st.write(f"- There's a **{spread:.0f}-minute** gap between the longest- and shortest-running rating categories")
            else:
                st.info("No Movie records available in the current filtered selection")
        else:
            st.error("Columns 'duration_num', 'rating', and/or 'type' not found in dataset")

        st.markdown("---")

        # Chart 15: Content Rating Trend Over Release Years (Area Chart)
        st.write("### Content Rating Mix Over Release Years")
        if 'rating' in df_viz.columns and 'release_year' in df_viz.columns:
            top_ratings_trend = df_viz['rating'].value_counts().head(5).index.tolist()
            trend_df = df_viz[df_viz['rating'].isin(top_ratings_trend) & (df_viz['release_year'] >= 2000)]
            rating_year_counts = trend_df.groupby(['release_year', 'rating']).size().reset_index(name='count')

            fig15 = px.area(
                rating_year_counts,
                x='release_year',
                y='count',
                color='rating',
                title="Top 5 Content Ratings: Release Mix Over Time (2000 onwards)",
                labels={'release_year': 'Release Year', 'count': 'Number of Titles', 'rating': 'Content Rating'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig15.update_layout(height=500)
            st.plotly_chart(fig15, use_container_width=True)

            st.write("**Key Insights:**")
            rating_year_totals = rating_year_counts.groupby('rating')['count'].sum().sort_values(ascending=False)
            latest_year_shown = rating_year_counts['release_year'].max()
            latest_year_mix = rating_year_counts[rating_year_counts['release_year'] == latest_year_shown].sort_values('count', ascending=False)
            st.write("- This view shows how the platform's rating mix has shifted release-year over release-year")
            st.write("- A widening `TV-MA`/`TV-14` band relative to family ratings would confirm the catalog's ongoing shift toward mature-audience programming")
            if len(rating_year_totals) > 0:
                st.write(f"- **{rating_year_totals.index[0]}** has contributed the most titles cumulatively since 2000 among the top 5 ratings ({int(rating_year_totals.iloc[0]):,} titles)")
            if len(latest_year_mix) > 0:
                st.write(f"- In **{int(latest_year_shown)}**, the largest release-year mix rating was **{latest_year_mix.iloc[0]['rating']}** ({int(latest_year_mix.iloc[0]['count']):,} titles)")
            st.write(f"- This chart covers **{int(rating_year_counts['release_year'].max() - rating_year_counts['release_year'].min())} years** of release history for the top 5 ratings")
        else:
            st.error("Columns 'rating' and/or 'release_year' not found in dataset")

        st.markdown("---")


    with tab_directors:
        # =========================================================================
        # Section F: Director Analysis
        # =========================================================================
        st.subheader("Director Analysis")

        # Chart 9: Top directors (Horizontal Bar Chart)
        st.write("### Top Directors by Number of Titles")
        if 'director' in df_viz.columns:
            director_list = explode_comma_column(df_viz['director'])
            if director_list:
                director_counts = pd.Series(director_list).value_counts().head(15).reset_index()
                director_counts.columns = ['director', 'count']

                fig9 = px.bar(
                    director_counts,
                    x='count',
                    y='director',
                    orientation='h',
                    title="Top 15 Directors by Number of Titles",
                    labels={'count': 'Number of Titles', 'director': 'Director'},
                    color='count',
                    color_continuous_scale='sunset'
                )
                fig9.update_layout(height=600, showlegend=False, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig9, use_container_width=True)

                st.write("**Key Insights:**")
                unknown_count = int((df_viz['director'] == 'Unknown').sum())
                unknown_pct = unknown_count / len(df_viz) * 100 if len(df_viz) > 0 else 0
                single_title_directors = (pd.Series(director_list).value_counts() == 1).sum()
                st.write(f"- **{director_counts.iloc[0]['director']}** has the most titles on Netflix ({director_counts.iloc[0]['count']} titles)")
                st.write("- A large share of titles list 'Unknown' directors, which was excluded from this ranking")
                if len(director_counts) > 1:
                    st.write(f"- **{director_counts.iloc[1]['director']}** ranks second with {director_counts.iloc[1]['count']} titles")
                st.write(f"- Titles with an 'Unknown' director make up **{unknown_pct:.0f}%** of the current filtered catalog ({unknown_count:,} titles)")
                st.write(f"- **{single_title_directors:,}** named directors in this view have contributed only a single title, showing most director credits are one-offs rather than repeat collaborations")
            else:
                st.info("No known director data available for visualization")
        else:
            st.error("Column 'director' not found in dataset")

        st.markdown("---")


    # =========================================================================
    # Summary Section
    # =========================================================================
    st.subheader("Data Analysis Summary")
    st.markdown("""
    This comprehensive analysis of the Netflix content dataset has revealed significant insights into the
    platform's catalog composition, genre mix, and geographic footprint. Through 15 targeted visualizations,
    we have uncovered valuable patterns that can guide content strategy and platform decisions.
    """)

    st.subheader("Major Findings & Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### **Content Composition**
        - **Format Split**: Movies significantly outnumber TV Shows in the overall catalog
        - **Genre Concentration**: A handful of genres (Dramas, Comedies, Documentaries) dominate the catalog
        - **Release Trends**: Content release volume has grown substantially over the past two decades

        ### **Geographic Footprint**
        - **Market Concentration**: A small number of countries (led by the United States and India) account for
          most of the catalog
        - **Format by Region**: The Movie/TV Show mix varies notably by country of production
        - **Global Reach**: The choropleth view shows Netflix content sourced from dozens of countries worldwide
        """)

    with col2:
        st.markdown("""
        ### **Ratings & Duration**
        - **Rating Skew**: Mature and teen ratings (TV-MA, TV-14) dominate the catalog
        - **Movie Length**: Most movies cluster around typical feature-length runtimes
        - **TV Show Length**: Most TV Shows are short-run, often just 1-2 seasons

        ### **Catalog Growth**
        - **Addition Trend**: Content additions to the platform accelerated sharply in recent years
        - **Seasonality**: Monthly addition patterns hint at seasonal content release strategy
        - **Director Concentration**: A small number of prolific directors account for a disproportionate share
          of named-director titles
        """)

    st.markdown("---")

    st.subheader("Business Implications & Recommendations")

    with st.expander("**For Content Strategists**", expanded=True):
        st.markdown("""
        **Strategic Recommendations:**

        1. **Genre Strategy**:
           - Continue investing in top-performing genres while monitoring emerging genre gaps
           - Balance catalog between crowd-pleasing genres and niche differentiation

        2. **Format Optimization**:
           - Evaluate whether the Movie/TV Show ratio matches viewing demand trends
           - Consider more multi-season TV commitments in regions where TV Shows under-index

        3. **Geographic Expansion**:
           - Identify under-represented countries as opportunities for local content investment
           - Study successful market mixes to replicate format strategy in new regions
        """)

    with st.expander("**For Platform Investors & Analysts**"):
        st.markdown("""
        **Investment Considerations:**

        1. **Catalog Gaps**:
           - Under-served genres and countries represent expansion opportunities
           - Rating distribution suggests room to grow family-friendly (G/PG/TV-Y) content

        2. **Growth Trajectory**:
           - Rising year-over-year additions signal continued content investment
           - Monthly addition trends can inform release-cadence planning

        3. **Risk Assessment**:
           - Heavy reliance on a few countries/genres could represent catalog concentration risk
        """)

    with st.expander("**For Market Researchers**"):
        st.markdown("""
        **Research Insights:**

        1. **Consumer Behavior**:
           - Rating and genre skew reflect the platform's core adult/teen audience
           - Duration patterns reflect standard industry runtimes for movies and season-based TV formats

        2. **Market Trends**:
           - Consolidation around top genres and countries
           - International (non-US) content share growing over time

        3. **Future Outlook**:
           - Continued diversification of genre and geographic mix likely
           - Format strategy (Movies vs TV Shows) will keep evolving with viewing trends
        """)

    st.markdown("---")
    st.subheader("Final Conclusion")

    if 'df_viz' in locals() and df_viz is not None:
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            if 'listed_in' in df_viz.columns:
                st.metric("Genres Analyzed", df_viz['listed_in'].nunique())

        with col2:
            total_titles = df_viz.shape[0]
            st.metric("Total Titles", f"{total_titles:,}")

        with col3:
            if 'country' in df_viz.columns:
                countries_analyzed = len(set(explode_comma_column(df_viz['country'])))
                st.metric("Countries Analyzed", countries_analyzed)

        with col4:
            if 'type' in df_viz.columns:
                movie_share = (df_viz['type'] == 'Movie').mean() * 100
                st.metric("Movies Share", f"{movie_share:.0f}%")

        with col5:
            if 'release_year' in df_viz.columns:
                st.metric("Median Release Year", int(df_viz['release_year'].median()))

    st.markdown("""
    ### **Project Impact**

    This analysis provides a comprehensive understanding of Netflix's content catalog through data-driven
    insights. The findings demonstrate the complex interplay between genre mix, geographic sourcing, content
    ratings, and release timing in shaping the platform's content strategy.

    **Key Takeaways:**

    1. **Data-Driven Catalog Planning**: Genre and country trends can directly inform acquisition priorities
    2. **Format Strategy**: The Movie/TV Show balance should be tuned to regional and genre-level demand
    3. **Growth Tracking**: Year-over-year and month-over-month addition trends reveal platform growth patterns
    4. **Audience Targeting**: Rating distribution reflects — and can help refine — audience targeting
    5. **Geographic Strategy**: Country-level analysis highlights where to expand local content investment
    """)

else:
    st.error("Dataset not available for visualization.")