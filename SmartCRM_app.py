import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

REQUIRED_COLUMNS = ["name", "contact_number", "date", "product", "amount_spend"]

@st.cache_data
def parse_data(uploaded_file):
    ext = uploaded_file.name.lower().split(".")[-1]
    if ext in ["csv"]:
        df = pd.read_csv(uploaded_file)
    elif ext in ["xls", "xlsx"]:
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload CSV/XLS/XLSX.")
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount_spend"] = pd.to_numeric(df["amount_spend"], errors="coerce")
    df["name"] = df["name"].astype(str).str.strip()
    df["contact_number"] = df["contact_number"].astype(str).str.replace(r"\s+","", regex=True)
    df = df.dropna(subset=["date","amount_spend","name","contact_number"])
    return df

def compute_rfm(df, today=None):
    today = pd.to_datetime(today) if today is not None else df["date"].max() + pd.Timedelta(days=1)
    grp = df.groupby(["name","contact_number"])
    recency = (today - grp["date"].max()).dt.days.rename("Recency")
    frequency = grp["date"].count().rename("Frequency")
    monetary = grp["amount_spend"].sum().rename("Monetary")
    rfm = pd.concat([recency, frequency, monetary], axis=1).reset_index()
    rfm["R_Score"] = pd.qcut(-rfm["Recency"], 5, labels=[1,2,3,4,5]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_Sum"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    return rfm

def kmeans_segmentation(rfm, k):
    feats = rfm[["Recency","Frequency","Monetary"]].copy()
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    km = KMeans(n_clusters=k, init="k-means++", n_init="auto", random_state=42)
    labels = km.fit_predict(X)
    rfm_out = rfm.copy()
    rfm_out["Segment"] = labels
    inertia = km.inertia_
    sil = silhouette_score(X, labels) if k > 1 else np.nan
    return rfm_out, inertia, sil

def rule_based_offers(row):
    offers = []
    notes = []
    if row["R_Score"] <= 2:
        offers.append("Win-back: 20% off next order")
        notes.append("Low recency; encourage return")
    if row["F_Score"] >= 4 and row["M_Score"] >= 4:
        offers.append("VIP: Early access + bundled upsell")
        notes.append("High frequency and spend")
    if row["F_Score"] == 1:
        offers.append("Starter: Buy 1 Get 1")
        notes.append("Low frequency; activate")
    if row["M_Score"] == 1 and row["R_Score"] >= 4:
        offers.append("Value pack suggestion")
        notes.append("Recent but low spend; upsell")
    if not offers:
        offers = ["Standard 10% off"]
        notes.append("General promotion")
    return offers, "; ".join(notes)

st.set_page_config(page_title="Advanced Customer Analytics & Sales Intelligence", layout="wide")

# Custom CSS
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 10px;
    margin: 5px 0;
}
.highlight-box {
    background-color: #e8f4f8;
    padding: 15px;
    border-left: 5px solid #1f77b4;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🛍️ Advanced Customer Analytics & Sales Intelligence Dashboard")
st.markdown("**Comprehensive analysis including customer segmentation, product trends, and time-based sales patterns**")

with st.sidebar:
    st.header("📁 Upload Data")
    upl = st.file_uploader("Upload CSV/XLS/XLSX with columns: name, contact_number, date, product, amount_spend", type=["csv","xls","xlsx"])
    
    st.header("⚙️ Analysis Settings")
    k = st.slider("K-Means: number of clusters (k)", min_value=2, max_value=10, value=4, step=1)
    today_override = st.text_input("Analysis date (optional, e.g., 2025-09-30)")
    
    st.header("📊 View Options")
    show_segmentation = st.checkbox("Customer Segmentation", value=True)
    show_product_analysis = st.checkbox("Product Analysis", value=True) 
    show_time_analysis = st.checkbox("Time Pattern Analysis", value=True)
    show_seasonal_heatmap = st.checkbox("Seasonal Heatmaps", value=True)

if upl is None:
    st.info("📤 Upload a data file to begin comprehensive analysis.")
    st.markdown("""
    ### Expected Data Format:
    - **name**: Customer name (string)
    - **contact_number**: Phone number (string) 
    - **date**: Transaction date (YYYY-MM-DD format)
    - **product**: Product name (string)
    - **amount_spend**: Purchase amount (numeric)
    """)
    st.stop()

try:
    df = parse_data(upl)
except Exception as e:
    st.error(str(e))
    st.stop()

st.success(f"✅ Loaded {len(df):,} transactions for {df[['name','contact_number']].drop_duplicates().shape[0]:,} customers across {df['product'].nunique():,} products.")

# === BASIC SALES OVERVIEW ===
st.header("📈 Sales Overview Dashboard")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Revenue", f"${df['amount_spend'].sum():,.2f}")
col2.metric("Total Transactions", f"{len(df):,}")
col3.metric("Unique Customers", f"{df[['name','contact_number']].drop_duplicates().shape[0]:,}")
col4.metric("Unique Products", f"{df['product'].nunique():,}")
col5.metric("Date Range", f"{df['date'].nunique()} days")

# Daily sales trend
sales_daily = df.groupby(df["date"].dt.date)["amount_spend"].sum().reset_index(name="revenue")
fig_sales = px.line(sales_daily, x="date", y="revenue", title="📊 Daily Revenue Trend")
fig_sales.update_layout(xaxis_title="Date", yaxis_title="Revenue ($)")
st.plotly_chart(fig_sales, use_container_width=True)

# === PRODUCT ANALYSIS SECTION ===
if show_product_analysis:
    st.header("🛍️ Product Analysis & Insights")
    
    # Product analysis
    product_trends = df.groupby("product").agg({
        "amount_spend": ["sum", "mean", "count"],
        "date": ["min", "max"]
    }).round(2)
    product_trends.columns = ["Total_Revenue", "Avg_Sale_Value", "Transaction_Count", "First_Sale", "Last_Sale"]
    product_trends = product_trends.reset_index().sort_values("Total_Revenue", ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Products by Revenue")
        fig_prod_rev = px.bar(
            product_trends.head(15), 
            x="product", 
            y="Total_Revenue",
            title="Top 15 Products by Total Revenue",
            color="Total_Revenue",
            color_continuous_scale="viridis"
        )
        fig_prod_rev.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_prod_rev, use_container_width=True)
        
        # Show top products table
        st.markdown("**📋 Top 10 Products Performance**")
        st.dataframe(
            product_trends.head(10)[["product", "Total_Revenue", "Transaction_Count", "Avg_Sale_Value"]].style.format({
                "Total_Revenue": "${:,.2f}",
                "Avg_Sale_Value": "${:,.2f}"
            })
        )
    
    with col2:
        st.subheader("Products by Transaction Frequency")
        fig_prod_freq = px.bar(
            product_trends.head(15), 
            x="product", 
            y="Transaction_Count",
            title="Top 15 Products by Transaction Count",
            color="Transaction_Count",
            color_continuous_scale="plasma"
        )
        fig_prod_freq.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_prod_freq, use_container_width=True)
        
        # Product performance metrics
        st.markdown("**📊 Product Metrics**")
        avg_products_per_transaction = len(df) / df[['name','contact_number']].drop_duplicates().shape[0]
        st.metric("Avg Transactions per Customer", f"{avg_products_per_transaction:.1f}")
        st.metric("Most Valuable Product", f"${product_trends.iloc[0]['Total_Revenue']:,.2f}")
        st.metric("Most Popular Product", f"{product_trends.iloc[0]['Transaction_Count']} sales")

# === TIME PATTERN ANALYSIS ===
if show_time_analysis:
    st.header("⏰ Time Pattern & Seasonal Analysis")
    
    # Prepare time analysis data
    df_time = df.copy()
    df_time["year"] = df_time["date"].dt.year
    df_time["month"] = df_time["date"].dt.month
    df_time["day_of_week"] = df_time["date"].dt.day_name()
    df_time["month_name"] = df_time["date"].dt.month_name()
    df_time["quarter"] = df_time["date"].dt.quarter
    
    # Time-based charts in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📅 Monthly Trends", "📆 Daily Patterns", "⏰ Hourly Analysis", "📈 Seasonal Overview"])
    
    with tab1:
        # Monthly trends
        monthly_data = df_time.groupby(["year", "month"])["amount_spend"].sum().reset_index()
        monthly_data["date_str"] = monthly_data["year"].astype(str) + "-" + monthly_data["month"].astype(str).str.zfill(2)
        fig_monthly = px.line(monthly_data, x="date_str", y="amount_spend", title="Monthly Sales Trends")
        fig_monthly.update_layout(xaxis_title="Month", yaxis_title="Revenue ($)")
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Show monthly statistics
        st.markdown("**📊 Monthly Performance**")
        monthly_stats = df_time.groupby("month_name")["amount_spend"].agg(["sum", "mean", "count"]).round(2)
        monthly_stats.columns = ["Total_Revenue", "Avg_Transaction", "Transaction_Count"]
        monthly_stats = monthly_stats.sort_values("Total_Revenue", ascending=False)
        st.dataframe(monthly_stats.style.format({
            "Total_Revenue": "${:,.2f}",
            "Avg_Transaction": "${:,.2f}"
        }))
    
    with tab2:
        # Day of week patterns
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_data = df_time.groupby("day_of_week")["amount_spend"].agg(["sum", "mean", "count"]).reset_index()
        dow_data = dow_data.set_index("day_of_week").reindex(dow_order).reset_index()
        
        fig_dow = px.bar(dow_data, x="day_of_week", y="sum", title="Sales by Day of Week")
        fig_dow.update_layout(xaxis_title="Day of Week", yaxis_title="Total Revenue ($)")
        st.plotly_chart(fig_dow, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Best Day", dow_data.loc[dow_data["sum"].idxmax(), "day_of_week"])
            st.metric("Peak Day Revenue", f"${dow_data['sum'].max():,.2f}")
        with col2:
            st.metric("Lowest Day", dow_data.loc[dow_data["sum"].idxmin(), "day_of_week"]) 
            st.metric("Lowest Day Revenue", f"${dow_data['sum'].min():,.2f}")
    
    with tab3:
        # Hourly patterns (if hour data available)
        if "hour" in df_time.columns and df_time["hour"].notna().any():
            hourly_data = df_time.groupby("hour")["amount_spend"].agg(["sum", "mean", "count"]).reset_index()
            fig_hourly = px.line(hourly_data, x="hour", y="sum", title="Sales by Hour of Day")
            fig_hourly.update_layout(xaxis_title="Hour", yaxis_title="Revenue ($)")
            st.plotly_chart(fig_hourly, use_container_width=True)
            
            peak_hour = hourly_data.loc[hourly_data["sum"].idxmax(), "hour"]
            st.info(f"🕐 Peak Sales Hour: {peak_hour}:00 with ${hourly_data['sum'].max():,.2f} in revenue")
        else:
            st.info("⏰ Hour-level data not available in the dataset.")
    
    with tab4:
        # Quarterly and seasonal analysis  
        quarterly_data = df_time.groupby(["year", "quarter"])["amount_spend"].sum().reset_index()
        quarterly_data["quarter_label"] = quarterly_data["year"].astype(str) + " Q" + quarterly_data["quarter"].astype(str)
        
        fig_quarterly = px.bar(quarterly_data, x="quarter_label", y="amount_spend", title="Quarterly Sales Performance")
        st.plotly_chart(fig_quarterly, use_container_width=True)
        
        # Seasonal insights
        months_order = ["January", "February", "March", "April", "May", "June", 
                       "July", "August", "September", "October", "November", "December"]
        seasonal_data = df_time.groupby("month_name")["amount_spend"].agg(["sum", "mean", "count"]).reset_index()
        seasonal_data = seasonal_data.set_index("month_name").reindex(months_order).reset_index()
        
        fig_seasonal = px.bar(seasonal_data, x="month_name", y="sum", title="Sales by Month (All Years Combined)")
        fig_seasonal.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_seasonal, use_container_width=True)

# === PRODUCT-TIME CORRELATION ANALYSIS ===
if show_product_analysis and show_time_analysis:
    st.header("🔍 Product-Time Correlation Analysis")
    
    # Product performance by month
    product_by_month = df_time.groupby(["product", "month_name"])["amount_spend"].sum().unstack(fill_value=0)
    
    # Best selling product each month
    monthly_best = df_time.groupby(["month_name", "product"])["amount_spend"].sum().reset_index()
    monthly_best = monthly_best.loc[monthly_best.groupby("month_name")["amount_spend"].idxmax()]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗓️ Best Selling Product Each Month")
        monthly_best_display = monthly_best[["month_name", "product", "amount_spend"]].copy()
        monthly_best_display.columns = ["Month", "Top_Product", "Revenue"]
        st.dataframe(monthly_best_display.style.format({"Revenue": "${:,.2f}"}))
        
    with col2:
        st.subheader("📊 Product Performance Seasonality")
        if not product_by_month.empty:
            # Get top 5 products for heatmap
            top_products = df.groupby("product")["amount_spend"].sum().nlargest(10).index
            product_month_subset = product_by_month.loc[product_by_month.index.isin(top_products)]
            
            if not product_month_subset.empty:
                fig_prod_heatmap = px.imshow(
                    product_month_subset.values,
                    x=product_month_subset.columns,
                    y=product_month_subset.index,
                    title="Top 10 Products: Monthly Revenue Heatmap",
                    color_continuous_scale="viridis",
                    aspect="auto"
                )
                fig_prod_heatmap.update_layout(height=400)
                st.plotly_chart(fig_prod_heatmap, use_container_width=True)

# === SEASONAL HEATMAPS ===
if show_seasonal_heatmap:
    st.header("🌡️ Seasonal Sales Heatmap")
    
    # Create heatmap data
    heatmap_data = df_time.groupby(["month_name", "day_of_week"])["amount_spend"].sum().unstack(fill_value=0)
    
    # Reorder days and months
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    months_order = ["January", "February", "March", "April", "May", "June", 
                   "July", "August", "September", "October", "November", "December"]
    
    if not heatmap_data.empty:
        heatmap_data = heatmap_data.reindex(columns=[d for d in days_order if d in heatmap_data.columns], fill_value=0)
        heatmap_data = heatmap_data.reindex([m for m in months_order if m in heatmap_data.index], fill_value=0)
        
        fig_heatmap = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            title="Sales Heatmap: Month vs Day of Week",
            color_continuous_scale="blues",
            aspect="auto"
        )
        fig_heatmap.update_layout(height=500)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Peak analysis
        max_value = heatmap_data.max().max()
        max_pos = heatmap_data.stack().idxmax()
        st.info(f"🔥 Peak Sales: {max_pos[0]} on {max_pos[1]} with ${max_value:,.2f}")

# === CUSTOMER SEGMENTATION (Original RFM Analysis) ===
if show_segmentation:
    st.header("👥 Customer Segmentation & Personalized Offers")
    
    today_val = pd.to_datetime(today_override) if today_override else None
    rfm = compute_rfm(df, today=today_val)
    
    # KMeans segmentation
    rfm_seg, inertia, sil = kmeans_segmentation(rfm, k=k)
    
    # Segment summary
    seg_summary = rfm_seg.groupby("Segment").agg(
        Recency_mean=("Recency","mean"),
        Frequency_mean=("Frequency","mean"),
        Monetary_mean=("Monetary","mean"),
        Customers=("Segment","count")
    ).reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Segment Characteristics")
        st.dataframe(seg_summary.style.format({
            "Recency_mean": "{:.1f} days",
            "Frequency_mean": "{:.1f}",
            "Monetary_mean": "${:,.2f}"
        }))
        
        sil_text = f"{sil:.3f}" if not np.isnan(sil) else "NA"
        st.caption(f"📊 Model Performance - Inertia: {inertia:,.2f} | Silhouette: {sil_text}")

    
    with col2:
        # 3D scatter plot
        fig_scatter = px.scatter_3d(
            rfm_seg, x="Recency", y="Frequency", z="Monetary",
            color="Segment", hover_data=["name","contact_number","RFM_Sum"],
            title="RFM Customer Segments (3D View)"
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Personalized offers
    offers_df = rfm_seg.copy()
    offers_list = []
    notes_list = []
    for _, row in offers_df.iterrows():
        offers, notes = rule_based_offers(row)
        offers_list.append(", ".join(offers))
        notes_list.append(notes)
    offers_df["Recommended_Offers"] = offers_list
    offers_df["Offer_Notes"] = notes_list
    
    st.subheader("🎯 Customer Recommendations")
    display_cols = ["name","contact_number","Recency","Frequency","Monetary",
                   "R_Score","F_Score","M_Score","RFM_Sum","Segment","Recommended_Offers","Offer_Notes"]
    st.dataframe(offers_df[display_cols], use_container_width=True)

# === CUSTOMER SEARCH & ANALYSIS ===
st.header("🔍 Customer Search & Individual Analysis")

col1, col2 = st.columns(2)
with col1:
    q_name = st.text_input("🔍 Search by name (partial match)")
with col2:
    q_phone = st.text_input("📱 Search by phone (partial match)")

# Filter customer data for search
if show_segmentation:
    filtered = offers_df.copy()
else:
    # Create basic customer summary if segmentation is disabled
    customer_summary = df.groupby(["name", "contact_number"]).agg({
        "amount_spend": ["sum", "count"],
        "date": ["min", "max"],
        "product": "nunique"
    }).round(2)
    customer_summary.columns = ["Total_Spent", "Transaction_Count", "First_Purchase", "Last_Purchase", "Unique_Products"]
    filtered = customer_summary.reset_index()

if q_name:
    filtered = filtered[filtered["name"].str.contains(q_name, case=False, na=False)]
if q_phone:
    filtered = filtered[filtered["contact_number"].str.contains(q_phone, case=False, na=False)]

st.write(f"🎯 **Search Results: {len(filtered)} customers found**")

if len(filtered) > 0:
    st.dataframe(filtered.head(20), use_container_width=True)

# Individual customer analysis
if len(filtered) == 1:
    cust = filtered.iloc[0]
    st.markdown("---")
    st.markdown(f"### 👤 Customer Profile: {cust['name']} ({cust['contact_number']})")
    
    # Customer metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if show_segmentation:
        col1.metric("RFM Score", f"{int(cust['RFM_Sum'])}/15")
        col2.metric("Customer Segment", f"Segment {int(cust['Segment'])}")
        col3.metric("Total Spent", f"${cust['Monetary']:,.2f}")
        col4.metric("Purchase Frequency", f"{int(cust['Frequency'])} times")
        
        st.markdown(f"**🎯 Recommended Offers:** {cust['Recommended_Offers']}")
        st.markdown(f"**📝 Notes:** {cust['Offer_Notes']}")
    
    # Purchase history
    hist = df[(df["name"]==cust["name"]) & (df["contact_number"]==cust["contact_number"])].sort_values("date")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Purchase History Timeline")
        hist_daily = hist.groupby(hist["date"].dt.date)["amount_spend"].sum().reset_index(name="revenue")
        fig_hist = px.bar(hist_daily, x="date", y="revenue", title="Customer Spending Over Time")
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        st.markdown("#### 🛍️ Product Preferences")
        prod_pref = hist.groupby("product")["amount_spend"].sum().sort_values(ascending=False)
        fig_prod_pref = px.pie(values=prod_pref.values, names=prod_pref.index, 
                              title="Customer's Product Spending Distribution")
        st.plotly_chart(fig_prod_pref, use_container_width=True)
    
    st.markdown("#### 📋 Complete Transaction History")
    st.dataframe(hist[["date", "product", "amount_spend"]].style.format({"amount_spend": "${:,.2f}"}))

elif len(filtered) > 1:
    st.info("🔍 Multiple customers found. Refine your search to see individual customer analysis.")

# === INSIGHTS & RECOMMENDATIONS ===
with st.expander("💡 Key Insights & Business Recommendations"):
    if not df.empty:
        # Calculate key insights
        top_product = df.groupby("product")["amount_spend"].sum().idxmax()
        top_product_revenue = df.groupby("product")["amount_spend"].sum().max()
        
        if "month_name" in df_time.columns:
            best_month = df_time.groupby("month_name")["amount_spend"].sum().idxmax()
            best_month_revenue = df_time.groupby("month_name")["amount_spend"].sum().max()
        else:
            best_month = "Data unavailable"
            best_month_revenue = 0
            
        if "day_of_week" in df_time.columns:
            best_day = df_time.groupby("day_of_week")["amount_spend"].sum().idxmax()
        else:
            best_day = "Data unavailable"
        
        st.markdown(f"""
        ### 📊 **Key Findings:**
        
        **Product Performance:**
        - 🏆 **Top Product**: {top_product} (${top_product_revenue:,.2f} in total revenue)
        - 📦 **Product Portfolio**: {df['product'].nunique()} unique products in catalog
        
        **Seasonal Patterns:**
        - 🌟 **Peak Month**: {best_month} (${best_month_revenue:,.2f} in revenue)
        - 📅 **Best Day of Week**: {best_day}
        
        **Customer Insights:**
        - 👥 **Customer Base**: {df[['name','contact_number']].drop_duplicates().shape[0]:,} unique customers
        - 💰 **Average Customer Value**: ${df.groupby(['name','contact_number'])['amount_spend'].sum().mean():,.2f}
        - 🔄 **Repeat Customer Rate**: {(df.groupby(['name','contact_number']).size() > 1).mean():.1%}
        
        ### 🎯 **Actionable Recommendations:**
        
        1. **Product Strategy**: Focus marketing efforts on promoting {top_product} as it shows highest revenue potential
        2. **Seasonal Marketing**: Plan major campaigns for {best_month} when customer spending peaks
        3. **Operational Planning**: Increase staffing and inventory on {best_day}s for optimal customer service
        4. **Customer Retention**: Implement targeted offers for low-frequency customers to increase repeat purchases
        5. **Inventory Management**: Stock high-performing products more heavily during peak seasons
        """)
