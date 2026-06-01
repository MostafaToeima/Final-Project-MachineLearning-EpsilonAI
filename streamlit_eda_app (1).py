"""
Student Performance EDA Dashboard
Epsilon AI Machine Learning Program | Real Dataset: 8,000 Students
streamlit_eda_app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F0FDFA; }
    .kpi-card {
        background: linear-gradient(135deg,#0D9488,#14B8A6);
        color:white; padding:18px 14px; border-radius:12px;
        text-align:center; box-shadow:0 4px 14px rgba(13,148,136,.25);
        margin-bottom:10px;
    }
    .kpi-card h2{font-size:2.1rem;margin:0;font-weight:800;}
    .kpi-card p{font-size:.82rem;margin:3px 0 0;opacity:.9;}
    .section-hdr{
        background:linear-gradient(90deg,#0D9488,#14B8A6);
        color:white;padding:9px 16px;border-radius:7px;
        font-size:1rem;font-weight:600;margin:14px 0 10px;
    }
    .insight-pos{background:white;border-left:5px solid #0D9488;
        padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:8px;
        box-shadow:0 2px 7px rgba(0,0,0,.06);}
    .insight-neg{background:#FFF7ED;border-left:5px solid #F59E0B;
        padding:10px 14px;border-radius:0 8px 8px 0;margin-bottom:8px;}
    .header-bar{
        background:linear-gradient(135deg,#065F46,#0D9488,#14B8A6);
        color:white;padding:26px;border-radius:14px;text-align:center;margin-bottom:20px;
    }
    .header-bar h1{font-size:1.85rem;margin:0;}
    .header-bar p{opacity:.9;margin:7px 0 0;}
</style>
""", unsafe_allow_html=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('student_performance_grade.csv')
    except FileNotFoundError:
        df = pd.read_csv('/mnt/user-data/uploads/student_performance_grade.csv')
    # Engineer features
    extra_flag = (df['Extracurricular'] == 'Yes').astype(int)
    df['Study_Efficiency']    = df['Hours_Studied'] * (df['Attendance'] / 100)
    df['Learning_Efficiency'] = df['Previous_GPA'] * df['Hours_Studied'] / (1 + df['Screen_Time'])
    df['Wellness_Score']      = df['Sleep_Hours']*10 - df['Stress_Level']*5 - df['Exam_Anxiety_Score']*3
    df['Engagement_Index']    = df['Attendance']*0.4 + df['Tutoring_Sessions_Per_Week']*10 + extra_flag*8 + df['Hours_Studied']*3
    return df

df = load_data()

GRADE_ORDER   = ['A', 'B', 'C', 'D', 'Fail']
GRADE_PALETTE = {'A':'#0D9488','B':'#14B8A6','C':'#F59E0B','D':'#F97316','Fail':'#EF4444'}
PALETTE       = ['#0D9488','#14B8A6','#5EEAD4','#0F766E','#F59E0B','#EF4444']

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Dashboard Filters")
    st.markdown("---")
    sel_grades  = st.multiselect("📊 Grade",   GRADE_ORDER,             default=GRADE_ORDER)
    sel_gender  = st.multiselect("👤 Gender",  sorted(df['Gender'].unique()), default=sorted(df['Gender'].unique()))
    sel_major   = st.multiselect("📚 Study Method", sorted(df['Study_Method'].unique()), default=sorted(df['Study_Method'].unique()))
    sel_diet    = st.multiselect("🥗 Diet Quality", sorted(df['Diet_Quality'].unique()), default=sorted(df['Diet_Quality'].unique()))
    att_range   = st.slider("📋 Attendance (%)", int(df['Attendance'].min()), 100, (int(df['Attendance'].min()), 100))
    st.markdown("---")
    st.markdown("**Epsilon AI ML Program**\n\n*8,000 Student Records*")

filt = (
    df['Grade'].isin(sel_grades) &
    df['Gender'].isin(sel_gender) &
    df['Study_Method'].isin(sel_major) &
    df['Diet_Quality'].isin(sel_diet) &
    df['Attendance'].between(att_range[0], att_range[1])
)
dff = df[filt].copy()

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-bar">
  <h1>🎓 Student Performance Analytics Dashboard</h1>
  <p>Exploratory Data Analysis · 8,000 Students · 19 Features · Epsilon AI Machine Learning Program</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI ROW ──────────────────────────────────────────────────────────────────
c1,c2,c3,c4,c5,c6 = st.columns(6)
kpis = [
    (c1, f"{len(dff):,}",             "👨‍🎓 Students"),
    (c2, f"{dff['Attendance'].mean():.1f}%",   "📋 Avg Attendance"),
    (c3, f"{dff['Previous_GPA'].mean():.2f}",  "📚 Avg GPA"),
    (c4, f"{dff['Final_Score'].mean():.1f}",   "🎯 Avg Score"),
    (c5, f"{dff['Hours_Studied'].mean():.1f}h","⏰ Study Hrs/Wk"),
    (c6, f"{(dff['Grade']=='A').mean()*100:.1f}%","🏆 Grade A Rate"),
]
for col, val, lbl in kpis:
    with col:
        st.markdown(f'<div class="kpi-card"><h2>{val}</h2><p>{lbl}</p></div>', unsafe_allow_html=True)

# ─── GRADE & SCORE DISTRIBUTIONS ─────────────────────────────────────────────
st.markdown('<div class="section-hdr">📊 Grade & Score Distributions</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    gc = dff['Grade'].value_counts().reindex(GRADE_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(5,3.5))
    bars = ax.bar(GRADE_ORDER, gc.values, color=[GRADE_PALETTE[g] for g in GRADE_ORDER], edgecolor='white', lw=1.5)
    for b,v in zip(bars,gc.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+10, f'{int(v):,}', ha='center', fontweight='bold', fontsize=9)
    ax.set_title('Grade Distribution', fontsize=11, fontweight='bold')
    ax.set_xlabel('Grade'); ax.set_ylabel('Students')
    ax.set_facecolor('#F0FDFA'); fig.patch.set_facecolor('#F0FDFA')
    plt.tight_layout(); st.pyplot(fig); plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(5,3.5))
    for grade in GRADE_ORDER:
        subset = dff[dff['Grade']==grade]['Final_Score']
        if len(subset)>0:
            ax.hist(subset, bins=25, alpha=0.65, label=grade, color=GRADE_PALETTE[grade], edgecolor='white')
    ax.set_title('Final Score by Grade', fontsize=11, fontweight='bold')
    ax.set_xlabel('Final Score'); ax.set_ylabel('Count')
    ax.legend(title='Grade', fontsize=8)
    ax.set_facecolor('#F0FDFA'); fig.patch.set_facecolor('#F0FDFA')
    plt.tight_layout(); st.pyplot(fig); plt.close()

with col3:
    fig, ax = plt.subplots(figsize=(5,3.5))
    mean_scores = dff.groupby('Grade')['Final_Score'].mean().reindex(GRADE_ORDER)
    ax.bar(GRADE_ORDER, mean_scores.values, color=[GRADE_PALETTE[g] for g in GRADE_ORDER], edgecolor='white')
    for i,v in enumerate(mean_scores.values):
        ax.text(i, v+0.3, f'{v:.1f}', ha='center', fontweight='bold', fontsize=9)
    ax.set_title('Mean Score per Grade', fontsize=11, fontweight='bold')
    ax.set_xlabel('Grade'); ax.set_ylabel('Mean Final Score')
    ax.set_ylim(25, 100)
    ax.set_facecolor('#F0FDFA'); fig.patch.set_facecolor('#F0FDFA')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ─── INTERACTIVE FEATURE EXPLORER ─────────────────────────────────────────────
st.markdown('<div class="section-hdr">🔍 Interactive Feature Explorer</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

feature_map = {
    'Hours Studied (Weekly)': 'Hours_Studied',
    'Attendance (%)': 'Attendance',
    'Previous GPA': 'Previous_GPA',
    'Exam Anxiety Score': 'Exam_Anxiety_Score',
    'Stress Level': 'Stress_Level',
    'Sleep Hours': 'Sleep_Hours',
    'Screen Time (hrs/day)': 'Screen_Time',
    'Tutoring Sessions/Week': 'Tutoring_Sessions_Per_Week',
}

with col1:
    feat_x = st.selectbox("X-axis Feature", list(feature_map.keys()), index=0)
    col_x  = feature_map[feat_x]
    fig, ax = plt.subplots(figsize=(6,4))
    sample  = dff.sample(min(2000, len(dff)), random_state=42)
    for grade in GRADE_ORDER:
        mask = sample['Grade']==grade
        ax.scatter(sample.loc[mask, col_x], sample.loc[mask,'Final_Score'],
                   alpha=0.35, s=12, color=GRADE_PALETTE[grade], label=grade)
    from scipy.stats import linregress as _lr
    if len(dff)>2:
        m,b,r,_,_ = _lr(dff[col_x], dff['Final_Score'])
        xs = np.linspace(dff[col_x].min(), dff[col_x].max(), 100)
        ax.plot(xs, m*xs+b, 'k--', lw=2, label=f'Trend (r={r:.2f})')
    ax.set_title(f'{feat_x} vs Final Score', fontsize=11, fontweight='bold')
    ax.set_xlabel(feat_x); ax.set_ylabel('Final Score')
    ax.legend(fontsize=8, markerscale=2.5)
    ax.set_facecolor('#F0FDFA'); fig.patch.set_facecolor('#F0FDFA')
    plt.tight_layout(); st.pyplot(fig); plt.close()

with col2:
    feat_bar = st.selectbox("Mean Score by Grade for:", list(feature_map.keys()), index=1)
    col_bar  = feature_map[feat_bar]
    mean_by  = dff.groupby('Grade')[col_bar].mean().reindex(GRADE_ORDER)
    fig, ax  = plt.subplots(figsize=(6,4))
    ax.bar(GRADE_ORDER, mean_by.values, color=[GRADE_PALETTE[g] for g in GRADE_ORDER], edgecolor='white')
    for i,v in enumerate(mean_by.values):
        ax.text(i, v+0.01*mean_by.max(), f'{v:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title(f'Mean {feat_bar} by Grade', fontsize=11, fontweight='bold')
    ax.set_xlabel('Grade'); ax.set_ylabel(f'Mean {feat_bar}')
    ax.set_facecolor('#F0FDFA'); fig.patch.set_facecolor('#F0FDFA')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ─── CORRELATION HEATMAP ──────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">🌡️ Correlation Heatmap</div>', unsafe_allow_html=True)
num_cols_h = ['Hours_Studied','Attendance','Sleep_Hours','Stress_Level','Screen_Time',
              'Previous_GPA','Tutoring_Sessions_Per_Week','Exam_Anxiety_Score','Final_Score']
corr_m = dff[num_cols_h].corr()
fig, ax = plt.subplots(figsize=(11,7))
mask = np.triu(np.ones_like(corr_m, dtype=bool))
sns.heatmap(corr_m, mask=mask, cmap=sns.diverging_palette(220,20,as_cmap=True),
            annot=True, fmt='.2f', vmin=-1, vmax=1, center=0, ax=ax,
            linewidths=0.5, linecolor='white', annot_kws={'size':9,'weight':'bold'})
ax.set_title('Feature Correlation Matrix (Real Data)', fontsize=13, fontweight='bold')
fig.patch.set_facecolor('#F0FDFA')
plt.tight_layout(); st.pyplot(fig); plt.close()

# ─── DEMOGRAPHIC & LIFESTYLE ANALYSIS ────────────────────────────────────────
st.markdown('<div class="section-hdr">👥 Demographic & Lifestyle Analysis</div>', unsafe_allow_html=True)
c1,c2,c3,c4 = st.columns(4)

for col, grp, title, ylim in [
    (c1, 'Gender',             'Avg Score by Gender',       (80,87)),
    (c2, 'Diet_Quality',       'Avg Score by Diet Quality', (78,88)),
    (c3, 'Internet_Quality',   'Avg Score by Internet',     (80,87)),
    (c4, 'Study_Method',       'Avg Score by Study Method', (80,87)),
]:
    with col:
        means = dff.groupby(grp)['Final_Score'].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(4.5,3.5))
        ax.bar(range(len(means)), means.values, color=PALETTE[:len(means)], edgecolor='white')
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=20, ha='right', fontsize=8)
        ax.set_title(title, fontsize=9.5, fontweight='bold')
        ax.set_ylabel('Avg Final Score')
        ax.set_ylim(ylim)
        for i,v in enumerate(means.values):
            ax.text(i, v+0.05, f'{v:.1f}', ha='center', fontsize=8, fontweight='bold')
        ax.set_facecolor('#F0FDFA'); fig.patch.set_facecolor('#F0FDFA')
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ─── HOURS STUDIED & ANXIETY IMPACT ──────────────────────────────────────────
st.markdown('<div class="section-hdr">📈 Key Driver Analysis</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    dff2 = dff.copy()
    dff2['Hours_Bucket'] = pd.cut(dff2['Hours_Studied'], bins=[0,2,4,6,8,12],
                                   labels=['0-2h','2-4h','4-6h','6-8h','8-12h'])
    h_means = dff2.groupby('Hours_Bucket', observed=True)['Final_Score'].mean()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(h_means.index, h_means.values, color=PALETTE[:5], edgecolor='white')
    for i,v in enumerate(h_means.values):
        ax.text(i, v+0.3, f'{v:.1f}', ha='center', fontweight='bold', fontsize=10)
    ax.set_title('Mean Score by Weekly Study Hours', fontsize=11, fontweight='bold')
    ax.set_xlabel('Hours Studied / Week'); ax.set_ylabel('Mean Final Score')
    ax.set_facecolor('#F0FDFA'); fig.patch.set_facecolor('#F0FDFA')
    plt.tight_layout(); st.pyplot(fig); plt.close()

with c2:
    dff3 = dff.copy()
    dff3['Anxiety_Bucket'] = pd.cut(dff3['Exam_Anxiety_Score'], bins=[0,3,5,7,10],
                                     labels=['Low\n1-3','Moderate\n3-5','High\n5-7','Very High\n7-10'])
    a_means = dff3.groupby('Anxiety_Bucket', observed=True)['Final_Score'].mean()
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(a_means.index, a_means.values,
           color=['#0D9488','#F59E0B','#F97316','#EF4444'], edgecolor='white')
    for i,v in enumerate(a_means.values):
        ax.text(i, v+0.3, f'{v:.1f}', ha='center', fontweight='bold', fontsize=10)
    ax.set_title('Mean Score by Exam Anxiety Level', fontsize=11, fontweight='bold')
    ax.set_xlabel('Anxiety Level'); ax.set_ylabel('Mean Final Score')
    ax.set_facecolor('#F0FDFA'); fig.patch.set_facecolor('#F0FDFA')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ─── KEY INSIGHTS ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">💡 Key Business Insights (Real Data)</div>', unsafe_allow_html=True)
c1,c2 = st.columns(2)

with c1:
    st.markdown("""
<div class="insight-pos"><strong>📚 #1 — Hours Studied is #1 Predictor (r = +0.591)</strong><br>
Students studying 8-12 hrs/week average 91+ points vs 70 points for those under 2 hrs.</div>
<div class="insight-pos"><strong>📋 #2 — Attendance Drives Grade</strong><br>
Grade A students average 87.8% attendance; Fail students average 65.4%.</div>
<div class="insight-pos"><strong>🥗 #3 — Good Diet = +4.3 Points</strong><br>
Students with good diet quality score 85.3 vs 81.0 for poor diet students.</div>
""", unsafe_allow_html=True)

with c2:
    st.markdown("""
<div class="insight-neg"><strong>😰 #4 — Exam Anxiety is #2 Risk Factor (r = -0.495)</strong><br>
Very High anxiety students (7-10) average 13 points lower than Low anxiety peers.</div>
<div class="insight-neg"><strong>🧘 #5 — Stress Costs Up to 10 Points (r = -0.297)</strong><br>
High stress students (8-10) average 77.4 vs 87.2 for low stress group.</div>
<div class="insight-pos"><strong>📖 #6 — Tutoring Sessions Help</strong><br>
Each additional weekly tutoring session correlates with ~1.8 higher final score.</div>
""", unsafe_allow_html=True)

# ─── RAW DATA ─────────────────────────────────────────────────────────────────
with st.expander("📋 View Raw Data (first 100 rows)"):
    st.dataframe(
        dff.drop('Student_ID', axis=1, errors='ignore').head(100).style.background_gradient(
            subset=['Final_Score','Hours_Studied','Attendance'], cmap='YlOrRd'),
        use_container_width=True
    )

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#0D9488;font-size:.88rem;'>
🎓 <strong>Epsilon AI Machine Learning Program</strong> — Student Performance EDA Dashboard<br>
Dataset: 8,000 Students · 19 Features · Real Data
</div>""", unsafe_allow_html=True)
