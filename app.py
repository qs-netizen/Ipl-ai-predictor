
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="IPL AI Predictor",
    page_icon="🏏",
    layout="wide"
)

# =========================
# CHECK MODEL FILES
# =========================

if not os.path.exists("models/ipl_model.pkl"):
    st.error("ipl_model.pkl not found! Run train_model.py first.")
    st.stop()

if not os.path.exists("models/encoders.pkl"):
    st.error("encoders.pkl not found! Run train_model.py first.")
    st.stop()

# =========================
# LOAD MODEL
# =========================

model = joblib.load("models/ipl_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# =========================
# LOAD DATA
# =========================

matches = pd.read_csv("data/matches.csv")

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

h1 {
    color: #ff4b4b;
    text-align: center;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    background-color: #ff4b4b;
    color: white;
    font-size: 18px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================

st.title("🏏 IPL AI Match Predictor")

st.write("Predict IPL match winners using Machine Learning and AI.")

# =========================
# SIDEBAR
# =========================

st.sidebar.header("🏟 Match Settings")

teams = list(encoders['team1'].classes_)
venues = list(encoders['venue'].classes_)

team1 = st.sidebar.selectbox("Team 1", teams)
team2 = st.sidebar.selectbox("Team 2", teams)
toss_winner = st.sidebar.selectbox("Toss Winner", teams)
venue = st.sidebar.selectbox("Venue", venues)

# =========================
# PREDICT BUTTON
# =========================

if st.sidebar.button("Predict Winner"):

    input_df = pd.DataFrame({
        'team1': [encoders['team1'].transform([team1])[0]],
        'team2': [encoders['team2'].transform([team2])[0]],
        'toss_winner': [encoders['toss_winner'].transform([toss_winner])[0]],
        'venue': [encoders['venue'].transform([venue])[0]]
    })

    prediction = model.predict(input_df)

    probability = model.predict_proba(input_df)

    winner = encoders['winner'].inverse_transform(prediction)[0]

    st.success(f"🏆 Predicted Winner: {winner}")

    # =========================
    # PROBABILITY CHART
    # =========================

    prob_df = pd.DataFrame({
        'Team': encoders['winner'].classes_,
        'Probability': probability[0]
    })

    prob_df = prob_df.sort_values(
        by='Probability',
        ascending=False
    )

    st.subheader("📈 Winning Probability")

    fig = px.bar(
        prob_df,
        x='Team',
        y='Probability',
        title='Winning Probability'
    )

    st.plotly_chart(fig, use_container_width=True)

# =========================
# TEAM STATS
# =========================

st.header("📊 Team Statistics")

wins = matches['winner'].value_counts()

stats_df = pd.DataFrame({
    'Team': wins.index,
    'Wins': wins.values
})

fig2 = px.pie(
    stats_df,
    names='Team',
    values='Wins',
    title='IPL Total Wins'
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# HEAD TO HEAD
# =========================

st.header("⚔ Head-to-Head")

h2h = matches[
    (
        (matches['team1'] == team1) &
        (matches['team2'] == team2)
    ) |
    (
        (matches['team1'] == team2) &
        (matches['team2'] == team1)
    )
]

h2h_wins = h2h['winner'].value_counts()

if len(h2h_wins) > 0:

    h2h_df = pd.DataFrame({
        'Team': h2h_wins.index,
        'Wins': h2h_wins.values
    })

    fig3 = px.bar(
        h2h_df,
        x='Team',
        y='Wins',
        title='Head-to-Head Wins'
    )

    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("No head-to-head data available.")

# =========================
# RECENT MATCHES
# =========================

st.header("🔥 Recent Matches")

recent_matches = matches.tail(10)[[
    'team1',
    'team2',
    'winner',
    'venue'
]]

st.dataframe(recent_matches)

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption("Made with Python + Streamlit + Machine Learning")

