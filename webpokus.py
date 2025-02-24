import os
import sqlite3
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# ✅ Define SQLite database path (works locally & online)
db_path = os.path.join(os.path.dirname(__file__), "database.db")

# ✅ Function to check if a table exists
def table_exists(table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

# ✅ Fetch team data (averages per game)
def fetch_team_data():
    if not table_exists("Teams"):
        st.error("⚠️ Error: 'Teams' table not found in the database.")
        return pd.DataFrame()  # Return empty DataFrame

    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        name AS Team,
        CASE WHEN tm = 1 THEN 'Home' ELSE 'Away' END AS Location,
        COUNT(game_id) AS Games_Played,
        AVG(p1_score + p2_score + p3_score + p4_score) AS Avg_Points,
        AVG(fouls_total) AS Avg_Fouls,
        AVG(free_throws_made) AS Avg_Free_Throws,
        AVG(field_goals_made) AS Avg_Field_Goals,
        AVG(assists) AS Avg_Assists,
        AVG(rebounds_total) AS Avg_Rebounds,
        AVG(steals) AS Avg_Steals,
        AVG(turnovers) AS Avg_Turnovers,
        AVG(blocks) AS Avg_Blocks
    FROM Teams
    GROUP BY name, tm
    ORDER BY Avg_Points DESC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ✅ Fetch Assists vs Turnovers
def fetch_assists_vs_turnovers():
    if not table_exists("Teams"):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    query = """
    SELECT name AS Team, AVG(assists) AS Avg_Assists, AVG(turnovers) AS Avg_Turnovers
    FROM Teams
    GROUP BY name
    ORDER BY Avg_Assists DESC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ✅ Fetch referee statistics
def fetch_referee_data():
    if not table_exists("Officials"):
        st.error("⚠️ Error: 'Officials' table not found in the database.")
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    query = """
    SELECT o.first_name || ' ' || o.last_name AS Referee,
           COUNT(t.game_id) AS Games_Officiated,
           AVG(t.fouls_total) AS Avg_Fouls_per_Game
    FROM Officials o
    JOIN Teams t ON o.game_id = t.game_id
    WHERE o.role NOT LIKE 'commissioner'
    GROUP BY Referee
    ORDER BY Avg_Fouls_per_Game DESC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ✅ Fetch Player Names for Dropdown
def fetch_players():
    if not table_exists("Shots"):
        return []

    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT player_name FROM Shots ORDER BY player_name;"
    players = pd.read_sql(query, conn)["player_name"].tolist()
    conn.close()
    return players
def fetch_player_game_stats(player_name):
    conn = sqlite3.connect(db_path)

    if ". " in player_name:
        first_initial, last_name = player_name.split(". ")
    else:
        parts = player_name.split(" ")
        first_initial = parts[0][0]
        last_name = " ".join(parts[1:])

    first_initial = first_initial.strip().lower()
    last_name = last_name.strip().lower()

    query = """
    SELECT 
        G.game_id AS 'Game ID',
        P.minutes_played AS 'MIN',
        P.points AS 'PTS',
        P.field_goals_made AS 'FGM',
        P.field_goals_attempted AS 'FGA',
        CASE WHEN P.field_goals_attempted > 0 THEN CAST(P.field_goals_made AS FLOAT) / P.field_goals_attempted ELSE 0 END AS 'FG%',

        P.three_pointers_made AS '3PM',
        P.three_pointers_attempted AS '3PA',
        CASE WHEN P.three_pointers_attempted > 0 THEN CAST(P.three_pointers_made AS FLOAT) / P.three_pointers_attempted ELSE 0 END AS '3P%',

        P.two_pointers_made AS '2PM',
        P.two_pointers_attempted AS '2PA',
        CASE WHEN P.two_pointers_attempted > 0 THEN CAST(P.two_pointers_made AS FLOAT) / P.two_pointers_attempted ELSE 0 END AS '2P%',

        P.free_throws_made AS 'FTM',
        P.free_throws_attempted AS 'FTA',
        CASE WHEN P.free_throws_attempted > 0 THEN CAST(P.free_throws_made AS FLOAT) / P.free_throws_attempted ELSE 0 END AS 'FT%',

        P.rebounds_total AS 'REB',
        P.assists AS 'AST',
        P.steals AS 'STL',
        P.blocks AS 'BLK',
        P.turnovers AS 'TO',

        -- Corrected PPS calculation here
        (CAST(P.points AS FLOAT) / NULLIF((P.field_goals_attempted + 0.44 * P.free_throws_attempted),0)) AS 'PPS'

    FROM Players P
    JOIN Games G ON P.game_id = G.game_id
    WHERE LOWER(SUBSTR(P.first_name, 1, 1)) = ?
      AND LOWER(P.last_name) = ?
    ORDER BY G.game_id DESC;
    """

    df = pd.read_sql(query, conn, params=(first_initial, last_name))
    conn.close()
    return df
def fetch_league_average_stats():
    conn = sqlite3.connect(db_path)

    query = """
    SELECT 
        AVG(CAST(points AS REAL)) AS 'PTS',
        AVG(CAST(field_goals_made AS REAL)) AS 'FGM',
        AVG(CAST(field_goals_attempted AS REAL)) AS 'FGA',
        SUM(CAST(field_goals_made AS REAL)) / NULLIF(SUM(field_goals_attempted),0) AS 'FG%',

        AVG(CAST(three_pointers_made AS REAL)) AS '3PM',
        AVG(CAST(three_pointers_attempted AS REAL)) AS '3PA',
        SUM(CAST(three_pointers_made AS REAL)) / NULLIF(SUM(three_pointers_attempted),0) AS '3P%',

        AVG(CAST(two_pointers_made AS REAL)) AS '2PM',
        AVG(CAST(two_pointers_attempted AS REAL)) AS '2PA',
        SUM(CAST(two_pointers_made AS REAL)) / NULLIF(SUM(two_pointers_attempted),0) AS '2P%',

        AVG(CAST(free_throws_made AS REAL)) AS 'FTM',
        AVG(CAST(free_throws_attempted AS REAL)) AS 'FTA',
        SUM(CAST(free_throws_made AS REAL)) / NULLIF(SUM(free_throws_attempted),0) AS 'FT%',

        AVG(CAST(rebounds_total AS REAL)) AS 'REB',
        AVG(CAST(assists AS REAL)) AS 'AST',
        AVG(CAST(steals AS REAL)) AS 'STL',
        AVG(CAST(blocks AS REAL)) AS 'BLK',
        AVG(CAST(turnovers AS REAL)) AS 'TO',

        SUM(CAST(points AS REAL)) / NULLIF(SUM(CAST(field_goals_attempted AS REAL) + 0.44 * CAST(free_throws_attempted AS REAL)),0) AS 'PPS'

    FROM Players;
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df
    
def fetch_player_stats(player_name):
    conn = sqlite3.connect(db_path)

    if ". " in player_name:
        first_initial, last_name = player_name.split(". ")
    else:
        parts = player_name.split(" ")
        first_initial = parts[0][0]
        last_name = " ".join(parts[1:])

    first_initial = first_initial.strip().lower()
    last_name = last_name.strip().lower()

    query = """
    SELECT 
        AVG(CAST(points AS REAL)) AS 'PTS',
        AVG(CAST(field_goals_made AS REAL)) AS 'FGM',
        AVG(CAST(field_goals_attempted AS REAL)) AS 'FGA',
        SUM(CAST(field_goals_made AS REAL))*1.0 / NULLIF(SUM(CAST(field_goals_attempted AS REAL)),0) AS 'FG%',

        AVG(CAST(three_pointers_made AS REAL)) AS '3PM',
        AVG(CAST(three_pointers_attempted AS REAL)) AS '3PA',
        SUM(CAST(three_pointers_made AS REAL))*1.0 / NULLIF(SUM(CAST(three_pointers_attempted AS REAL)),0) AS '3P%',

        AVG(CAST(two_pointers_made AS REAL)) AS '2PM',
        AVG(CAST(two_pointers_attempted AS REAL)) AS '2PA',
        SUM(CAST(two_pointers_made AS REAL))*1.0 / NULLIF(SUM(CAST(two_pointers_attempted AS REAL)),0) AS '2P%',

        AVG(CAST(free_throws_made AS REAL)) AS 'FTM',
        AVG(CAST(free_throws_attempted AS REAL)) AS 'FTA',
        SUM(CAST(free_throws_made AS REAL))*1.0 / NULLIF(SUM(CAST(free_throws_attempted AS REAL)),0) AS 'FT%',

        AVG(CAST(rebounds_total AS REAL)) AS 'REB',
        AVG(CAST(assists AS REAL)) AS 'AST',
        AVG(CAST(steals AS REAL)) AS 'STL',
        AVG(CAST(blocks AS REAL)) AS 'BLK',
        AVG(CAST(turnovers AS REAL)) AS 'TO',

        -- Corrected PPS calculation here
        SUM(CAST(points AS REAL)) / NULLIF(SUM(CAST(field_goals_attempted AS REAL) + 0.44 * CAST(free_throws_attempted AS REAL)),0) AS 'PPS'

    FROM Players
    WHERE LOWER(SUBSTR(first_name, 1, 1)) = ?
      AND LOWER(last_name) = ?
    GROUP BY LOWER(first_name), LOWER(last_name);
    """

    df = pd.read_sql(query, conn, params=(first_initial, last_name))
    conn.close()
    return df

# ✅ Generate Shot Chart
def generate_shot_chart(player_name):
    """Generate a shot chart with heatmap restricted within the court boundaries."""

    if not os.path.exists("fiba_courtonly.jpg"):
        st.error("⚠️ Court image file 'fiba_courtonly.jpg' is missing!")
        return

    conn = sqlite3.connect(db_path)
    query = """
    SELECT x_coord, y_coord, shot_result
    FROM Shots 
    WHERE player_name = ?;
    """
    df_shots = pd.read_sql_query(query, conn, params=(player_name,))
    conn.close()

    if df_shots.empty:
        st.warning(f"❌ No shot data found for {player_name}.")
        return

    # ✅ Convert shot_result to match 'made' or 'missed' conditions
    df_shots["shot_result"] = df_shots["shot_result"].astype(str)
    df_shots["shot_result"] = df_shots["shot_result"].replace({"1": "made", "0": "missed"})

    # ✅ Scale coordinates to match court image dimensions
    df_shots["x_coord"] = df_shots["x_coord"] * 2.8  
    df_shots["y_coord"] = 261 - (df_shots["y_coord"] * 2.61)

    # ✅ Load court image
    court_img = mpimg.imread("fiba_courtonly.jpg")

    # ✅ Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(court_img, extent=[0, 280, 0, 261], aspect="auto")

    # ✅ Heatmap (restrict to court area)
    sns.kdeplot(
        data=df_shots, 
        x="x_coord", y="y_coord", 
        cmap="coolwarm", fill=True, alpha=0.5, ax=ax, 
        bw_adjust=0.5, clip=[[0, 280], [0, 261]]  # 🔥 Restrict heatmap within the court
    )

    # ✅ Plot individual shots
    made_shots = df_shots[df_shots["shot_result"] == "made"]
    missed_shots = df_shots[df_shots["shot_result"] == "missed"]

    ax.scatter(made_shots["x_coord"], made_shots["y_coord"], 
               c="lime", edgecolors="black", s=35, alpha=1, zorder=3, label="Made Shots")

    ax.scatter(missed_shots["x_coord"], missed_shots["y_coord"], 
               c="red", edgecolors="black", s=35, alpha=1, zorder=3, label="Missed Shots")

    # ✅ Remove all axis elements (clean chart)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis("off")  # Hide axis

    # ✅ Display chart in Streamlit
    st.pyplot(fig)

# ✅ Main Function
# ✅ Main Function
def main():
    st.title("🏀 Basketball Stats Viewer")
    page = st.sidebar.selectbox("📌 Choose a page", ["Team Season Boxscore", "Head-to-Head Comparison", "Referee Stats", "Shot Chart"])

    if page == "Team Season Boxscore":
        df = fetch_team_data()
        if df.empty:
            st.warning("No team data available.")
        else:
            st.subheader("📊 Season Team Statistics (Averages Per Game)")
            numeric_cols = df.select_dtypes(include=['number']).columns
            st.dataframe(df.style.format({col: "{:.1f}" for col in numeric_cols}))

    elif page == "Head-to-Head Comparison":
        df = fetch_team_data()
        if df.empty:
            st.warning("No team data available.")
            return
        team_options = df["Team"].unique()
        st.subheader("🔄 Compare Two Teams Head-to-Head")
        team1 = st.selectbox("Select Team 1", team_options)
        team2 = st.selectbox("Select Team 2", team_options)
        if team1 != team2:
            st.subheader(f"📊 Season Stats Comparison: {team1} vs {team2}")
            numeric_cols = df.columns[3:]
            team1_stats = df[df["Team"] == team1][numeric_cols]
            team2_stats = df[df["Team"] == team2][numeric_cols]
            if team1_stats.empty or team2_stats.empty:
                st.error("⚠️ Error: One or both teams have no recorded stats.")
            else:
                team1_stats = team1_stats.T.rename(columns={team1_stats.index[0]: "Value"})
                team2_stats = team2_stats.T.rename(columns={team2_stats.index[0]: "Value"})
                team1_stats, team2_stats = team1_stats.align(team2_stats, join='outer', axis=0, fill_value=0)
                team1_stats["Stat"] = team1_stats.index
                team2_stats["Stat"] = team2_stats.index
                st.subheader(f"📉 {team1} Stats Per Game")
                fig1 = px.bar(team1_stats, x="Stat", y="Value", title=f"{team1} Stats Per Game", color="Stat")
                st.plotly_chart(fig1)
                st.subheader(f"📉 {team2} Stats Per Game")
                fig2 = px.bar(team2_stats, x="Stat", y="Value", title=f"{team2} Stats Per Game", color="Stat")
                st.plotly_chart(fig2)

    elif page == "Referee Stats":
        df_referee = fetch_referee_data()
        if df_referee.empty:
            st.warning("No referee data available.")
        else:
            st.subheader("🦺 Referee Statistics")
            st.dataframe(df_referee.style.format({"Avg_Fouls_per_Game": "{:.1f}"}))
            st.subheader("📉 Referee Stats: Average Fouls Called Per Game")
            fig_referee = px.bar(df_referee, x="Referee", y="Avg_Fouls_per_Game", labels={'Avg_Fouls_per_Game': 'Avg Fouls per Game'}, title="Average Fouls Per Game by Referee", color="Referee")
            st.plotly_chart(fig_referee)

    elif page == "Shot Chart":
        st.subheader("🎯 Player Shot Chart")
        players = fetch_players()
        if not players:
            st.warning("No player data available.")
        else:
            player_name = st.selectbox("Select a Player", players)
            generate_shot_chart(player_name)
            player_stats = fetch_player_stats(player_name)
            if not player_stats.empty:
                st.subheader(f"📊 {player_name} - Average Stats per Game")
                league_avg_stats = fetch_league_average_stats()
                league_avg_stats.insert(0, "Comparison", "League Average")
                player_stats.insert(0, "Comparison", player_name)
                combined_stats = pd.concat([player_stats, league_avg_stats], ignore_index=True)
                st.dataframe(combined_stats.style.format({
                    "PTS": "{:.1f}", "FG%": "{:.1%}", "3P%": "{:.1%}", "2P%": "{:.1%}", "FT%": "{:.1%}", "PPS": "{:.2f}"
                }))
            else:
                st.warning(f"No statistics available for {player_name}.")
            player_game_stats = fetch_player_game_stats(player_name)
            if not player_game_stats.empty:
                st.subheader(f"📋 {player_name} - Game by Game Statistics")
                mean_values = player_game_stats.mean(numeric_only=True)
                mean_values['Game ID'] = 'Player Average'
                mean_values['MIN'] = '-'
                player_game_stats_with_mean = pd.concat([player_game_stats, mean_values.to_frame().T], ignore_index=True)
                st.dataframe(player_game_stats_with_mean.style.format({
                    "FG%": "{:.1%}", "3P%": "{:.1%}", "2P%": "{:.1%}", "FT%": "{:.1%}", "PPS": "{:.2f}", "PTS": "{:.1f}", "FGM": "{:.1f}", "FGA": "{:.1f}", "3PM": "{:.1f}", "3PA": "{:.1f}", "2PM": "{:.1f}", "2PA": "{:.1f}", "FTM": "{:.1f}", "FTA": "{:.1f}", "REB": "{:.1f}", "AST": "{:.1f}", "STL": "{:.1f}", "BLK": "{:.1f}", "TO": "{:.1f}"
                }))
            else:
                st.warning(f"No game-by-game stats available for {player_name}.")

if __name__ == "__main__":
    main()
