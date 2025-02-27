import os
import sqlite3
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np

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

# Fetch team data (averages per game)
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
        ROUND(AVG(p1_score + p2_score + p3_score + p4_score), 1) AS Avg_Points,
        ROUND(AVG(fouls_total), 1) AS Avg_Fouls,
        ROUND(AVG(free_throws_made), 1) AS Avg_Free_Throws,
        ROUND(AVG(field_goals_made), 1) AS Avg_Field_Goals,
        ROUND(AVG(assists), 1) AS Avg_Assists,
        ROUND(AVG(rebounds_total), 1) AS Avg_Rebounds,
        ROUND(AVG(steals), 1) AS Avg_Steals,
        ROUND(AVG(turnovers), 1) AS Avg_Turnovers,
        ROUND(AVG(blocks), 1) AS Avg_Blocks
    FROM Teams
    GROUP BY name, tm
    ORDER BY Avg_Points DESC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def fetch_assists_vs_turnovers(game_type):
    if not table_exists("Teams"):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT name AS Team, AVG(assists) AS Avg_Assists, AVG(turnovers) AS Avg_Turnovers
    FROM Teams
    """
    
    if game_type == 'Home':
        query += "WHERE tm = 1 "
    elif game_type == 'Away':
        query += "WHERE tm = 0 "
    
    query += "GROUP BY name ORDER BY Avg_Assists DESC;"

    df = pd.read_sql(query, conn)
    conn.close()
    
    # Print the query results for debugging
    print(f"Query Results for {game_type} games: {df}")
    
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
        (CAST(P.points AS FLOAT) / NULLIF((P.two_pointers_attempted + P.three_pointers_attempted + 0.44 * P.free_throws_attempted),0)) AS 'PPS'

    FROM Players P
    JOIN Games G ON P.game_id = G.game_id
    WHERE LOWER(SUBSTR(P.first_name, 1, 1)) = ?
      AND LOWER(P.last_name) = ?
    ORDER BY G.game_id DESC;
    """

    df = pd.read_sql(query, conn, params=(first_initial, last_name))
    conn.close()
    return df

def fetch_shot_data(player_name):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT x_coord, y_coord, shot_result
    FROM Shots 
    WHERE player_name = ?;
    """
    df_shots = pd.read_sql_query(query, conn, params=(player_name,))
    conn.close()
    return df_shots
def fetch_league_shot_data():
    conn = sqlite3.connect(db_path)
    query = """
    SELECT x_coord, y_coord, shot_result
    FROM Shots;
    """
    df_league_shots = pd.read_sql_query(query, conn)
    conn.close()
    return df_league_shots
    
def fetch_player_and_league_stats_per_40(player_name):
    """Calculate player's stats per 40 minutes and compare against league averages per 40 minutes."""
    conn = sqlite3.connect(db_path)

    # ✅ Extract first initial and last name properly
    if ". " in player_name:
        first_initial, last_name = player_name.split(". ")
    else:
        parts = player_name.split(" ")
        first_initial = parts[0][0]
        last_name = " ".join(parts[1:])

    first_initial = first_initial.strip().lower()
    last_name = last_name.strip().lower()

    # ✅ Query for Player Stats
    player_query = """
    SELECT 
        minutes_played, points, rebounds_total, assists, steals, blocks, turnovers, field_goals_attempted
    FROM Players
    WHERE LOWER(SUBSTR(first_name, 1, 1)) = ?
      AND LOWER(last_name) = ?
    """

    df_player = pd.read_sql(player_query, conn, params=(first_initial, last_name))

    # ✅ Query for League Stats
    league_query = """
    SELECT 
        minutes_played, points, rebounds_total, assists, steals, blocks, turnovers, field_goals_attempted
    FROM Players
    """

    df_league = pd.read_sql(league_query, conn)
    conn.close()

    # ✅ Convert 'MM:SS' format to total minutes played
    def minutes_to_float(time_str):
        if time_str == "0:00" or not time_str:
            return 0
        mm, ss = map(int, time_str.split(":"))
        return mm + (ss / 60)  # Convert minutes + seconds to float format

    # ✅ Apply conversion for Player and League
    df_player["Total Minutes"] = df_player["minutes_played"].apply(minutes_to_float)
    df_league["Total Minutes"] = df_league["minutes_played"].apply(minutes_to_float)

    # ✅ Player Stats Calculation
    total_minutes_player = df_player["Total Minutes"].sum()  # Sum all valid minutes
    total_points_player = df_player["points"].sum()
    total_rebounds_player = df_player["rebounds_total"].sum()
    total_assists_player = df_player["assists"].sum()
    total_steals_player = df_player["steals"].sum()
    total_blocks_player = df_player["blocks"].sum()
    total_turnovers_player = df_player["turnovers"].sum()
    total_fga_player = df_player["field_goals_attempted"].sum()

    # ✅ League Stats Calculation
    total_minutes_league = df_league["Total Minutes"].sum()
    total_points_league = df_league["points"].sum()
    total_rebounds_league = df_league["rebounds_total"].sum()
    total_assists_league = df_league["assists"].sum()
    total_steals_league = df_league["steals"].sum()
    total_blocks_league = df_league["blocks"].sum()
    total_turnovers_league = df_league["turnovers"].sum()
    total_fga_league = df_league["field_goals_attempted"].sum()

    # ✅ Scale stats per 40 minutes for player
    if total_minutes_player > 0:
        scale_factor_player = 40 / total_minutes_player
        stats_per_40_player = {
            "Comparison": "Player per 40 min",
            "PTS": total_points_player * scale_factor_player,
            "REB": total_rebounds_player * scale_factor_player,
            "AST": total_assists_player * scale_factor_player,
            "STL": total_steals_player * scale_factor_player,
            "BLK": total_blocks_player * scale_factor_player,
            "TO": total_turnovers_player * scale_factor_player,
            "FGA": total_fga_player * scale_factor_player,
            "PPS": (total_points_player / (total_fga_player + 0.44 * total_turnovers_player)) if (total_fga_player + 0.44 * total_turnovers_player) > 0 else 0
        }
    else:
        stats_per_40_player = { "Comparison": "Player per 40 min", "PTS": 0, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TO": 0, "FGA": 0, "PPS": 0 }

    # ✅ Scale stats per 40 minutes for league (normalized)
    if total_minutes_league > 0:
        scale_factor_league = 40 / (total_minutes_league / len(df_league))  # Normalize by number of players
        stats_per_40_league = {
            "Comparison": "League per 40 min",
            "PTS": total_points_league * scale_factor_league / len(df_league),
            "REB": total_rebounds_league * scale_factor_league / len(df_league),
            "AST": total_assists_league * scale_factor_league / len(df_league),
            "STL": total_steals_league * scale_factor_league / len(df_league),
            "BLK": total_blocks_league * scale_factor_league / len(df_league),
            "TO": total_turnovers_league * scale_factor_league / len(df_league),
            "FGA": total_fga_league * scale_factor_league / len(df_league),
            "PPS": (total_points_league / (total_fga_league + 0.44 * total_turnovers_league)) if (total_fga_league + 0.44 * total_turnovers_league) > 0 else 0
        }
    else:
        stats_per_40_league = { "Comparison": "League per 40 min", "PTS": 0, "REB": 0, "AST": 0, "STL": 0, "BLK": 0, "TO": 0, "FGA": 0, "PPS": 0 }

    # ✅ Convert to DataFrame
    df_result = pd.DataFrame([stats_per_40_player, stats_per_40_league])

    return df_result

def plot_shot_coordinates(player_name):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT x_coord, y_coord
    FROM Shots 
    WHERE player_name = ?;
    """
    df_shots = pd.read_sql_query(query, conn, params=(player_name,))
    conn.close()

    if df_shots.empty:
        st.warning(f"No shot data found for {player_name}.")
        return

    # Invert x-coordinates for shots on the right side to the left side
    df_shots['x_coord'] = df_shots['x_coord'].apply(lambda x: 100 - x if x > 50 else x)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot individual shots
    ax.scatter(df_shots["x_coord"], df_shots["y_coord"], c="blue", s=35, alpha=0.6)

    # Remove all axis elements (clean chart)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis("off")  # Hide axis

    # Display chart in Streamlit
    st.pyplot(fig)

def plot_assists_vs_turnovers(data, game_type):
    st.subheader(f"📊 Assists vs Turnovers ({game_type} games)")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot
    sns.scatterplot(data=data, x="Avg_Turnovers", y="Avg_Assists", hue="Team", s=200, ax=ax)

    # Add mesh grid
    x_min, x_max = data["Avg_Turnovers"].min() - 1, data["Avg_Turnovers"].max() + 1
    y_min, y_max = data["Avg_Assists"].min() - 1, data["Avg_Assists"].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5), np.arange(y_min, y_max, 0.5))
    ax.plot(xx, yy, linestyle='-', color='grey', alpha=0.5)

    ax.set_xlabel("Average Turnovers per Game")
    ax.set_ylabel("Average Assists per Game")
    ax.set_title(f"Assists vs Turnovers ({game_type} games)")
    st.pyplot(fig)

# Function to calculate distance from the basket
def calculate_distance_from_basket(x, y, basket_x=50, basket_y=0):
    return np.sqrt((x - basket_x) ** 2 + (y - basket_y) ** 2)

def fetch_first_5_shots(player_name):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT x_coord, y_coord, shot_result
    FROM Shots 
    WHERE player_name = ?
    """
    df_shots = pd.read_sql_query(query, conn, params=(player_name,))
    conn.close()
    return df_shots

def plot_first_5_shots(df_shots):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect('equal')

    # Plot shots as dots
    ax.scatter(df_shots['x_coord'], df_shots['y_coord'], c='blue', s=100)

    # Annotate shots with their coordinates
    for i, row in df_shots.iterrows():
        ax.annotate(f"({row['x_coord']}, {row['y_coord']})", (row['x_coord'], row['y_coord']))

    ax.set_title("First 5 Shots and Coordinates")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    plt.grid(True)
    st.pyplot(fig)


def fetch_player_expected_stats(player_name):
    conn = sqlite3.connect(db_path)

    if ". " in player_name:
        first_initial, last_name = player_name.split(". ")
    else:
        parts = player_name.split(" ")
        first_initial = parts[0][0]
        last_name = " ".join(parts[1:])

    first_initial = first_initial.strip().lower()
    last_name = last_name.strip().lower()

    # Query to get total minutes played by the player
    query_player_minutes = """
    SELECT SUM((CAST(substr(minutes_played, 1, instr(minutes_played, ':') - 1) AS REAL) * 60) + 
               CAST(substr(minutes_played, instr(minutes_played, ':') + 1) AS REAL)) AS total_seconds
    FROM Players
    WHERE LOWER(SUBSTR(first_name, 1, 1)) = ? 
      AND LOWER(last_name) = ?;
    """
    
    df_player_minutes = pd.read_sql(query_player_minutes, conn, params=(first_initial, last_name))
    total_seconds = df_player_minutes.iloc[0, 0]  # Get the total seconds played

    if total_seconds is None or total_seconds == 0:
        conn.close()
        return pd.DataFrame()  # No minutes played, return empty DataFrame

    total_minutes = total_seconds / 60  # Convert seconds to minutes

    # Query to get league-wide per-minute averages
    query_league_per_minute = """
    SELECT 
        SUM(points) / SUM((CAST(substr(minutes_played, 1, instr(minutes_played, ':') - 1) AS REAL) * 60) + 
                          CAST(substr(minutes_played, instr(minutes_played, ':') + 1) AS REAL)) AS 'PTS',
        SUM(rebounds_total) / SUM((CAST(substr(minutes_played, 1, instr(minutes_played, ':') - 1) AS REAL) * 60) + 
                                  CAST(substr(minutes_played, instr(minutes_played, ':') + 1) AS REAL)) AS 'REB',
        SUM(assists) / SUM((CAST(substr(minutes_played, 1, instr(minutes_played, ':') - 1) AS REAL) * 60) + 
                           CAST(substr(minutes_played, instr(minutes_played, ':') + 1) AS REAL)) AS 'AST',
        SUM(steals) / SUM((CAST(substr(minutes_played, 1, instr(minutes_played, ':') - 1) AS REAL) * 60) + 
                          CAST(substr(minutes_played, instr(minutes_played, ':') + 1) AS REAL)) AS 'STL',
        SUM(blocks) / SUM((CAST(substr(minutes_played, 1, instr(minutes_played, ':') - 1) AS REAL) * 60) + 
                          CAST(substr(minutes_played, instr(minutes_played, ':') + 1) AS REAL)) AS 'BLK',
        SUM(turnovers) / SUM((CAST(substr(minutes_played, 1, instr(minutes_played, ':') - 1) AS REAL) * 60) + 
                             CAST(substr(minutes_played, instr(minutes_played, ':') + 1) AS REAL)) AS 'TO',
        SUM(field_goals_attempted) / SUM((CAST(substr(minutes_played, 1, instr(minutes_played, ':') - 1) AS REAL) * 60) + 
                                         CAST(substr(minutes_played, instr(minutes_played, ':') + 1) AS REAL)) AS 'FGA',
        SUM(points) / SUM(field_goals_attempted + 0.44 * free_throws_attempted) AS 'PPS'
    FROM Players;
    """

    df_league_per_minute = pd.read_sql(query_league_per_minute, conn)
    conn.close()

    # Multiply per-minute stats by the player's total minutes played
    df_expected_stats = df_league_per_minute * total_minutes
    df_expected_stats.insert(0, "Minutes Played", round(total_minutes, 1))

    return df_expected_stats
    
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
    COUNT(CASE WHEN minutes_played <> '0:00' THEN 1 END) AS games_played,

    ROUND(SUM(CAST(points AS REAL)) * 1.0 / NULLIF(COUNT(CASE WHEN minutes_played <> '0:00' THEN 1 END),0), 1) AS 'PTS',

    ROUND(SUM(CAST(field_goals_made AS REAL)) / NULLIF(SUM(CAST(field_goals_attempted AS REAL)),0), 2) AS 'FG%',
    
    ROUND(SUM(CAST(three_pointers_made AS REAL)) / NULLIF(SUM(CAST(three_pointers_attempted AS REAL)),0), 2) AS '3P%',
    
    ROUND(SUM(CAST(two_pointers_made AS REAL)) / NULLIF(SUM(CAST(two_pointers_attempted AS REAL)),0), 2) AS '2P%',
    
    ROUND(SUM(CAST(free_throws_made AS REAL)) / NULLIF(SUM(CAST(free_throws_attempted AS REAL)),0), 2) AS 'FT%',
    
    ROUND(SUM(CAST(rebounds_total AS REAL)) * 1.0 / NULLIF(COUNT(CASE WHEN minutes_played <> '0:00' THEN 1 END),0), 1) AS 'REB',
    
    ROUND(SUM(CAST(assists AS REAL)) * 1.0 / NULLIF(COUNT(CASE WHEN minutes_played <> '0:00' THEN 1 END),0), 1) AS 'AST',
    
    ROUND(SUM(CAST(steals AS REAL)) * 1.0 / NULLIF(COUNT(CASE WHEN minutes_played <> '0:00' THEN 1 END),0), 1) AS 'STL',
    
    ROUND(SUM(CAST(blocks AS REAL)) * 1.0 / NULLIF(COUNT(CASE WHEN minutes_played <> '0:00' THEN 1 END),0), 1) AS 'BLK',
    
    ROUND(SUM(CAST(turnovers AS REAL)) * 1.0 / NULLIF(COUNT(CASE WHEN minutes_played <> '0:00' THEN 1 END),0), 1) AS 'TO',
    
    ROUND(SUM(CAST(points AS REAL)) / NULLIF(SUM(CAST(field_goals_attempted AS REAL) + 0.44 * CAST(free_throws_attempted AS REAL)),0), 3) AS 'PPS'

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
           
            # Select numeric columns from the 4th column to the end
            numeric_cols = df.columns[3:]

            # Function to apply formatting and styling only to numeric columns from 4th column onward
            def format_and_style(df):
                styled_df = df.style.format({col: "{:.1f}" for col in numeric_cols})
                for col in numeric_cols:
                    styled_df = styled_df.highlight_max(subset=[col], color='lightgreen')
                    styled_df = styled_df.background_gradient(subset=[col], cmap='Blues')
                return styled_df

            # Apply the formatting and styling function
            styled_df = format_and_style(df)
            
            # Display the styled DataFrame
            st.dataframe(styled_df)

            col1, col2 = st.columns(2)

        with col1:
            # Dropdown menu for selecting stats
            stat_options = [
                'Avg_Points', 'Avg_Fouls', 'Avg_Free_Throws', 'Avg_Field_Goals', 
                'Avg_Assists', 'Avg_Rebounds', 'Avg_Steals', 'Avg_Turnovers', 'Avg_Blocks'
            ]
            selected_stat = st.selectbox("Select the statistic to display", stat_options)

        with col2:
            # Radio buttons for selecting combined, home, or away
            game_type = st.radio("Select game type", ('Combined', 'Home', 'Away'))

        # Filter the DataFrame based on the selected game type
        if game_type == 'Home':
            filtered_df = df[df['Location'] == 'Home']
        elif game_type == 'Away':
            filtered_df = df[df['Location'] == 'Away']
        else:
            filtered_df = df

        # Plot the selected statistic
        if not filtered_df.empty:
            st.subheader(f"{selected_stat} Statistics ({game_type} games)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Team', y=selected_stat, data=filtered_df, ax=ax, palette='viridis', ci=None)

            # Adding data labels
            for p in ax.patches:
                ax.annotate(format(p.get_height(), '.1f'),
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center',
                            xytext=(0, 9),
                            textcoords='offset points')

            ax.set_xlabel("Team")
            ax.set_ylabel(selected_stat)
            ax.set_title(f"{selected_stat} per Team ({game_type} games)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            sns.despine(top=True)  # Remove the top spine
            st.pyplot(fig)
        else:
            st.warning(f"No data available for {selected_stat} ({game_type} games)")

        # Add Assists vs Turnovers graph
        assists_vs_turnovers_df = fetch_assists_vs_turnovers(game_type)
        if not assists_vs_turnovers_df.empty:
            plot_assists_vs_turnovers(assists_vs_turnovers_df, game_type)
        else:
            st.warning("No data available for Assists vs Turnovers")                                  



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
            st.dataframe(df_referee.style.format({"Avg_Fouls_per_Game": "{:.3f}"}))
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
            df_shots = fetch_first_5_shots(player_name)
            st.dataframe(df_shots)
            plot_shot_coordinates(player_name)

            # Mean stats per game
            player_stats = fetch_player_stats(player_name)
            if not player_stats.empty:
                st.subheader(f"📊 {player_name} - Average Stats per Game")
                st.dataframe(player_stats.style.format({
    				"PTS": "{:.1f}",
    				"FG%": "{:.1%}",
    				"3P%": "{:.1%}",
    				"2P%": "{:.1%}",
    				"FT%": "{:.1%}",
    				"PPS": "{:.2f}"
				}))
            else:
                st.warning(f"No statistics available for {player_name}.")

            # Game-by-game stats
            player_game_stats = fetch_player_game_stats(player_name)
            if not player_game_stats.empty:
                st.subheader(f"📋 {player_name} - Game by Game Statistics")
                mean_values = player_game_stats.mean(numeric_only=True)
                mean_values['Game ID'] = 'Average'
                mean_values['MIN'] = '-'
                player_game_stats_with_mean = pd.concat([player_game_stats, mean_values.to_frame().T], ignore_index=True)
                st.dataframe(player_game_stats_with_mean.style.format({
                    "FG%": "{:.3f}%",
                    "3P%": "{:.3f}%",
                    "2P%": "{:.3f}%",
                    "FT%": "{:.3f}%",
                    "PPS": "{:.3f}",
                    "PTS": "{:.3f}",
                    "REB": "{:.3f}",
                    "AST": "{:.3f}",
                    "STL": "{:.3f}",
                    "BLK": "{:.3f}",
                    "TO": "{:.3f}"
                }))
            else:
                st.warning(f"No game-by-game stats available for {player_name}.")

            player_vs_league_40 = fetch_player_and_league_stats_per_40(player_name)
            if not player_vs_league_40.empty:
                st.subheader(f"📊 {player_name} vs. League - Stats per 40 Minutes")
                st.dataframe(player_vs_league_40.style.format({
                    "PTS": "{:.3f}",
                    "REB": "{:.3f}",
                    "AST": "{:.3f}",
                    "STL": "{:.3f}",
                    "BLK": "{:.3f}",
                    "TO": "{:.3f}",
                    "FGA": "{:.3f}",
                    "PPS": "{:.3f}"
                }))
            else:
                st.warning(f"No per-40 stats available for {player_name}.")
                st.warning(f"No per-40 stats available for {player_name}.")

if __name__ == "__main__":
    main()
