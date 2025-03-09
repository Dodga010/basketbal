import os
import sqlite3
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d

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
        ROUND(AVG(blocks), 1) AS Avg_Blocks,
        ROUND(AVG(field_goals_made * 100.0 / field_goals_attempted), 2) AS eFG_percentage,
        ROUND(AVG(turnovers * 100.0 / (field_goals_attempted + 0.44 * free_throws_attempted)), 2) AS TOV_percentage,
        ROUND(AVG(rebounds_offensive * 100.0 / (rebounds_offensive + rebounds_defensive)), 2) AS ORB_percentage,
        ROUND(AVG(free_throws_attempted * 100.0 / field_goals_attempted), 2) AS FTR_percentage
    FROM Teams
    GROUP BY name, tm
    ORDER BY Avg_Points DESC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def fetch_team_matches(team_name):
    if not table_exists("Teams"):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT 
        t1.game_id,
        t1.name AS Team,
        (t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) AS Team_Score,
        (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score) AS Opponent_Score,
        t2.name AS Opponent,
        ROUND((t1.field_goals_made + 0.5 * t1.three_pointers_made) * 100.0 / t1.field_goals_attempted, 2) AS eFG_percentage,
        ROUND(t1.turnovers * 100.0 / (t1.field_goals_attempted + 0.44 * t1.free_throws_attempted), 2) AS TOV_percentage,
        ROUND(t1.rebounds_offensive * 100.0 / (t1.rebounds_offensive + t1.rebounds_defensive), 2) AS ORB_percentage,
        ROUND(t1.free_throws_attempted * 100.0 / t1.field_goals_attempted, 2) AS FTR_percentage
    FROM Teams t1
    JOIN Teams t2 ON t1.game_id = t2.game_id AND t1.name != t2.name
    WHERE t1.name = '{team_name}'
    ORDER BY t1.game_id;
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
        CASE WHEN P.field_goals_attempted > 0 THEN CAST(P.field_goals_made AS FLOAT) * 100 / P.field_goals_attempted ELSE 0 END AS 'FG%',

        P.three_pointers_made AS '3PM',
        P.three_pointers_attempted AS '3PA',
        CASE WHEN P.three_pointers_attempted > 0 THEN CAST(P.three_pointers_made AS FLOAT) * 100 / P.three_pointers_attempted ELSE 0 END AS '3P%',

        P.two_pointers_made AS '2PM',
        P.two_pointers_attempted AS '2PA',
        CASE WHEN P.two_pointers_attempted > 0 THEN CAST(P.two_pointers_made AS FLOAT) * 100 / P.two_pointers_attempted ELSE 0 END AS '2P%',

        P.free_throws_made AS 'FTM',
        P.free_throws_attempted AS 'FTA',
        CASE WHEN P.free_throws_attempted > 0 THEN CAST(P.free_throws_made AS FLOAT) * 100 / P.free_throws_attempted ELSE 0 END AS 'FT%',

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

def fetch_matches():
    conn = sqlite3.connect(db_path)
    query = """
    SELECT DISTINCT game_id FROM Games ORDER BY game_id DESC;
    """
    game_ids = pd.read_sql_query(query, conn)["game_id"].tolist()
    
    match_dict = {}
    for game_id in game_ids:
        team_query = "SELECT name FROM Teams WHERE game_id = ? ORDER BY tm;"
        teams = pd.read_sql_query(team_query, conn, params=(game_id,))["name"].tolist()
        if len(teams) == 2:
            match_name = f"{teams[0]} vs {teams[1]}"
            match_dict[match_name] = game_id
    
    conn.close()
    return match_dict

def fetch_team_stats(game_id, team_id):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT name, score, field_goals_made, field_goals_attempted, field_goal_percentage,
           three_pointers_made, three_pointers_attempted, three_point_percentage,
           two_pointers_made, two_pointers_attempted, two_point_percentage,
           free_throws_made, free_throws_attempted, free_throw_percentage,
           rebounds_total, assists, turnovers, steals, blocks, fouls_total
    FROM Teams
    WHERE game_id = ? AND tm = ?;
    """
    df = pd.read_sql_query(query, conn, params=(game_id, team_id))
    conn.close()
    return df

def count_substitutions(game_id):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT team_id, COUNT(*) / 2 AS substitutions
    FROM PlayByPlay
    WHERE game_id = ? AND action_type = 'substitution' AND sub_type IN ('in', 'out')
    GROUP BY team_id;
    """
    df = pd.read_sql_query(query, conn, params=(game_id,))
    conn.close()
    return df

def display_match_detail():
    st.subheader("🏀 Match Detail Analysis")
    match_dict = fetch_matches()
    selected_match_name = st.selectbox("Select a match:", list(match_dict.keys()))
    selected_match = match_dict[selected_match_name]
    
    if selected_match:
        teams = [1, 2]
        team_names = []
        team_stats = []
        
        for team in teams:
            stats = fetch_team_stats(selected_match, team)
            if not stats.empty:
                team_names.append(stats.iloc[0]["name"])
                team_stats.append(stats)
        
        if len(team_stats) == 2:
            st.write(f"### {team_names[0]} vs {team_names[1]}")
            
            # Reshape Data for Better Readability
            combined_stats = pd.concat(team_stats, ignore_index=True)
            combined_stats = combined_stats.T
            combined_stats.columns = [team_names[0], team_names[1]]
            combined_stats = combined_stats.reset_index().rename(columns={"index": "Statistic"})
            
            # Remove empty row names
            combined_stats = combined_stats.dropna(subset=[team_names[0], team_names[1]], how='all')
            
            # Convert numeric columns properly
            numeric_cols = combined_stats.columns[1:]
            combined_stats[numeric_cols] = combined_stats[numeric_cols].apply(pd.to_numeric, errors='coerce')
            
            # Display Table Without Highlighting
            st.dataframe(
                combined_stats.style.format(
                    {team_names[0]: "{:.1f}", team_names[1]: "{:.1f}"}
                )
            )
            
            # Fetch and Display Substitution Counts
            substitutions = count_substitutions(selected_match)
            if not substitutions.empty:
                st.subheader("🔄 Substitutions")
                substitutions["team_id"] = substitutions["team_id"].map({1: team_names[0], 2: team_names[1]})
                st.dataframe(substitutions.rename(columns={"team_id": "Team", "substitutions": "Total Substitutions"}))
    
def fetch_teams():
    if not table_exists("Teams"):
        return []
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT name FROM Teams ORDER BY name;"
    teams = pd.read_sql(query, conn)["name"].tolist()
    conn.close()
    return teams

def fetch_team_games(team_name):
    if not table_exists("Teams") or not table_exists("Games"):
        return []
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT t1.game_id, t2.name AS opponent_name
    FROM Teams t1
    JOIN Teams t2 ON t1.game_id = t2.game_id AND t1.name != t2.name
    WHERE t1.name = '{team_name}'
    ORDER BY t1.game_id;
    """
    games = pd.read_sql(query, conn)[["game_id", "opponent_name"]].to_records(index=False)
    conn.close()
    return games

def fetch_pbp_actions(game_id, selected_quarter=None):
    """Fetch play-by-play data for a specific quarter or the full game."""
    if not table_exists("PlayByPlay"):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)

    # If a specific quarter is selected, filter by period
    if selected_quarter is not None:
        query = """
        SELECT * FROM PlayByPlay
        WHERE game_id = ? AND period = ?
        ORDER BY pbp_id;
        """
        params = (game_id, selected_quarter)
    else:
        # Fetch all quarters if no specific quarter is selected
        query = """
        SELECT * FROM PlayByPlay
        WHERE game_id = ?
        ORDER BY period, pbp_id;
        """
        params = (game_id,)

    actions = pd.read_sql(query, conn, params=params)
    conn.close()
    return actions

def display_pbp_actions(actions):
    if actions.empty:
        st.warning("No actions found for the selected game and quarter.")
    else:
        st.dataframe(actions)

def plot_score_lead_full_game(game_id):
    """Plot the score lead progression for the entire game with correct left-to-right time direction."""
    pbp_data = fetch_pbp_actions(game_id)  # Fetch all quarters' data

    if pbp_data.empty:
        st.warning(f"No play-by-play data found for Game ID {game_id}.")
        return

    # Convert game_time to total seconds, ensuring each quarter starts at 10:00 and counts down
    pbp_data["Seconds"] = pbp_data.apply(
        lambda row: (row["period"] - 1) * 600 + (10 - int(row["game_time"].split(":")[0])) * 60 - int(row["game_time"].split(":")[1]),
        axis=1
    )

    # Sort data so that time progresses correctly across all quarters
    pbp_data = pbp_data.sort_values(by=["Seconds"], ascending=False)  # Reverse order to start from the left

    # Smooth the lead line using a rolling average
    pbp_data["Smoothed Lead"] = pbp_data["lead"].rolling(window=3, min_periods=1).mean()

    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=pbp_data["Seconds"], y=pbp_data["Smoothed Lead"], color="dodgerblue", linewidth=2, label="Lead Progression")

    # Formatting
    ax.axhline(0, color='gray', linestyle='--')  # Baseline where lead is neutral
    ax.set_xlabel("Game Time (Minutes:Seconds)", fontsize=12)
    ax.set_ylabel("Lead (Team 1 - Team 2)", fontsize=12)
    ax.set_title(f"Score Lead Progression - Full Game", fontsize=14)

    # Set x-ticks at proper intervals (every 5 minutes)
    max_seconds = 2400  # 4 quarters * 10 minutes * 60 seconds
    tick_positions = np.arange(0, max_seconds + 1, 300)  # Every 5 minutes
    tick_labels = [f"{(2400 - t) // 60}:{(2400 - t) % 60:02d}" for t in tick_positions]  # Format as MM:SS
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)

    # Show quarter separators
    for q in range(1, 4):  # Add dashed lines for Q1, Q2, Q3 (Q4 ends at 0:00)
        ax.axvline(x=q * 600, color='red', linestyle='dashed', alpha=0.5, label=f"End of Q{q}")

    # DO NOT invert x-axis (now it stays in correct left-to-right order)
    # Reversing the sorting above ensures correct plotting while keeping x-axis same

    # Show grid and legend
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    st.pyplot(fig)


def fetch_pbp_data(game_id):
    if not table_exists("PlayByPlay"):
        return pd.DataFrame()
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT 
        pbp_id,
        game_id,
        team_id,
        player_id,
        period,
        'REGULAR' as period_type,
        game_time,
        clock,
        current_score_team1,
        current_score_team2,
        lead,
        action_type,
        sub_type,
        previous_action,
        success,
        scoring,
        qualifiers
    FROM PlayByPlay
    WHERE game_id = {game_id}
    ORDER BY game_time DESC;
    """
    pbp_data = pd.read_sql(query, conn)
    conn.close()
    
    action_type_outcomes = ["2pt", "3pt", "assist", "block", "foul", "foulon", "freethrow", "game", "jumpball", "period", "rebound", "steal", "substitution", "timeout", "turnover"]
    sub_type_outcomes = ["layup", "dunk", "hook_shot", "fadeaway", "jump_shot", "tip_in", 
                         "bank_shot", "pull_up", "step_back", "floater", "alley_oop", 
                         "putback", "runner", "turnaround", "catch_and_shoot", "off_dribble"]
    qualifier_outcomes = ["fast_break", "second_chance", "contested", "open", "assisted", "and_one", 
                          "putback", "buzzer_beater", "catch_and_shoot", "off_dribble", 
                          "heavily_contested", "transition", "iso_play", "pick_and_roll", 
                          "spot_up", "post_up", "drive", "kick_out", "pull_up", 
                          "step_back", "floater", "alley_oop", "putback", "runner", "turnaround"]
    
    for index, row in pbp_data.iterrows():
        if row['action_type'] not in action_type_outcomes:
            pbp_data.at[index, 'action_type'] = None
        
        if row['sub_type'] not in sub_type_outcomes:
            pbp_data.at[index, 'sub_type'] = None
        
        if row['previous_action'] == 'blanket':
            pbp_data.at[index, 'previous_action'] = pbp_data.at[index + 1, 'previous_action'] if index + 1 < len(pbp_data) else None
        
        pbp_data.at[index, 'success'] = 1 if row['success'] == 1 else 0
        pbp_data.at[index, 'scoring'] = 1 if row['scoring'] == 1 else 0
        
        qualifiers = row['qualifiers'].split(", ") if row['qualifiers'] else []
        pbp_data.at[index, 'qualifiers'] = ", ".join([q for q in qualifiers if q in qualifier_outcomes])
    
    return pbp_data

def fetch_starting_five(game_id):
    """Fetches the starting five players for each team."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT first_name || ' ' || last_name AS player_name, team_id
    FROM Players
    WHERE game_id = ? AND starter = 1;
    """
    df = pd.read_sql_query(query, conn, params=(game_id,))
    conn.close()
    return df

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
def calculate_distance_from_basket(x, y, basket_x=6.2, basket_y=50):
    # Invert x-coordinates for shots on the right side
    x = 100 - x if x > 50 else x
    return np.sqrt((x - basket_x) ** 2 + (y - basket_y) ** 2)

def convert_units_to_meters(distance_units):
    # 35 units is 6.75 meters
    return distance_units * (6.75 / 35)

def display_shot_data_with_distance(player_name):
    # Fetch shot data
    df_shots = fetch_shot_data(player_name)
    
    if df_shots.empty:
        st.warning(f"No shot data found for {player_name}.")
        return
def calculate_moving_average(df_shots, window_size=5):
    distances = df_shots["distance_from_basket_m"]
    # Calculate the histogram
    hist, bin_edges = np.histogram(distances, bins=np.arange(0, distances.max() + 1, 0.5), density=True)
    # Calculate the moving average
    moving_avg = np.convolve(hist, np.ones(window_size)/window_size, mode='valid')
    x_grid = bin_edges[:len(moving_avg)]
    return x_grid, moving_avg
    
def calculate_moving_average(df_shots, window_size=5):
    distances = df_shots["distance_from_basket_m"]
    # Calculate the histogram
    hist, bin_edges = np.histogram(distances, bins=np.arange(0, distances.max() + 1, 0.5), density=True)
    # Calculate the moving average
    moving_avg = np.convolve(hist, np.ones(window_size)/window_size, mode='valid')
    x_grid = bin_edges[:len(moving_avg)]
    return x_grid, moving_avg

    # Calculate distances in units
    df_shots["distance_from_basket_units"] = df_shots.apply(lambda row: calculate_distance_from_basket(row["x_coord"], row["y_coord"]), axis=1)

    # Convert distances to meters
    df_shots["distance_from_basket_m"] = df_shots["distance_from_basket_units"].apply(convert_units_to_meters)

    # Display the DataFrame
    st.dataframe(df_shots)

def calculate_interpolated_distribution(df_shots):
    distances = df_shots["distance_from_basket_m"]
    hist, bin_edges = np.histogram(distances, bins=np.arange(0, distances.max() + 1, 1), density=True)
    
    # Midpoints of bins
    x = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Interpolation with smoothing factor
    spline = UnivariateSpline(x, hist, s=0)
    x_smooth = np.linspace(x.min(), x.max(), 100)
    y_smooth = spline(x_smooth)
    
    return x_smooth, y_smooth
    
def plot_fg_percentage_with_frequency(player_name, window_size=5):
    df_player_shots = fetch_shot_data(player_name)
    df_league_shots = fetch_league_shot_data()

    if df_player_shots.empty:
        st.warning(f"No shot data found for {player_name}.")
        return

    player_fg_percentage, player_shot_counts = calculate_fg_percentage_by_distance(df_player_shots, window_size=window_size)
    league_fg_percentage, league_shot_counts = calculate_fg_percentage_by_distance(df_league_shots, window_size=window_size)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot FG percentage
    ax1.plot(player_fg_percentage.index, player_fg_percentage.values, color='green', label=f'{player_name} FG%')
    ax1.plot(league_fg_percentage.index, league_fg_percentage.values, linestyle='--', color='blue', label="League Mean FG%")
    ax1.set_xlabel("Distance from Basket (meters)")
    ax1.set_ylabel("Field Goal Percentage (%)")
    ax1.set_title(f"Field Goal Percentage by Distance for {player_name}")
    ax1.legend(loc='upper left')

    # Plot the frequency as bars
    ax2 = ax1.twinx()
    ax2.bar(player_fg_percentage.index, player_shot_counts, color='gray', alpha=0.3, width=0.5, label='Frequency of Shots')
    ax2.set_ylabel("Frequency of Shots")

    fig.tight_layout()
    st.pyplot(fig)
	    
from scipy.interpolate import UnivariateSpline

def calculate_fg_percentage_by_distance(df_shots, bin_size=1, window_size=5):
    df_shots["distance"] = df_shots.apply(lambda row: calculate_distance_from_basket(row["x_coord"], row["y_coord"]), axis=1)
    df_shots["distance"] = df_shots["distance"].apply(convert_units_to_meters)

    bins = np.arange(0, df_shots["distance"].max() + bin_size, bin_size)
    df_shots["distance_bin"] = pd.cut(df_shots["distance"], bins, right=False)

    fg_percentage = df_shots.groupby("distance_bin")["shot_result"].mean() * 100
    fg_percentage.index = fg_percentage.index.categories.left  # Convert index to numeric
    
    shot_counts = df_shots.groupby("distance_bin").size()
    shot_counts.index = shot_counts.index.categories.left  # Convert index to numeric

    # Apply moving average for smoothing FG%
    fg_percentage = fg_percentage.rolling(window=window_size, min_periods=1, center=True).mean()
    
    return fg_percentage, shot_counts

def plot_interpolated_distribution(player_name):
    df_shots = fetch_shot_data(player_name)
    if df_shots.empty:
        st.warning(f"No shot data found for {player_name}.")
        return
    
    df_shots["distance_from_basket_units"] = df_shots.apply(lambda row: calculate_distance_from_basket(row["x_coord"], row["y_coord"]), axis=1)
    df_shots["distance_from_basket_m"] = df_shots["distance_from_basket_units"].apply(convert_units_to_meters)
    
    x_smooth, y_smooth = calculate_interpolated_distribution(df_shots)
    
    # Fetch league shot data and calculate the interpolated distribution
    df_league_shots = fetch_league_shot_data()
    df_league_shots["distance_from_basket_units"] = df_league_shots.apply(lambda row: calculate_distance_from_basket(row["x_coord"], row["y_coord"]), axis=1)
    df_league_shots["distance_from_basket_m"] = df_league_shots["distance_from_basket_units"].apply(convert_units_to_meters)
    x_smooth_league, y_smooth_league = calculate_interpolated_distribution(df_league_shots)
    
    # Plot the percentage distribution for the player and the league mean
    st.subheader(f"Percentage Distribution of Shots for {player_name}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_smooth, y_smooth, label='Shot Distribution')
    ax.plot(x_smooth_league, y_smooth_league, label='League Mean Distribution', linestyle='--')
    ax.set_xlabel("Distance from Basket (meters)")
    ax.set_ylabel("Density")
    ax.set_title("Percentage Distribution of Shots Based on Distance (Interpolated)")
    ax.legend()
    st.pyplot(fig)


def calculate_shot_distribution(df_shots):
    # Create bins for distance
    bins = np.arange(0, df_shots["distance_from_basket_m"].max() + 1, 1)
    df_shots["distance_bin"] = pd.cut(df_shots["distance_from_basket_m"], bins, right=False)

    # Calculate distribution
    distribution = df_shots["distance_bin"].value_counts(normalize=True).sort_index() * 100
    return distribution

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

def plot_shot_distribution(distribution):
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot line graph
    distribution.plot(kind='line', ax=ax)

    # Set labels and title
    ax.set_xlabel("Distance from Basket (meters)")
    ax.set_ylabel("Percentage of Shots (%)")
    ax.set_title("Percentage Distribution of Shots Based on Distance")

    # Display plot
    st.pyplot(fig)

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

def fetch_team_four_factors(team_name):
    if not table_exists("Teams"):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT 
        game_id,
        ROUND((field_goals_made + 0.5 * three_pointers_made) * 100.0 / field_goals_attempted, 2) AS eFG_percentage,
        ROUND(turnovers * 100.0 / (field_goals_attempted + 0.44 * free_throws_attempted), 2) AS TOV_percentage,
        ROUND(rebounds_offensive * 100.0 / (rebounds_offensive + rebounds_defensive), 2) AS ORB_percentage,
        ROUND(free_throws_attempted * 100.0 / field_goals_attempted, 2) AS FTR_percentage
    FROM Teams
    WHERE name = '{team_name}'
    ORDER BY game_id;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def plot_four_factors_stats(team1, team2, selected_stats):
    df_team1 = fetch_team_four_factors(team1)
    df_team2 = fetch_team_four_factors(team2)

    if df_team1.empty or df_team2.empty:
        st.error("One or both teams have no recorded stats.")
        return

    # Apply smoothing using rolling mean
    window_size = 5  # Adjust the window size for more or less smoothing
    df_team1 = df_team1.set_index('game_id').rolling(window=window_size).mean().reset_index()
    df_team2 = df_team2.set_index('game_id').rolling(window=window_size).mean().reset_index()

    fig, ax = plt.subplots(figsize=(12, 8))
    for stat in selected_stats:
        ax.plot(df_team1['game_id'], df_team1[stat], label=f"{team1} - {stat}")
        ax.plot(df_team2['game_id'], df_team2[stat], label=f"{team2} - {stat}", linestyle='--')

    ax.set_xlabel("Game ID")
    ax.set_ylabel("Value")
    ax.set_title(f"{team1} vs {team2} - Four Factors Statistics")
    ax.legend()
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

def plot_team_stats_correlation(df):
    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Correlation Heatmap of Team Statistics")
    st.pyplot(plt)

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

def fetch_team_avg_substitutions():
    conn = sqlite3.connect(db_path)
    query = """
    SELECT t.name AS team_name, AVG(sub_count / 2) AS avg_substitutions
    FROM (
        SELECT game_id, team_id, COUNT(*) AS sub_count
        FROM PlayByPlay
        WHERE action_type = 'substitution'
        GROUP BY game_id, team_id
    ) AS subs
    JOIN Teams t ON subs.team_id = t.tm AND subs.game_id = t.game_id
    GROUP BY t.name
    ORDER BY avg_substitutions DESC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Define SQLite database path
db_path = os.path.join(os.path.dirname(__file__), "database.db")

# Function to check if a table exists
def table_exists(table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def fetch_final_lead_value(game_id, team_id):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT lead
    FROM PlayByPlay
    WHERE game_id = ?
    ORDER BY action_number DESC
    LIMIT 1;
    """
    final_lead_value = pd.read_sql_query(query, conn, params=(game_id,)).squeeze()
    conn.close()
    if team_id == 2:
        final_lead_value = -final_lead_value
    return final_lead_value

def fetch_player_games(player_name):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT p.game_id, p.team_id, p.starter, 
           pbp.sub_type AS substitution, 
           pbp.lead AS lead_value
    FROM Players p
    JOIN PlayByPlay pbp ON pbp.game_id = p.game_id AND pbp.player_id = p.json_player_id AND pbp.team_id = p.team_id
    WHERE p.first_name || ' ' || p.last_name = ? AND pbp.action_type = 'substitution'
    ORDER BY p.game_id ASC, pbp.action_number ASC
    """
    df = pd.read_sql_query(query, conn, params=(player_name,))
    conn.close()

    player_data = {}
    for _, row in df.iterrows():
        player_key = (row['game_id'], row['team_id'], row['starter'])
        player_data.setdefault(player_key, []).append((row['substitution'], -row['lead_value'] if row['team_id'] == 2 else row['lead_value']))

    max_subs = max(len(events) for events in player_data.values()) if player_data else 0

    formatted_data = []
    for key, events in player_data.items():
        row_data = list(key)

        if key[2] == 1:
            row_data.append('in')
            row_data.append(0)

        for event in events:
            row_data.append(event[0])
            row_data.append(event[1])

        if events and events[-1][0] == 'in':
            row_data.append('out')
            row_data.append(events[-1][1])

        while len(row_data) < 3 + (max_subs + 2) * 2:
            row_data.append(None)
        formatted_data.append(row_data)

    columns = ['game_id', 'team_id', 'starter']
    for i in range(max_subs + 2):
        columns.extend([f'substitution_{i+1}', f'lead_value_{i+1}'])

    df_formatted = pd.DataFrame(formatted_data, columns=columns)

    df_formatted['plus_minus_on'] = df_formatted.apply(lambda row: calculate_plus_minus(row, on_court=True), axis=1)
    df_formatted['final_lead_value'] = df_formatted.apply(lambda row: fetch_final_lead_value(row['game_id'], row['team_id']), axis=1)
    df_formatted['plus_minus_off'] = df_formatted['final_lead_value'] - df_formatted['plus_minus_on']

    return df_formatted

def calculate_plus_minus(row, on_court=True):
    plus_minus = 0
    in_game_lead = None
    events = [(row[i], row[i + 1]) for i in range(3, len(row) - 1, 2) if pd.notna(row[i])]

    if on_court:
        for event, lead in events:
            if event == 'in':
                in_game_lead = lead
            elif event == 'out' and in_game_lead is not None:
                plus_minus += lead - in_game_lead
                in_game_lead = None

    return plus_minus

def player_game_summary_page():
    st.title("🏀 Player Game Summary")
    st.write("Select a player to view all games they have played in, whether they were a starter, and their substitutions with lead scores in structured columns. If the player finished the game on the court, the last lead score is recorded.")
    
    player_list = get_player_list()
    selected_player = st.selectbox("Select Player", player_list)
    
    if selected_player:
        df_games = fetch_player_games(selected_player)
        st.write(f"### Games Played by {selected_player}")
        st.dataframe(df_games)
        
        st.write(f"### Plus-Minus for {selected_player}")
        st.dataframe(df_games[['game_id', 'plus_minus_on', 'plus_minus_off', 'final_lead_value']])

def get_player_list():
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT first_name || ' ' || last_name AS player_name FROM Players ORDER BY player_name"
    player_list = pd.read_sql(query, conn)["player_name"].tolist()
    conn.close()
    return player_list


def load_substitutions(game_id, team_id):
    """Fetch all substitutions for a given game and team, including lead values."""
    conn = sqlite3.connect("database.db")
    query = """
    SELECT game_time, period, player_id, sub_type, lead
    FROM PlayByPlay
    WHERE game_id = ? AND team_id = ? AND sub_type IN ('in', 'out')
    ORDER BY game_time
    """
    df = pd.read_sql(query, conn, params=(game_id, team_id))
    conn.close()
    return df

def display_avg_substitutions_graph():
    avg_substitutions = fetch_team_avg_substitutions()
    if not avg_substitutions.empty:
        st.subheader("📊 Average Substitutions Per Game")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=avg_substitutions["team_name"], y=avg_substitutions["avg_substitutions"], ax=ax, palette="coolwarm", order=avg_substitutions["team_name"])
        ax.set_ylabel("Avg Substitutions")
        ax.set_xlabel("Team")
        ax.set_title("Average Substitutions Per Game by Team")
        ax.set_xticklabels(avg_substitutions["team_name"], rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        st.pyplot(fig)

    
def main():
    st.title("🏀 Basketball Stats Viewer")
    page = st.sidebar.selectbox("📌 Choose a page", ["Team Season Boxscore", "Head-to-Head Comparison", "Referee Stats", "Shot Chart", "Four Factors", "Match Detail","Lebron"])

    if page == "Match Detail":
        display_match_detail()

        teams = fetch_teams()
        selected_team = st.selectbox("Select a Team", teams)

        if selected_team:
            games = fetch_team_games(selected_team)
            game_options = [f"{game[1]} (Game ID: {game[0]})" for game in games]
            selected_game = st.selectbox("Select a Game", game_options)

            if selected_game:
                selected_game_id = int(selected_game.split("Game ID: ")[1].replace(")", ""))

                # Fetch starting five players for the selected game
                starting_five = fetch_starting_five(selected_game_id)

                if not starting_five.empty:
                    # Extract home and away teams from the starting five data
                    team_ids = starting_five['team_id'].unique()
                    home_team = team_ids[0]
                    away_team = team_ids[1] if len(team_ids) > 1 else "Unknown"

                    st.subheader("Starting Five Players")

                    st.write(f"**{home_team}**")
                    home_starting_five = starting_five[starting_five['team_id'] == home_team]
                    st.table(home_starting_five[['player_name']])

                    st.write(f"**{away_team}**")
                    away_starting_five = starting_five[starting_five['team_id'] == away_team]
                    st.table(away_starting_five[['player_name']])
                else:
                    st.warning("No starting five players found for the selected game.")

                # Fetch all quarters data correctly
                actions = fetch_pbp_actions(selected_game_id)
                display_pbp_actions(actions)

                # 🏀 Add Full Game Score Progression Chart
                st.subheader(f"📈 Score Lead Progression - Full Game")
                plot_score_lead_full_game(selected_game_id)


    elif page == "Four Factors":
        df = fetch_team_data()
        if df.empty:
            st.warning("No team data available.")
        else:
            st.subheader("📊 Four Factors Statistics (Averages Per Game)")
            four_factors = ['eFG_percentage', 'TOV_percentage', 'ORB_percentage', 'FTR_percentage']
            st.dataframe(df[['Team', 'Location'] + four_factors].style.format({
                'eFG_percentage': "{:.2f}",
                'TOV_percentage': "{:.2f}",
                'ORB_percentage': "{:.2f}",
                'FTR_percentage': "{:.2f}"
            }))
            team_options = df["Team"].unique()
        team1 = st.selectbox("Select Team 1", team_options, key="team1_selectbox")
        team2 = st.selectbox("Select Team 2", team_options, key="team2_selectbox")

        four_factors = ['eFG_percentage', 'TOV_percentage', 'ORB_percentage', 'FTR_percentage']
        selected_stats = st.multiselect("Select statistics to display", four_factors, default=four_factors, key="stats_multiselect")

        if team1 and team2 and selected_stats:
            plot_four_factors_stats(team1, team2, selected_stats)

        if team1:
            st.subheader(f"📝 Detailed Matches for {team1}")
            df_matches = fetch_team_matches(team1)
            if not df_matches.empty:
                st.dataframe(df_matches)
            else:
                st.warning(f"No match data available for {team1}.")
    
    elif page == "Lebron":
        # Run the function to display the page
        player_game_summary_page()
        
    elif page == "Team Season Boxscore":
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
                    'Avg_Assists', 'Avg_Rebounds', 'Avg_Steals', 'Avg_Turnovers', 'Avg_Blocks',
                    'eFG', 'TOV', 'ORB', 'FTR'
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

            display_avg_substitutions_graph()

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
            # Display shot data with distance
            display_shot_data_with_distance(player_name)

            # Calculate and plot interpolated distribution
            df_shots_with_distance = fetch_shot_data(player_name)
            if not df_shots_with_distance.empty:
                df_shots_with_distance["distance_from_basket_units"] = df_shots_with_distance.apply(lambda row: calculate_distance_from_basket(row["x_coord"], row["y_coord"]), axis=1)
                df_shots_with_distance["distance_from_basket_m"] = df_shots_with_distance["distance_from_basket_units"].apply(convert_units_to_meters)
                x_smooth, y_smooth = calculate_interpolated_distribution(df_shots_with_distance)
                # Call the plot_interpolated_distribution function without columns
                plot_interpolated_distribution(player_name)

            # Plot FG percentage with frequency
            plot_fg_percentage_with_frequency(player_name)

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
                    "FG%": "{:.1f}%",
                    "3P%": "{:.1f}%",
                    "2P%": "{:.1f}%",
                    "FT%": "{:.1f}%",
                    "PPS": "{:.1f}",
                    "PTS": "{:.1f}",
                    "REB": "{:.1f}",
                    "AST": "{:.1f}",
                    "STL": "{:.1f}",
                    "BLK": "{:.1f}",
                    "TO": "{:.1f}"
                }))
            else:
                st.warning(f"No game-by-game stats available for {player_name}.")

            player_vs_league_40 = fetch_player_and_league_stats_per_40(player_name)
            if not player_vs_league_40.empty:
                st.subheader(f"📊 {player_name} vs. League - Stats per 40 Minutes")
                st.dataframe(player_vs_league_40.style.format({
                    "PTS": "{:.1f}",
                    "REB": "{:.1f}",
                    "AST": "{:.1f}",
                    "STL": "{:.1f}",
                    "BLK": "{:.1f}",
                    "TO": "{:.1f}",
                    "FGA": "{:.1f}",
                    "PPS": "{:.2f}"
                }))
            else:
                st.warning(f"No per-40 stats available for {player_name}.")
if __name__ == "__main__":
    main()
