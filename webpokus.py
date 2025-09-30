
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
import plotly.graph_objects as go
from datetime import datetime
import os
from collections import defaultdict
from itertools import combinations

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
    
def calculate_fg_percentage_by_distance(df_shots, bin_size=1, window_size=5):
    """Calculate PPS (Points Per Shot) by distance with 3-point line consideration"""
    # Calculate distances
    df_shots["distance"] = df_shots.apply(
        lambda row: calculate_distance_from_basket(row["x_coord"], row["y_coord"]), 
        axis=1
    )
    df_shots["distance"] = df_shots["distance"].apply(convert_units_to_meters)

    # Function to determine if a shot is a 3-pointer based on court position
    def is_three_pointer(x, y):
        distance = np.sqrt((x - 50)**2 + (y - 25)**2)
        return distance >= 23.75  # NBA 3-point line distance in feet

    # Add shot type classification
    df_shots["is_three"] = df_shots.apply(lambda row: is_three_pointer(row["x_coord"], row["y_coord"]), axis=1)
    
    # Calculate points for each shot (2 or 3 points)
    df_shots["points"] = df_shots.apply(lambda row: 3 if row["is_three"] and row["shot_result"] == 1 
                                      else 2 if row["shot_result"] == 1 else 0, axis=1)

    # Create shot zone classification
    def classify_shot_zone(row):
        distance = row["distance"]
        is_three = row["is_three"]
        
        if distance <= 1:
            return 'At Rim'
        elif distance <= 3:
            return 'Close Range'
        elif distance <= 5:
            return 'Mid Range'
        elif not is_three:
            return 'Long Mid'
        elif distance <= 8:
            return 'Three Point'
        else:
            return 'Deep Three'

    # Apply shot zone classification
    df_shots["distance_bin"] = df_shots.apply(classify_shot_zone, axis=1)

    # Calculate PPS and counts
    grouped = df_shots.groupby("distance_bin")
    pps = grouped["points"].sum() / grouped.size()
    shot_counts = grouped.size()

    # Define the desired order of zones
    zone_order = ['At Rim', 'Close Range', 'Mid Range', 'Long Mid', 'Three Point', 'Deep Three']
    
    # Reindex based on the desired order, filling any missing categories with 0
    pps = pps.reindex(zone_order, fill_value=0)
    shot_counts = shot_counts.reindex(zone_order, fill_value=0)

    # Apply smoothing
    pps = pps.rolling(window=window_size, min_periods=1, center=True).mean()
    
    return pps, shot_counts

def plot_fg_percentage_with_frequency(player_name, window_size=5):
    """Plot PPS and shot frequency by distance"""
    df_player_shots = fetch_shot_data(player_name)
    df_league_shots = fetch_league_shot_data()

    if df_player_shots.empty:
        st.warning(f"No shot data found for {player_name}.")
        return

    # Calculate PPS and counts
    player_pps, player_shot_counts = calculate_fg_percentage_by_distance(df_player_shots, window_size=window_size)
    league_pps, league_shot_counts = calculate_fg_percentage_by_distance(df_league_shots, window_size=window_size)

    # Create figure
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot PPS
    ax1.plot(range(len(player_pps)), player_pps.values, 
             color='green', label=f'{player_name} PPS', linewidth=2)
    ax1.plot(range(len(league_pps)), league_pps.values, 
             linestyle='--', color='blue', label="League PPS", linewidth=2)

    # Set y-axis limits
    ax1.set_ylim(0, max(3, player_pps.max() * 1.1))

    ax1.set_xlabel("Shot Zone")
    ax1.set_ylabel("Points Per Shot (PPS)")
    ax1.set_title(f"Shot Distribution Analysis for {player_name}")

    # Plot frequency as bars
    ax2 = ax1.twinx()
    bars = ax2.bar(range(len(player_shot_counts)), player_shot_counts.values, 
            color='gray', alpha=0.3, width=0.5, label='Shot Frequency')
    ax2.set_ylabel("Number of Shots")

    # Set x-ticks to show distance labels
    ax1.set_xticks(range(len(player_pps)))
    ax1.set_xticklabels(player_pps.index, rotation=45)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # Add grid for better readability
    ax1.grid(True, alpha=0.3)

    # Add efficiency metrics
    shot_efficiency = pd.DataFrame({
        'Zone': player_pps.index,
        'PPS': player_pps.values,
        'Shots': player_shot_counts.values,
        'Shot Distribution %': (player_shot_counts.values / player_shot_counts.sum() * 100)
    })
    
    fig.tight_layout()
    st.pyplot(fig)

    # Display efficiency table
    st.write("### Shot Efficiency Breakdown")
    formatted_efficiency = shot_efficiency.copy()
    formatted_efficiency['PPS'] = formatted_efficiency['PPS'].round(2)
    formatted_efficiency['Shot Distribution %'] = formatted_efficiency['Shot Distribution %'].round(1).astype(str) + '%'
    st.dataframe(formatted_efficiency.set_index('Zone'))

    # Calculate and display overall metrics
    total_points = df_player_shots['points'].sum()
    total_shots = len(df_player_shots)
    overall_pps = total_points / total_shots if total_shots > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall PPS", f"{overall_pps:.2f}")
    with col2:
        st.metric("Total Points", f"{total_points}")
    with col3:
        st.metric("Total Shots", f"{total_shots}")

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
        # Instead of using game_id, create sequential numbers
        match_numbers1 = range(1, len(df_team1) + 1)
        match_numbers2 = range(1, len(df_team2) + 1)
        
        ax.plot(match_numbers1, df_team1[stat], label=f"{team1} - {stat}")
        ax.plot(match_numbers2, df_team2[stat], label=f"{team2} - {stat}", linestyle='--')

    ax.set_xlabel("Match Number")
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

def generate_shot_chart(player_name, show_heatmap=False, shot_types=None):
    """Generate a shot chart with optional heatmap and shot type filtering."""
    if not os.path.exists("fiba_courtonly.jpg"):
        st.error("⚠️ Court image file 'fiba_courtonly.jpg' is missing!")
        return

    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        x_coord, 
        y_coord, 
        shot_result,
        CASE 
            WHEN action_type = '3pt' THEN '3PT'
            ELSE '2PT'
        END as shot_type
    FROM Shots 
    WHERE player_name = ?
    """
    df_shots = pd.read_sql_query(query, conn, params=(player_name,))
    conn.close()

    if df_shots.empty:
        st.warning(f"No shot data found for {player_name}.")
        return

    # Filter by shot type if specified
    if shot_types and "All" not in shot_types:
        df_shots = df_shots[df_shots['shot_type'].isin(shot_types)]

    # Scale coordinates
    df_shots['x_coord'] = df_shots['x_coord'] * 2.8
    df_shots['y_coord'] =(df_shots['y_coord'] * 2.61)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw court
    court_img = mpimg.imread("fiba_courtonly.jpg")
    ax.imshow(court_img, extent=[0, 280, 0, 261], aspect="auto")

    # Add heatmap if requested
    if show_heatmap:
        sns.kdeplot(
            data=df_shots, 
            x="x_coord", y="y_coord", 
            cmap="YlOrRd", fill=True, alpha=0.5,
            ax=ax, bw_adjust=0.5,
            clip=[[0, 280], [0, 261]]
        )

    # Plot shots
    made_shots = df_shots[df_shots['shot_result'] == 1]
    missed_shots = df_shots[df_shots['shot_result'] == 0]

    # Made shots (green circles)
    ax.scatter(made_shots['x_coord'], made_shots['y_coord'],
              marker='o', c='green', s=50, alpha=0.7,
              label='Made Shots')

    # Missed shots (red X's)
    ax.scatter(missed_shots['x_coord'], missed_shots['y_coord'],
              marker='x', c='red', s=50, alpha=0.7,
              label='Missed Shots')

    # Add legend
    ax.legend(loc='upper right')

    # Add shot statistics
    total_shots = len(df_shots)
    made_shots_count = len(made_shots)
    fg_percentage = (made_shots_count / total_shots * 100) if total_shots > 0 else 0
    
    ax.set_title(f"{player_name}'s Shot Chart\n"
                 f"FG: {made_shots_count}/{total_shots} ({fg_percentage:.1f}%)")

    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    st.pyplot(fig)

    # Display shot statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Shots", total_shots)
    with col2:
        st.metric("Made Shots", made_shots_count)
    with col3:
        st.metric("Field Goal %", f"{fg_percentage:.1f}%")

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

import pandas as pd
import sqlite3
import streamlit as st
import os

db_path = os.path.join(os.path.dirname(__file__), "database.db")

def fetch_player_games(player_name):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT p.game_id, p.team_id, p.starter,
           pbp.sub_type AS substitution,
           pbp.lead AS lead_value,
           pbp.current_score_team1,
           pbp.current_score_team2,
           pbp.action_type,
           pbp.sub_type,
           pbp.team_id AS pbp_team_id,
           pbp.action_number,
           p.minutes_played,
           p.points AS PTS,
           p.assists AS AST,
           p.three_pointers_made AS threePM,
           p.three_pointers_attempted AS threePA,
           p.free_throws_attempted AS FTA,
           p.rebounds_offensive AS ORB,
           p.steals AS STL,
           p.blocks AS BLK,
           p.rebounds_defensive AS DREB
    FROM Players p
    JOIN PlayByPlay pbp ON pbp.game_id = p.game_id AND pbp.player_id = p.json_player_id AND pbp.team_id = p.team_id
    WHERE p.first_name || ' ' || p.last_name = ?
      AND pbp.action_type IN ('substitution', '2pt', '3pt', 'rebound', 'turnover', 'freethrow')
    ORDER BY p.game_id ASC, pbp.action_number ASC
    """
    df = pd.read_sql_query(query, conn, params=(player_name,))
    conn.close()

    player_data = {}
    for _, row in df.iterrows():
        player_key = (row['game_id'], row['team_id'], row['starter'], row['minutes_played'], row['PTS'], row['AST'], row['threePM'], row['threePA'], row['FTA'], row['ORB'], row['STL'], row['BLK'], row['DREB'])
        player_data.setdefault(player_key, []).append((row['substitution'], row['lead_value'], row['current_score_team1'], row['current_score_team2'], row['action_type'], row['sub_type'], row['pbp_team_id'], row['action_number']))

    max_subs = max(len(events) for events in player_data.values()) if player_data else 0

    formatted_data = []
    for key, events in player_data.items():
        row_data = list(key[:3])  # exclude minutes_played and other stats from key
        minutes_played, PTS, AST, threePM, threePA, FTA, ORB, STL, BLK, DREB = key[3:]

        if isinstance(minutes_played, str) and ':' in minutes_played:
            mm, ss = map(int, minutes_played.split(':'))
            minutes_played = mm + ss / 60  # Convert MM:SS to total minutes
        else:
            minutes_played = float(minutes_played)

        if key[2] == 1:
            row_data.append('in')
            row_data.append(0)
            row_data.append(0)
            row_data.append(0)
            row_data.append(None)
            row_data.append(None)
            row_data.append(None)
            row_data.append(1)

        for event in events:
            lead_value = -event[1] if key[1] == 2 else event[1]  # Invert lead value if team_id is 2
            row_data.extend((event[0], lead_value, event[2], event[3], event[4], event[5], event[6], event[7]))

        if events and events[-1][0] == 'in':
            last_action_number = fetch_last_action_number(events[-1][0])
            lead_value = -events[-1][1] if key[1] == 2 else events[-1][1]  # Invert lead value if team_id is 2
            row_data.append('out')
            row_data.append(lead_value)
            row_data.append(events[-1][2])
            row_data.append(events[-1][3])
            row_data.append(None)
            row_data.append(None)
            row_data.append(None)
            row_data.append(last_action_number)

        while len(row_data) < 3 + (max_subs + 2) * 8:
            row_data.append(None)
        row_data.extend([minutes_played, PTS, AST, threePM, threePA, FTA, ORB, STL, BLK, DREB])  # Adding minutes_played and other stats

        # Calculate weight factor
        weight_factor = 100 / (minutes_played + 100)
        row_data.append(weight_factor)  # Adding weight factor at the end

        formatted_data.append(row_data)

    columns = ['game_id', 'team_id', 'starter']
    for i in range(max_subs + 2):
        columns.extend([f'substitution_{i+1}', f'lead_value_{i+1}', f'current_score_team1_{i+1}', f'current_score_team2_{i+1}', f'action_type_{i+1}', f'sub_type_{i+1}', f'pbp_team_id_{i+1}', f'action_number_{i+1}'])
    columns.extend(['minutes_played', 'PTS', 'AST', 'threePM', 'threePA', 'FTA', 'ORB', 'STL', 'BLK', 'DREB', 'weight_factor'])  # Adding weight_factor and player stats columns

    # Ensure all rows in formatted_data have the same length as columns
    for row in formatted_data:
        while len(row) < len(columns):
            row.append(None)
        while len(row) > len(columns):
            row.pop()

    df_formatted = pd.DataFrame(formatted_data, columns=columns)

    # Calculate other statistics
    df_formatted['plus_minus_on'] = df_formatted.apply(lambda row: calculate_plus_minus(row, on_court=True), axis=1)
    df_formatted['final_lead_value'] = df_formatted.apply(lambda row: fetch_final_lead_value(row['game_id'], row['team_id']), axis=1)
    df_formatted['plus_minus_off'] = df_formatted['final_lead_value'] - df_formatted['plus_minus_on']
    df_formatted['points_allowed_on'] = df_formatted.apply(lambda row: calculate_points_allowed(row), axis=1)
    df_formatted['efga_by_opponent'] = df_formatted.apply(lambda row: calculate_efga_by_opponent(row), axis=1)
    df_formatted['offensive_rebounds_by_opponent'] = df_formatted.apply(lambda row: calculate_offensive_rebounds_by_opponent(row), axis=1)
    df_formatted['turnovers_by_opponent'] = df_formatted.apply(lambda row: calculate_turnovers_by_opponent(row), axis=1)
    df_formatted['freethrows_by_opponent'] = df_formatted.apply(lambda row: calculate_freethrows_by_opponent(row), axis=1)
    df_formatted['possessions_by_opponent'] = df_formatted.apply(lambda row: calculate_possessions(row), axis=1)
    df_formatted['defensive_rating'] = df_formatted.apply(lambda row: calculate_defensive_rating(row), axis=1)

    # Calculate PM league average
    pm_league_avg = df_formatted['plus_minus_on'].mean()

    # Calculate PM_bias
    df_formatted['PM_bias'] = (df_formatted['plus_minus_on'] + df_formatted['weight_factor'] * pm_league_avg) / (1 + df_formatted['weight_factor'])

    # Update oLEBRON and dLEBRON calculations to use PM_bias
    df_formatted['oLEBRON'] = (0.80 * df_formatted['PTS']) + (0.75 * df_formatted['AST']) + (0.60 * df_formatted['threePM']) + (0.45 * df_formatted['threePA']) + (0.55 * df_formatted['FTA']) + (0.40 * df_formatted['ORB']) + (0.50 * df_formatted['PM_bias'])
    df_formatted['dLEBRON'] = (0.85 * df_formatted['STL']) + (0.70 * df_formatted['BLK']) + (0.50 * df_formatted['DREB']) + (0.40 * df_formatted['defensive_rating']) + (0.55 * df_formatted['PM_bias'])

    # Calculate standard LEBRON with equal weights
    df_formatted['LEBRON_total'] = 0.7 * df_formatted['oLEBRON'] + 0.3 * df_formatted['dLEBRON']

    return df_formatted

def fetch_last_action_number(game_id):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT MAX(action_number) AS last_action_number
    FROM PlayByPlay
    WHERE game_id = ?
    """
    last_action_number = pd.read_sql_query(query, conn, params=(game_id,)).squeeze()
    conn.close()
    return last_action_number

def fetch_final_scores(game_id):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT current_score_team1, current_score_team2
    FROM PlayByPlay
    WHERE game_id = ?
    ORDER BY action_number DESC
    LIMIT 1;
    """
    scores = pd.read_sql_query(query, conn, params=(game_id,)).squeeze()
    conn.close()
    return scores

def calculate_plus_minus(row, on_court=True):
    plus_minus = 0
    in_game_lead = None
    events = [(row[i], row[i + 1]) for i in range(3, len(row) - 2, 8) if pd.notna(row[i])]

    if on_court:
        for event, lead in events:
            if event == 'in':
                in_game_lead = lead
            elif event == 'out' and in_game_lead is not None:
                plus_minus += lead - in_game_lead
                in_game_lead = None

    return plus_minus

def calculate_points_allowed(row):
    points_allowed = 0
    in_game_score = None
    max_index = len(row) - 1
    events = [(row[i], row[i + 1], row[i + 2], row[i + 3], row[i + 7]) for i in range(3, max_index - 7, 8) if pd.notna(row[i])]

    for event, lead, score1, score2, action_number in events:
        if event == 'in':
            in_game_score = score1 if row['team_id'] == 2 else score2
        elif event == 'out' and in_game_score is not None:
            current_score = score1 if row['team_id'] == 2 else score2
            points_allowed += current_score - in_game_score
            in_game_score = None

    if in_game_score is not None:  # If player is still in the game till the end
        final_scores = fetch_final_scores(row['game_id'])
        current_score = final_scores['current_score_team1'] if row['team_id'] == 2 else final_scores['current_score_team2']
        points_allowed += current_score - in_game_score

    return points_allowed

def calculate_efga_by_opponent(row):
    efga_by_opponent = 0
    in_game = False
    action_start = None
    max_index = len(row) - 1
    events = [(row[i], row[i + 4], row[i + 5], row[i + 6], row[i + 7]) for i in range(3, max_index - 7, 8) if pd.notna(row[i])]

    for event, action_type, sub_type, pbp_team_id, action_number in events:
        if event == 'in':
            in_game = True
            action_start = action_number
        elif event == 'out':
            in_game = False
            action_end = action_number
            efga_by_opponent += count_efga_between(row['game_id'], action_start, action_end, row['team_id'])
        elif in_game and action_type in ['2pt', '3pt'] and pbp_team_id != row['team_id']:
            if action_type == '2pt':
                efga_by_opponent += 1
            elif action_type == '3pt':
                efga_by_opponent += 0.5

    if in_game:  # If player is still in the game till the end
        action_end = fetch_last_action_number(row['game_id'])
        efga_by_opponent += count_efga_between(row['game_id'], action_start, action_end, row['team_id'])

    return efga_by_opponent

def count_efga_between(game_id, start_action, end_action, team_id):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT SUM(CASE WHEN action_type = '2pt' THEN 1 ELSE 0.5 END) AS efga_sum
    FROM PlayByPlay
    WHERE game_id = ? AND action_number BETWEEN ? AND ? AND action_type IN ('2pt', '3pt') AND team_id != ?
    """
    efga_sum = pd.read_sql_query(query, conn, params=(game_id, start_action, end_action, team_id)).squeeze()
    conn.close()
    return efga_sum if efga_sum is not None else 0

def calculate_offensive_rebounds_by_opponent(row):
    offensive_rebounds_by_opponent = 0
    in_game = False
    action_start = None
    max_index = len(row) - 1
    events = [(row[i], row[i + 4], row[i + 5], row[i + 6], row[i + 7]) for i in range(3, max_index - 7, 8) if pd.notna(row[i])]

    for event, action_type, sub_type, pbp_team_id, action_number in events:
        if event == 'in':
            in_game = True
            action_start = action_number
        elif event == 'out':
            in_game = False
            action_end = action_number
            offensive_rebounds_by_opponent += count_offensive_rebounds_between(row['game_id'], action_start, action_end, row['team_id'])
        elif in_game and action_type == 'rebound' and sub_type == 'offensive' and pbp_team_id != row['team_id']:
            offensive_rebounds_by_opponent += 1

    if in_game:  # If player is still in the game till the end
        action_end = fetch_last_action_number(row['game_id'])
        offensive_rebounds_by_opponent += count_offensive_rebounds_between(row['game_id'], action_start, action_end, row['team_id'])

    return offensive_rebounds_by_opponent

def count_offensive_rebounds_between(game_id, start_action, end_action, team_id):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT COUNT(*)
    FROM PlayByPlay
    WHERE game_id = ? AND action_number BETWEEN ? AND ? AND action_type = 'rebound' AND sub_type = 'offensive' AND team_id != ?
    """
    count = pd.read_sql_query(query, conn, params=(game_id, start_action, end_action, team_id)).squeeze()
    conn.close()
    return count

def calculate_turnovers_by_opponent(row):
    turnovers_by_opponent = 0
    in_game = False
    action_start = None
    max_index = len(row) - 1
    events = [(row[i], row[i + 4], row[i + 5], row[i + 6], row[i + 7]) for i in range(3, max_index - 7, 8) if pd.notna(row[i])]

    for event, action_type, sub_type, pbp_team_id, action_number in events:
        if event == 'in':
            in_game = True
            action_start = action_number
        elif event == 'out':
            in_game = False
            action_end = action_number
            turnovers_by_opponent += count_turnovers_between(row['game_id'], action_start, action_end, row['team_id'])
        elif in_game and action_type == 'turnover' and pbp_team_id != row['team_id']:
            turnovers_by_opponent += 1

    if in_game:  # If player is still in the game till the end
        action_end = fetch_last_action_number(row['game_id'])
        turnovers_by_opponent += count_turnovers_between(row['game_id'], action_start, action_end, row['team_id'])

    return turnovers_by_opponent

def count_turnovers_between(game_id, start_action, end_action, team_id):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT COUNT(*)
    FROM PlayByPlay
    WHERE game_id = ? AND action_number BETWEEN ? AND ? AND action_type = 'turnover' AND team_id != ?
    """
    count = pd.read_sql_query(query, conn, params=(game_id, start_action, end_action, team_id)).squeeze()
    conn.close()
    return count

def calculate_freethrows_by_opponent(row):
    freethrows_by_opponent = 0
    in_game = False
    action_start = None
    max_index = len(row) - 1
    events = [(row[i], row[i + 4], row[i + 5], row[i + 6], row[i + 7]) for i in range(3, max_index - 7, 8) if pd.notna(row[i])]

    for event, action_type, sub_type, pbp_team_id, action_number in events:
        if event == 'in':
            in_game = True
            action_start = action_number
        elif event == 'out':
            in_game = False
            action_end = action_number
            freethrows_by_opponent += count_freethrows_between(row['game_id'], action_start, action_end, row['team_id'])
        elif in_game and action_type == 'freethrow' and pbp_team_id != row['team_id']:
            freethrows_by_opponent += 1

    if in_game:  # If player is still in the game till the end
        action_end = fetch_last_action_number(row['game_id'])
        freethrows_by_opponent += count_freethrows_between(row['game_id'], action_start, action_end, row['team_id'])

    return freethrows_by_opponent

def count_freethrows_between(game_id, start_action, end_action, team_id):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT COUNT(*)
    FROM PlayByPlay
    WHERE game_id = ? AND action_number BETWEEN ? AND ? AND action_type = 'freethrow' AND team_id != ?
    """
    count = pd.read_sql_query(query, conn, params=(game_id, start_action, end_action, team_id)).squeeze()
    conn.close()
    return count

def calculate_possessions(row):
    fga = row['efga_by_opponent'] * 2  # Convert effective FGA back to total FGA
    orb = row['offensive_rebounds_by_opponent']
    tov = row['turnovers_by_opponent']
    fta = row['freethrows_by_opponent']
    possessions = fga - orb + tov + (0.44 * fta)
    return possessions

def calculate_defensive_rating(row):
    if row['possessions_by_opponent'] == 0:
        return 0
    defensive_rating = 100 * (row['points_allowed_on'] / row['possessions_by_opponent'])
    return defensive_rating

def fetch_final_lead_value(game_id, team_id):
    conn = sqlite3.connect(db_path)
    query = """
    SELECT lead
    FROM PlayByPlay
    WHERE game_id = ?
    ORDER BY action_number DESC
    LIMIT 1;
    """
    cursor = conn.cursor()
    cursor.execute(query, (game_id,))
    result = cursor.fetchone()
    conn.close()
    
    # Get the lead value from the result tuple and adjust based on team_id
    final_lead_value = result[0] if result else 0
    if team_id == 2:
        final_lead_value = -final_lead_value
    return final_lead_value

def calculate_weighted_lebron_for_player(player_name, df_games):
    # Convert minutes_played to numeric if it's in MM:SS format
    def convert_minutes(time_str):
        if pd.isna(time_str):
            return 0
        if isinstance(time_str, str) and ':' in time_str:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes + seconds/60
        return float(time_str)
    
    df_games['minutes_numeric'] = df_games['minutes_played'].apply(convert_minutes)
    total_minutes = df_games['minutes_numeric'].sum()
    
    if total_minutes == 0:
        return {
            'Player': player_name,
            'Weighted oLEBRON': 0,
            'Weighted dLEBRON': 0,
            'Weighted Total LEBRON': 0,
            'Total Minutes': 0
        }
    
    # Calculate weighted averages
    weighted_oLEBRON = (df_games['oLEBRON'] * df_games['minutes_numeric']).sum() / total_minutes
    weighted_dLEBRON = (df_games['dLEBRON'] * df_games['minutes_numeric']).sum() / total_minutes
    weighted_LEBRON_total = (df_games['LEBRON_total'] * df_games['minutes_numeric']).sum() / total_minutes
    
    return {
        'Player': player_name,
        'Weighted oLEBRON': weighted_oLEBRON,
        'Weighted dLEBRON': weighted_dLEBRON,
        'Weighted Total LEBRON': weighted_LEBRON_total,
        'Total Minutes': total_minutes
    }

def calculate_all_players_lebron():
    """Calculate LEBRON stats for all players in a single database query"""
    conn = sqlite3.connect(db_path)
    
    query = """
    WITH PlayerMinutes AS (
        SELECT 
            p.first_name || ' ' || p.last_name AS player_name,
            p.game_id,
            p.team_id,
            CAST(SUBSTR(p.minutes_played, 1, INSTR(p.minutes_played, ':') - 1) AS INTEGER) * 60 + 
            CAST(SUBSTR(p.minutes_played, INSTR(p.minutes_played, ':') + 1) AS INTEGER) AS minutes_numeric,
            p.points AS PTS,
            p.assists AS AST,
            p.three_pointers_made AS threePM,
            p.three_pointers_attempted AS threePA,
            p.free_throws_attempted AS FTA,
            p.rebounds_offensive AS ORB,
            p.steals AS STL,
            p.blocks AS BLK,
            p.rebounds_defensive AS DREB,
            p.defensive_rating
        FROM Players p
        WHERE p.minutes_played != '0:00'
    )
    SELECT 
        player_name,
        SUM(minutes_numeric) as total_minutes,
        SUM(minutes_numeric * ((0.80 * PTS) + (0.75 * AST) + (0.60 * threePM) + 
            (0.45 * threePA) + (0.55 * FTA) + (0.40 * ORB))) / SUM(minutes_numeric) as weighted_oLEBRON,
        SUM(minutes_numeric * ((0.85 * STL) + (0.70 * BLK) + (0.50 * DREB) + 
            (0.40 * defensive_rating))) / SUM(minutes_numeric) as weighted_dLEBRON
    FROM PlayerMinutes
    GROUP BY player_name
    HAVING total_minutes > 0
    ORDER BY (weighted_oLEBRON + weighted_dLEBRON) DESC
    """
    
    df_stats = pd.read_sql_query(query, conn)
    conn.close()
    
    # Calculate total LEBRON
    df_stats['weighted_total_LEBRON'] = (df_stats['weighted_oLEBRON'] + df_stats['weighted_dLEBRON']) / 2
    
    return df_stats

def fetch_team_players(team_name):
    """Fetch all players from a specific team."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT DISTINCT first_name || ' ' || last_name AS player_name
    FROM Players p
    JOIN Teams t ON p.game_id = t.game_id AND p.team_id = t.tm
    WHERE t.name = ?
    """
    players = pd.read_sql_query(query, conn, params=(team_name,))["player_name"].tolist()
    conn.close()
    return players

def player_game_summary_page():
    st.title("🏀 Player Game Summary")
    
    # Helper function for converting minutes
    def convert_minutes(time_str):
        if pd.isna(time_str):
            return 0
        if isinstance(time_str, str) and ':' in time_str:
            minutes, seconds = map(int, time_str.split(':'))
            return minutes + seconds/60
        return float(time_str)
    
    # Get lists of players and teams
    player_list = get_player_list()
    team_list = fetch_teams()
    
    # Create columns for layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("### Select Players")
        
        # Add radio buttons for selection type
        selection_type = st.radio("Select by:", ["Individual Players", "Team Players"])
        
        selected_players = []
        if selection_type == "Individual Players":
            # Use multiselect for individual player selection
            selected_players = st.multiselect(
                "Choose players to compare:",
                options=player_list,
                default=[player_list[0]] if player_list else None
            )
        else:  # Team Players
            # Select team to load its players
            selected_team = st.selectbox(
                "Choose team:",
                options=team_list,
                index=0 if team_list else None
            )
            
            if selected_team:
                # Get all players from the selected team
                team_players = fetch_team_players(selected_team)
                if team_players:
                    st.write(f"#### Players from {selected_team}:")
                    selected_players = st.multiselect(
                        "Select players to compare:",
                        options=team_players,
                        default=team_players  # By default select all team players
                    )
                else:
                    st.warning(f"No players found for {selected_team}")
    
    if selected_players:
        # Container for all selected stats
        all_stats = []
        
        # Fetch and calculate stats for selected players
        for player in selected_players:
            df_games = fetch_player_games(player)
            
            df_games['minutes_numeric'] = df_games['minutes_played'].apply(convert_minutes)
            total_minutes = df_games['minutes_numeric'].sum()
            
            # Calculate weighted averages
            if total_minutes > 0:
                weighted_oLEBRON = (df_games['oLEBRON'] * df_games['minutes_numeric']).sum() / total_minutes
                weighted_dLEBRON = (df_games['dLEBRON'] * df_games['minutes_numeric']).sum() / total_minutes
                weighted_LEBRON_total = (df_games['LEBRON_total'] * df_games['minutes_numeric']).sum() / total_minutes
            else:
                weighted_oLEBRON = weighted_dLEBRON = weighted_LEBRON_total = 0
            
            # Add to stats collection
            all_stats.append({
                'Player': player,
                'Team': selected_team if selection_type == "Team Players" else "Various",
                'Minutes': total_minutes,
                'oLEBRON': weighted_oLEBRON,
                'dLEBRON': weighted_dLEBRON,
                'Total LEBRON': weighted_LEBRON_total
            })
        
        # Create DataFrame with all selected stats
        df_stats = pd.DataFrame(all_stats)
        
        with col2:
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot points
            scatter = ax.scatter(df_stats['dLEBRON'], 
                               df_stats['oLEBRON'],
                               s=100)
            
            # Add labels for each point
            for idx, row in df_stats.iterrows():
                ax.annotate(row['Player'], 
                          (row['dLEBRON'], row['oLEBRON']),
                          xytext=(5, 5), 
                          textcoords='offset points')
            
            # Add quadrant lines
            ax.axhline(y=df_stats['oLEBRON'].mean(), color='gray', linestyle='--', alpha=0.3)
            ax.axvline(x=df_stats['dLEBRON'].mean(), color='gray', linestyle='--', alpha=0.3)
            
            # Labels and title
            ax.set_xlabel('Defensive LEBRON (dLEBRON)')
            ax.set_ylabel('Offensive LEBRON (oLEBRON)')
            title = f"Player Comparison: Offensive vs Defensive LEBRON"
            if selection_type == "Team Players":
                title += f"\n{selected_team} Players"
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Show plot
            st.pyplot(fig)
            
            # Show stats table
            st.write("### Selected Players' Statistics")
            display_df = df_stats.round(3).set_index('Player')
            st.dataframe(display_df)
            
            # Add analysis for team players
            if selection_type == "Team Players":
                st.write(f"### {selected_team} Team Analysis")
                team_avg_o = df_stats['oLEBRON'].mean()
                team_avg_d = df_stats['dLEBRON'].mean()
                st.write(f"Team Average oLEBRON: {team_avg_o:.2f}")
                st.write(f"Team Average dLEBRON: {team_avg_d:.2f}")
                st.write(f"Team Average Total LEBRON: {((team_avg_o + team_avg_d)/2):.2f}")
    else:
        st.write("Please select players to compare.")

        
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

def create_match_selector():
    """Create an enhanced match selection widget with search and filtering capabilities."""
    st.write("### 🏀 Select Match")
    
    # Get matches data with corrected table names
    conn = sqlite3.connect(db_path)
    
    # Query to get match data
    matches_query = """
    SELECT DISTINCT 
        p.game_id,
        p.period,
        t1.name as team1_name,
        t2.name as team2_name,
        MAX(CASE WHEN t1.tm = 1 THEN t1.points END) as score1,
        MAX(CASE WHEN t2.tm = 2 THEN t2.points END) as score2
    FROM PlayByPlay p
    JOIN Teams t1 ON p.game_id = t1.game_id AND t1.tm = 1
    JOIN Teams t2 ON p.game_id = t2.game_id AND t2.tm = 2
    GROUP BY p.game_id
    ORDER BY p.game_id DESC
    """
    
    try:
        df_matches = pd.read_sql_query(matches_query, conn)
        
        if df_matches.empty:
            st.warning("No matches found in the database.")
            conn.close()
            return None
            
        # Add a formatted display string for each match
        df_matches['display_string'] = (
            df_matches['team1_name'] + " vs " + 
            df_matches['team2_name'] + " (" +
            df_matches['score1'].astype(str) + "-" +
            df_matches['score2'].astype(str) + ")"
        )
        
        # Create filter columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Team filter
            all_teams = sorted(list(set(
                list(df_matches['team1_name'].unique()) + 
                list(df_matches['team2_name'].unique())
            )))
            selected_team = st.selectbox(
                "Filter by Team",
                ["All Teams"] + all_teams,
                key="team_filter"
            )
        
        with col2:
            # Search box
            search_term = st.text_input(
                "Search Matches",
                placeholder="Search by team name...",
                key="match_search"
            ).lower()
        
        # Apply filters
        mask = pd.Series(True, index=df_matches.index)
        
        # Team filter
        if selected_team != "All Teams":
            mask &= (df_matches['team1_name'] == selected_team) | \
                    (df_matches['team2_name'] == selected_team)
        
        # Search filter
        if search_term:
            mask &= df_matches['display_string'].str.lower().str.contains(search_term)
        
        # Apply all filters
        filtered_matches = df_matches[mask]
        
        if filtered_matches.empty:
            st.warning("No matches found with the current filters.")
            conn.close()
            return None
        
        # Create the final dropdown with filtered matches
        selected_match = st.selectbox(
            "Select Match",
            options=filtered_matches['game_id'].tolist(),
            format_func=lambda x: filtered_matches[filtered_matches['game_id'] == x]['display_string'].iloc[0],
            key="match_selector"
        )
        
        # Display match details
        if selected_match:
            match_data = filtered_matches[filtered_matches['game_id'] == selected_match].iloc[0]
            
            # Create an expander for additional match details
            with st.expander("Match Details", expanded=False):
                st.write("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Home Team**")
                    st.write(f"🏠 {match_data['team1_name']}")
                    st.write(f"**Score:** {match_data['score1']}")
                
                with col2:
                    st.write("**Away Team**")
                    st.write(f"🏃 {match_data['team2_name']}")
                    st.write(f"**Score:** {match_data['score2']}")
        
        conn.close()
        return selected_match
    
    except Exception as e:
        st.error("Error loading matches. Please try again.")
        conn.close()
        return None
    
def create_and_populate_team_stats():
    """Create TeamStats table and populate it with data from PlayByPlay"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the TeamStats table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS TeamStats (
        game_id INTEGER,
        team_id INTEGER,
        field_goals_made INTEGER,
        field_goals_attempted INTEGER,
        field_goal_percentage REAL,
        three_pointers_made INTEGER,
        three_pointers_attempted INTEGER,
        three_point_percentage REAL,
        free_throws_made INTEGER,
        free_throws_attempted INTEGER,
        free_throw_percentage REAL,
        rebounds_offensive INTEGER,
        rebounds_defensive INTEGER,
        rebounds_total INTEGER,
        assists INTEGER,
        steals INTEGER,
        blocks INTEGER,
        turnovers INTEGER,
        fouls INTEGER,
        points INTEGER,
        PRIMARY KEY (game_id, team_id)
    )""")

    # Populate with data
    cursor.execute("""
    INSERT OR REPLACE INTO TeamStats
    SELECT 
        game_id,
        team_id,
        SUM(CASE WHEN (action_type = '2pt' OR action_type = '3pt') AND sub_type = 'made' THEN 1 ELSE 0 END),
        SUM(CASE WHEN action_type = '2pt' OR action_type = '3pt' THEN 1 ELSE 0 END),
        ROUND(CAST(SUM(CASE WHEN (action_type = '2pt' OR action_type = '3pt') AND sub_type = 'made' THEN 1 ELSE 0 END) AS FLOAT) / 
              NULLIF(SUM(CASE WHEN action_type = '2pt' OR action_type = '3pt' THEN 1 ELSE 0 END), 0) * 100, 1),
        SUM(CASE WHEN action_type = '3pt' AND sub_type = 'made' THEN 1 ELSE 0 END),
        SUM(CASE WHEN action_type = '3pt' THEN 1 ELSE 0 END),
        ROUND(CAST(SUM(CASE WHEN action_type = '3pt' AND sub_type = 'made' THEN 1 ELSE 0 END) AS FLOAT) / 
              NULLIF(SUM(CASE WHEN action_type = '3pt' THEN 1 ELSE 0 END), 0) * 100, 1),
        SUM(CASE WHEN action_type = 'freethrow' AND sub_type = 'made' THEN 1 ELSE 0 END),
        SUM(CASE WHEN action_type = 'freethrow' THEN 1 ELSE 0 END),
        ROUND(CAST(SUM(CASE WHEN action_type = 'freethrow' AND sub_type = 'made' THEN 1 ELSE 0 END) AS FLOAT) / 
              NULLIF(SUM(CASE WHEN action_type = 'freethrow' THEN 1 ELSE 0 END), 0) * 100, 1),
        SUM(CASE WHEN action_type = 'rebound' AND sub_type = 'offensive' THEN 1 ELSE 0 END),
        SUM(CASE WHEN action_type = 'rebound' AND sub_type = 'defensive' THEN 1 ELSE 0 END),
        SUM(CASE WHEN action_type = 'rebound' THEN 1 ELSE 0 END),
        SUM(CASE WHEN action_type = 'assist' THEN 1 ELSE 0 END),
        SUM(CASE WHEN action_type = 'steal' THEN 1 ELSE 0 END),
        SUM(CASE WHEN action_type = 'block' THEN 1 ELSE 0 END),
        SUM(CASE WHEN action_type = 'turnover' THEN 1 ELSE 0 END),
        SUM(CASE WHEN action_type = 'foul' THEN 1 ELSE 0 END),
        SUM(CASE 
            WHEN action_type = '2pt' AND sub_type = 'made' THEN 2
            WHEN action_type = '3pt' AND sub_type = 'made' THEN 3
            WHEN action_type = 'freethrow' AND sub_type = 'made' THEN 1
            ELSE 0 
        END)
    FROM PlayByPlay
    GROUP BY game_id, team_id
    """)

    conn.commit()
    conn.close()

def display_team_stats(game_id):
    """Display team statistics from the Teams table for a selected game."""
    try:
        conn = sqlite3.connect(db_path)
        
        stats_query = """
        SELECT 
            name,
            tm,
            points,
            field_goals_made as fg,
            field_goals_attempted as fga,
            ROUND(CAST(field_goals_made AS FLOAT) / NULLIF(field_goals_attempted, 0) * 100, 1) as fg_percentage,
            two_pointers_made,
            two_pointers_attempted,
            ROUND(CAST(two_pointers_made AS FLOAT) / NULLIF(two_pointers_attempted, 0) * 100, 1) as two_point_percentage,
            three_pointers_made as threep,
            three_pointers_attempted as threepa,
            ROUND(CAST(three_pointers_made AS FLOAT) / NULLIF(three_pointers_attempted, 0) * 100, 1) as three_percentage,
            free_throws_made as ft,
            free_throws_attempted as fta,
            ROUND(CAST(free_throws_made AS FLOAT) / NULLIF(free_throws_attempted, 0) * 100, 1) as ft_percentage,
            rebounds_offensive as orb,
            rebounds_defensive as drb,
            (rebounds_offensive + rebounds_defensive) as total_rebounds,
            assists as ast,
            steals as stl,
            blocks as blk,
            turnovers as tov,
            fouls as pf
        FROM Teams
        WHERE game_id = ?
        ORDER BY tm
        """
        
        df_stats = pd.read_sql_query(stats_query, conn, params=(game_id,))
        conn.close()

        if not df_stats.empty:
            st.write("### 📊 Team Statistics")
            col1, col2 = st.columns(2)

            for idx, team_stats in df_stats.iterrows():
                with col1 if idx == 0 else col2:
                    st.write(f"### {team_stats['name']}")
                    
                    # Shooting Stats
                    st.write("🎯 **Shooting**")
                    st.write(f"Total Field Goals: {team_stats['fg']}/{team_stats['fga']} ({team_stats['fg_percentage']}%)")
                    st.write(f"2-Pointers: {team_stats['two_pointers_made']}/{team_stats['two_pointers_attempted']} ({team_stats['two_point_percentage']}%)")
                    st.write(f"3-Pointers: {team_stats['threep']}/{team_stats['threepa']} ({team_stats['three_percentage']}%)")
                    st.write(f"Free Throws: {team_stats['ft']}/{team_stats['fta']} ({team_stats['ft_percentage']}%)")
                    
                    # Points Breakdown
                    st.write("🏀 **Points Breakdown**")
                    two_points = 2 * team_stats['two_pointers_made']
                    three_points = 3 * team_stats['threep']
                    free_throw_points = team_stats['ft']
                    st.write(f"2-Point Points: {two_points}")
                    st.write(f"3-Point Points: {three_points}")
                    st.write(f"Free Throw Points: {free_throw_points}")
                    st.write(f"Total Points: {team_stats['points']}")
                    
                    # Rebounds
                    st.write("🏀 **Rebounds**")
                    st.write(f"Total: {team_stats['total_rebounds']}")
                    st.write(f"Offensive: {team_stats['orb']}")
                    st.write(f"Defensive: {team_stats['drb']}")
                    
                    # Other Stats
                    st.write("⚡ **Other Stats**")
                    st.write(f"Assists: {team_stats['ast']}")
                    st.write(f"Steals: {team_stats['stl']}")
                    st.write(f"Blocks: {team_stats['blk']}")
                    st.write(f"Turnovers: {team_stats['tov']}")
                    st.write(f"Fouls: {team_stats['pf']}")

    except Exception as e:
        st.error(f"Error displaying team statistics: {str(e)}")

def generate_player_performance_comparison(game_id):
    """Generate player performance comparison for the game."""
    st.subheader("🏀 Player Performance Comparison")
    
    # Fetch player stats for both teams
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        first_name || ' ' || last_name as player_name,
        team_id,
        minutes_played,
        points,
        rebounds_total,
        assists,
        steals,
        blocks,
        turnovers,
        field_goal_percentage,
        three_point_percentage
    FROM Players
    WHERE game_id = ? AND minutes_played != '0:00'
    ORDER BY team_id, points DESC
    """
    df_players = pd.read_sql_query(query, conn, params=(game_id,))
    conn.close()
    
    if df_players.empty:
        st.warning("No player data available for this game.")
        return
        
    # Split players by team
    team1_players = df_players[df_players['team_id'] == 1]
    team2_players = df_players[df_players['team_id'] == 2]
    
    # Display players side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Team 1**")
        st.dataframe(team1_players.drop('team_id', axis=1))
        
    with col2:
        st.write("**Team 2**")
        st.dataframe(team2_players.drop('team_id', axis=1))

def generate_advanced_metrics(game_id):
    """Generate advanced metrics for the game."""
    st.subheader("📈 Advanced Metrics")
    
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        team_id,
        field_goal_percentage,
        three_point_percentage,
        free_throw_percentage,
        (field_goals_made + 0.5 * three_pointers_made) * 100.0 / field_goals_attempted as efg_percentage,
        turnovers * 100.0 / (field_goals_attempted + 0.44 * free_throws_attempted) as tov_percentage,
        rebounds_offensive * 100.0 / (rebounds_offensive + rebounds_defensive) as orb_percentage,
        free_throws_attempted * 100.0 / field_goals_attempted as ftr_percentage
    FROM Teams
    WHERE game_id = ?
    """
    df_metrics = pd.read_sql_query(query, conn, params=(game_id,))
    conn.close()
    
    if df_metrics.empty:
        st.warning("No advanced metrics available for this game.")
        return
        
    # Format metrics for display
    metrics_cols = st.columns(2)
    
    for idx, team in enumerate(['Team 1', 'Team 2'], 1):
        with metrics_cols[idx-1]:
            st.write(f"**{team}**")
            team_metrics = df_metrics[df_metrics['team_id'] == idx].iloc[0]
            st.metric("Effective FG%", f"{team_metrics['efg_percentage']:.1f}%")
            st.metric("Turnover Rate", f"{team_metrics['tov_percentage']:.1f}%")
            st.metric("Offensive Rebound Rate", f"{team_metrics['orb_percentage']:.1f}%")
            st.metric("Free Throw Rate", f"{team_metrics['ftr_percentage']:.1f}%")

    # Display key moments in a timeline
    for _, moment in df_moments.iterrows():
        st.write(f"**Q{moment['period']} - {moment['game_time']}**")
        st.write(f"Score: {moment['current_score_team1']}-{moment['current_score_team2']}")
        st.write(f"Action: {moment['action_type']} ({moment['sub_type'] if pd.notna(moment['sub_type']) else ''})")
        st.write("---")

def generate_game_insights(game_id):
    """Generate detailed game insights and momentum analysis."""
    st.subheader("🏀 Game Flow Insights")
    
    conn = sqlite3.connect(db_path)
    
    # Get scoring runs and momentum shifts
    runs_query = """
    WITH ScoringEvents AS (
        SELECT 
            period,
            game_time,
            action_type,
            team_id,
            current_score_team1,
            current_score_team2,
            lead,
            LAG(lead) OVER (ORDER BY period, action_number) as previous_lead
        FROM PlayByPlay
        WHERE game_id = ? 
        AND action_type IN ('2pt', '3pt', 'freethrow')
        ORDER BY period, action_number
    )
    SELECT 
        period,
        game_time,
        team_id,
        current_score_team1,
        current_score_team2,
        lead - COALESCE(previous_lead, 0) as run_size
    FROM ScoringEvents
    WHERE ABS(lead - COALESCE(previous_lead, 0)) >= 6
    ORDER BY period, game_time DESC
    """
    
    # Get critical plays with importance classification
    critical_plays_query = """
    SELECT 
        period,
        game_time,
        action_type,
        sub_type,
        team_id,
        current_score_team1,
        current_score_team2,
        lead,
        CASE 
            WHEN ABS(lead) <= 5 AND period = 4 AND 
                CAST(SUBSTR(game_time, 1, INSTR(game_time, ':') - 1) AS INTEGER) * 60 + 
                CAST(SUBSTR(game_time, INSTR(game_time, ':') + 1) AS INTEGER) <= 120 THEN 'Clutch Time'
            WHEN action_type = '3pt' AND ABS(lead) <= 8 THEN 'Momentum Changer'
            WHEN action_type IN ('block', 'steal') AND ABS(lead) <= 10 THEN 'Big Defense'
            WHEN action_type IN ('2pt', '3pt') AND period = 4 AND ABS(lead) <= 10 THEN 'Key Basket'
            ELSE 'Normal'
        END as play_importance
    FROM PlayByPlay
    WHERE game_id = ? 
    AND (
        -- Clutch time plays in close game
        (period = 4 
         AND CAST(SUBSTR(game_time, 1, INSTR(game_time, ':') - 1) AS INTEGER) * 60 + 
             CAST(SUBSTR(game_time, INSTR(game_time, ':') + 1) AS INTEGER) <= 120
         AND ABS(lead) <= 5
        )
        OR
        -- Big shots that change momentum
        (action_type = '3pt' AND ABS(lead) <= 8)
        OR
        -- Critical defensive plays
        (action_type IN ('block', 'steal') AND ABS(lead) <= 10)
        OR
        -- Important baskets in 4th quarter
        (action_type IN ('2pt', '3pt') AND period = 4 AND ABS(lead) <= 10)
    )
    ORDER BY period DESC, game_time DESC
    """
    
    # Get team names
    teams_query = "SELECT tm, name FROM Teams WHERE game_id = ? ORDER BY tm"
    
    # Execute queries
    df_runs = pd.read_sql_query(runs_query, conn, params=(game_id,))
    df_critical = pd.read_sql_query(critical_plays_query, conn, params=(game_id,))
    df_teams = pd.read_sql_query(teams_query, conn, params=(game_id,))
    
    # Create team names dictionary
    team_names = dict(zip(df_teams['tm'], df_teams['name']))
    
    conn.close()
    
    # Display game summary
    if not df_runs.empty:
        st.write("### 📊 Game Summary")
        
        total_runs = len(df_runs)
        avg_run_size = df_runs['run_size'].abs().mean()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Scoring Runs", total_runs)
        with col2:
            st.metric("Avg Run Size", f"{avg_run_size:.1f}")
        with col3:
            if total_runs > 8:
                st.write("🏃‍♂️ High-paced game!")
            elif total_runs > 4:
                st.write("⚖️ Balanced pace")
            else:
                st.write("🐢 Defensive battle")
    
    # Display biggest runs
    if not df_runs.empty:
        st.write("### 🔥 Biggest Runs")
        
        # Sort runs by size and take top 3
        biggest_runs = df_runs.nlargest(3, 'run_size', key=abs)
        
        for _, run in biggest_runs.iterrows():
            col1, col2 = st.columns([2, 3])
            
            with col1:
                team_name = team_names.get(run['team_id'], f"Team {run['team_id']}")
                st.write(f"**Q{run['period']} - {run['game_time']}**")
                st.write(f"**{team_name}**")
            
            with col2:
                st.write(f"Score: {run['current_score_team1']}-{run['current_score_team2']}")
                run_size = abs(run['run_size'])
                if run_size >= 10:
                    st.write(f"🌟 {run_size}-0 Run!")
                else:
                    st.write(f"✨ {run_size}-0 Run")
            
            st.markdown("<hr style='margin: 5px 0px'>", unsafe_allow_html=True)
    
    # Display critical plays
    if not df_critical.empty:
        st.write("### ⭐ Game-Changing Plays")
        
        # Define importance emojis and descriptions
        importance_info = {
            'Clutch Time': {'emoji': '🔥', 'desc': 'Crucial play in clutch time!'},
            'Momentum Changer': {'emoji': '🌊', 'desc': 'Momentum-shifting play'},
            'Big Defense': {'emoji': '🛡️', 'desc': 'Key defensive stop'},
            'Key Basket': {'emoji': '🎯', 'desc': 'Important basket'},
            'Normal': {'emoji': '⚡', 'desc': ''}
        }
        
        for _, play in df_critical.iterrows():
            col1, col2, col3 = st.columns([1, 2, 1])
            
            team_name = team_names.get(play['team_id'], f"Team {play['team_id']}")
            play_type = play['action_type']
            if pd.notnull(play['sub_type']):
                play_type += f" ({play['sub_type']})"
            
            importance = play['play_importance']
            emoji = importance_info[importance]['emoji']
            desc = importance_info[importance]['desc']
            
            with col1:
                st.write(f"**Q{play['period']} - {play['game_time']}**")
            
            with col2:
                st.write(f"{emoji} {team_name}")
                st.write(f"{play_type.upper()}")
            
            with col3:
                score_diff = abs(play['current_score_team1'] - play['current_score_team2'])
                score_text = f"{play['current_score_team1']}-{play['current_score_team2']}"
                if score_diff <= 3:
                    st.write(f"😱 {score_text}")
                else:
                    st.write(score_text)
            
            if desc:
                st.write(f"*{desc}*")
            
            st.markdown("<hr style='margin: 5px 0px'>", unsafe_allow_html=True)
    
    # If no data available
    if df_runs.empty and df_critical.empty:
        st.warning("No detailed game insights available for this match.")

def generate_match_report(game_id):
    """Generate a comprehensive single-game analysis with detailed statistics."""
    match_data, scorers_data, quarters_data = fetch_match_report_data(game_id)
    
    if match_data.empty:
        st.error("No match data available.")
        return
    
    # Match header
    st.title(f"🏀 Match Report: {match_data.iloc[0]['home_team']} vs {match_data.iloc[0]['away_team']}")
    
    # Score display
    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.subheader(match_data.iloc[0]['home_team'])
        st.header(f"{int(match_data.iloc[0]['home_score'])}")
    with col2:
        st.subheader("VS")
    with col3:
        st.subheader(match_data.iloc[0]['away_team'])
        st.header(f"{int(match_data.iloc[0]['away_score'])}")
    
    # Quarter by quarter breakdown
    st.subheader("📊 Quarter by Quarter")
    quarter_cols = st.columns(4)
    for idx, quarter in enumerate(['Q1', 'Q2', 'Q3', 'Q4'], 1):
        with quarter_cols[idx-1]:
            st.metric(
                f"Quarter {idx}",
                f"{quarters_data.iloc[0][quarter]}-{quarters_data.iloc[1][quarter]}",
                delta=int(quarters_data.iloc[0][quarter]) - int(quarters_data.iloc[1][quarter])
            )
    
    # Key statistics comparison
    st.subheader("📈 Key Statistics")
    stats_cols = st.columns([2, 2, 2])
    
    with stats_cols[0]:
        home_fg_pct = float(match_data.iloc[0]['home_fg_pct'])
        away_fg_pct = float(match_data.iloc[0]['away_fg_pct'])
        st.metric(
            "Field Goal %",
            f"{home_fg_pct:.1f}% vs {away_fg_pct:.1f}%",
            delta=f"{home_fg_pct - away_fg_pct:.1f}%"
        )
    
    with stats_cols[1]:
        home_3p_pct = float(match_data.iloc[0]['home_3p_pct'])
        away_3p_pct = float(match_data.iloc[0]['away_3p_pct'])
        st.metric(
            "Three Point %",
            f"{home_3p_pct:.1f}% vs {away_3p_pct:.1f}%",
            delta=f"{home_3p_pct - away_3p_pct:.1f}%"
        )
    
    with stats_cols[2]:
        home_assists = int(match_data.iloc[0]['home_assists'])
        away_assists = int(match_data.iloc[0]['away_assists'])
        st.metric(
            "Assists",
            f"{home_assists} vs {away_assists}",
            delta=home_assists - away_assists
        )
    
    # Top scorers
    st.subheader("🏆 Top Performers")
    
    # Split scorers by team
    home_scorers = scorers_data[scorers_data['team_id'] == 1]
    away_scorers = scorers_data[scorers_data['team_id'] == 2]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{match_data.iloc[0]['home_team']}**")
        for _, scorer in home_scorers.iterrows():
            st.write(f"{scorer['player_name']}: {scorer['points']} pts ({scorer['fg']} FG, {scorer['three_pt']} 3PT)")
    
    with col2:
        st.write(f"**{match_data.iloc[0]['away_team']}**")
        for _, scorer in away_scorers.iterrows():
            st.write(f"{scorer['player_name']}: {scorer['points']} pts ({scorer['fg']} FG, {scorer['three_pt']} 3PT)")
    
    # Game flow chart
    st.subheader("📈 Game Flow")
    plot_score_lead_full_game(game_id)
    
    # Shot Chart
    st.subheader("🎯 Shot Chart")
    plot_match_shot_chart(game_id)
    
    # Player Performance Comparison
    generate_player_performance_comparison(game_id)
    
    # Advanced Metrics
    generate_advanced_metrics(game_id)

   # Add this line to display team stats
    display_team_stats(game_id)
        

def fetch_match_report_data(game_id):
    """Fetch comprehensive match report data."""
    conn = sqlite3.connect(db_path)
    
    # Basic match info
    match_query = """
    SELECT 
        t1.name as home_team,
        t2.name as away_team,
        (t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) as home_score,
        (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score) as away_score,
        t1.field_goal_percentage as home_fg_pct,
        t2.field_goal_percentage as away_fg_pct,
        t1.three_point_percentage as home_3p_pct,
        t2.three_point_percentage as away_3p_pct,
        t1.rebounds_total as home_rebounds,
        t2.rebounds_total as away_rebounds,
        t1.assists as home_assists,
        t2.assists as away_assists
    FROM Teams t1
    JOIN Teams t2 ON t1.game_id = t2.game_id
    WHERE t1.game_id = ? AND t1.tm = 1 AND t2.tm = 2;
    """
    
    # Top scorers
    scorers_query = """
    SELECT 
        first_name || ' ' || last_name as player_name,
        points,
        minutes_played,
        field_goals_made || '/' || field_goals_attempted as fg,
        three_pointers_made || '/' || three_pointers_attempted as three_pt,
        team_id
    FROM Players
    WHERE game_id = ? AND points > 0
    ORDER BY points DESC
    LIMIT 6;
    """
    
    # Quarter by quarter
    quarters_query = """
    SELECT 
        tm,
        p1_score as Q1,
        p2_score as Q2,
        p3_score as Q3,
        p4_score as Q4
    FROM Teams
    WHERE game_id = ?
    ORDER BY tm;
    """
    
    match_data = pd.read_sql_query(match_query, conn, params=(game_id,))
    scorers_data = pd.read_sql_query(scorers_query, conn, params=(game_id,))
    quarters_data = pd.read_sql_query(quarters_query, conn, params=(game_id,))
    
    conn.close()
    
    return match_data, scorers_data, quarters_data

def plot_match_shot_chart(game_id):
    """Plot the shot chart for a specific game with made/missed shots for both teams."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        player_name,
        x_coord,
        y_coord,
        shot_result,
        team_id
    FROM Shots
    WHERE game_id = ?;
    """
    df_shots = pd.read_sql_query(query, conn, params=(game_id,))
    conn.close()

    if df_shots.empty:
        st.warning("No shot data available for this game.")
        return

    # Convert coordinates to court image scale
    df_shots['x_coord'] = df_shots['x_coord'] * 2.8
    df_shots['y_coord'] =(df_shots['y_coord'] * 2.61)

    # Load court image
    court_img = mpimg.imread("fiba_courtonly.jpg")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(court_img, extent=[0, 280, 0, 261], aspect="auto")

    # Plot shots for each team with made/missed distinction
    for team_id in [1, 2]:  # Team 1 (Home) and Team 2 (Away)
        team_shots = df_shots[df_shots['team_id'] == team_id]
        
        # Made shots (circles)
        made_shots = team_shots[team_shots['shot_result'] == 1]
        # Missed shots (crosses)
        missed_shots = team_shots[team_shots['shot_result'] == 0]
        
        if team_id == 1:  # Home team
            ax.scatter(made_shots['x_coord'], made_shots['y_coord'], 
                      marker='o', c='red', s=50, label='Home Made', alpha=0.7)
            ax.scatter(missed_shots['x_coord'], missed_shots['y_coord'], 
                      marker='x', c='red', s=50, label='Home Missed', alpha=0.7)
        else:  # Away team
            ax.scatter(made_shots['x_coord'], made_shots['y_coord'], 
                      marker='o', c='blue', s=50, label='Away Made', alpha=0.7)
            ax.scatter(missed_shots['x_coord'], missed_shots['y_coord'], 
                      marker='x', c='blue', s=50, label='Away Missed', alpha=0.7)

    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))

    # Remove all axis elements
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.axis("off")

    st.pyplot(fig)

def display_match_report():
    selected_match = create_match_selector()
    
    if selected_match:
        generate_match_report(selected_match)
    else:
        st.info("Please select a match to view the report.")

def generate_advanced_metrics(game_id):
    """Generate advanced metrics for the game."""
    st.subheader("📈 Advanced Metrics")
    
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        tm as team_id,
        name as team_name,
        field_goal_percentage,
        three_point_percentage,
        free_throw_percentage,
        (field_goals_made + 0.5 * three_pointers_made) * 100.0 / NULLIF(field_goals_attempted, 0) as efg_percentage,
        turnovers * 100.0 / NULLIF((field_goals_attempted + 0.44 * free_throws_attempted), 0) as tov_percentage,
        rebounds_offensive * 100.0 / NULLIF((rebounds_offensive + rebounds_defensive), 0) as orb_percentage,
        free_throws_attempted * 100.0 / NULLIF(field_goals_attempted, 0) as ftr_percentage
    FROM Teams
    WHERE game_id = ?
    ORDER BY tm
    """
    df_metrics = pd.read_sql_query(query, conn, params=(game_id,))
    conn.close()
    
    if df_metrics.empty:
        st.warning("No advanced metrics available for this game.")
        return
        
    # Format metrics for display
    metrics_cols = st.columns(2)
    
    for idx, (_, team_metrics) in enumerate(df_metrics.iterrows()):
        with metrics_cols[idx]:
            st.write(f"**{team_metrics['team_name']}**")
            st.metric("Effective FG%", 
                     f"{team_metrics['efg_percentage']:.1f}%" if pd.notnull(team_metrics['efg_percentage']) else "N/A")
            st.metric("Turnover Rate", 
                     f"{team_metrics['tov_percentage']:.1f}%" if pd.notnull(team_metrics['tov_percentage']) else "N/A")
            st.metric("Offensive Rebound Rate", 
                     f"{team_metrics['orb_percentage']:.1f}%" if pd.notnull(team_metrics['orb_percentage']) else "N/A")
            st.metric("Free Throw Rate", 
                     f"{team_metrics['ftr_percentage']:.1f}%" if pd.notnull(team_metrics['ftr_percentage']) else "N/A")

def display_in_game_page():
    """Display the In Game page with live game stats and analysis."""
    st.title("🏀 In Game Analysis")
    
    # Game selector
    selected_match = create_match_selector()
    
    if selected_match:
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "Live Stats 📊", 
            "Play by Play 🎯", 
            "Team Analysis 📈",
            "Player Stats 👤"
        ])
        
        with tab1:
            st.subheader("Live Game Statistics")
            # Display current game stats
            display_team_stats(selected_match)
            
            # Add score progression chart
            st.subheader("Score Progression")
            plot_score_lead_full_game(selected_match)
        
        with tab2:
            st.subheader("Play by Play")
            # Add quarter selector
            quarter = st.selectbox(
                "Select Quarter",
                ["All Quarters", "Q1", "Q2", "Q3", "Q4"],
                key="quarter_selector"
            )
            
            # Fetch and display play by play data
            selected_quarter = None if quarter == "All Quarters" else int(quarter[-1])
            actions = fetch_pbp_actions(selected_match, selected_quarter)
            display_pbp_actions(actions)
        
        with tab3:
            st.subheader("Team Analysis")
            # Show team comparisons and advanced metrics
            generate_advanced_metrics(selected_match)
            
            # Show substitution patterns
            substitutions = count_substitutions(selected_match)
            if not substitutions.empty:
                st.subheader("Substitution Patterns")
                st.dataframe(substitutions)
        
        with tab4:
            st.subheader("Player Statistics")
            # Show player performance comparison
            generate_player_performance_comparison(selected_match)
            
            # Show starting lineups
            starting_five = fetch_starting_five(selected_match)
            if not starting_five.empty:
                st.subheader("Starting Lineups")
                
                col1, col2 = st.columns(2)
                team1_starters = starting_five[starting_five['team_id'] == 1]
                team2_starters = starting_five[starting_five['team_id'] == 2]
                
                with col1:
                    st.write("**Home Team**")
                    st.dataframe(team1_starters[['player_name']])
                
                with col2:
                    st.write("**Away Team**")
                    st.dataframe(team2_starters[['player_name']])
    else:
        st.info("Please select a match to view in-game analysis.")

def analyze_shot_patterns(player_name):
    """Analyze shot patterns for a player."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        x_coord, 
        y_coord, 
        shot_result,
        action_type,
        shot_sub_type,
        period,
        game_id
    FROM Shots 
    WHERE player_name = ?;
    """
    df_shots = pd.read_sql_query(query, conn, params=(player_name,))
    conn.close()

    if df_shots.empty:
        st.warning(f"No shot data found for {player_name}.")
        return

    # Calculate shooting percentages
    total_shots = len(df_shots)
    made_shots = df_shots['shot_result'].sum()
    shooting_pct = (made_shots / total_shots * 100) if total_shots > 0 else 0

    # Calculate distance for each shot
    df_shots['distance'] = df_shots.apply(
        lambda row: calculate_distance_from_basket(row['x_coord'], row['y_coord']), 
        axis=1
    )

    # Shot type and subtype distribution
    shot_analysis = df_shots.groupby(['action_type', 'shot_sub_type']).agg({
        'shot_result': ['count', 'mean']
    }).round(3)
    
    shot_analysis.columns = ['Attempts', 'Success Rate']
    shot_analysis['Success Rate'] = (shot_analysis['Success Rate'] * 100).round(1).astype(str) + '%'

    # Display overall statistics
    st.write("### Overall Shooting Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Shots", total_shots)
    with col2:
        st.metric("Made Shots", made_shots)
    with col3:
        st.metric("Field Goal %", f"{shooting_pct:.1f}%")

    # Shot distribution
    st.write("### Shot Type Distribution")
    
    # Create two columns for shot types and shot chart
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(shot_analysis)
    
    with col2:
        # Create shot chart
        fig, ax = plt.subplots(figsize=(10, 8))
        court_img = mpimg.imread("fiba_courtonly.jpg")
        ax.imshow(court_img, extent=[0, 280, 0, 261], aspect="auto")

        # Scale coordinates
        df_shots['x_coord_scaled'] = df_shots['x_coord'] * 2.8
        df_shots['y_coord_scaled'] = (df_shots['y_coord'] * 2.61)

        # Plot made shots
        made = df_shots[df_shots['shot_result'] == 1]
        ax.scatter(made['x_coord_scaled'], made['y_coord_scaled'], 
                  c='green', marker='o', s=50, alpha=0.7, label='Made')

        # Plot missed shots
        missed = df_shots[df_shots['shot_result'] == 0]
        ax.scatter(missed['x_coord_scaled'], missed['y_coord_scaled'], 
                  c='red', marker='x', s=50, alpha=0.7, label='Missed')

        ax.legend()
        ax.axis('off')
        st.pyplot(fig)

    # Shot distance analysis
    st.write("### Shot Distance Analysis")
    distance_bins = [0, 3, 5, 7, 10, float('inf')]
    distance_labels = ['Close Range (0-3m)', 'Short Range (3-5m)', 
                      'Mid Range (5-7m)', 'Long Mid (7-10m)', 'Long Range (10m+)']
    df_shots['distance_range'] = pd.cut(df_shots['distance'], bins=distance_bins, labels=distance_labels)
    
    distance_stats = df_shots.groupby('distance_range').agg({
        'shot_result': ['count', 'mean']
    }).round(3)
    
    distance_stats.columns = ['Attempts', 'Success Rate']
    distance_stats['Success Rate'] = (distance_stats['Success Rate'] * 100).round(1).astype(str) + '%'
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(distance_stats)
    
    with col2:
        # Visualize shot success by distance
        shot_success_by_distance = df_shots.groupby('distance_range')['shot_result'].mean() * 100
        fig, ax = plt.subplots(figsize=(10, 6))
        shot_success_by_distance.plot(kind='bar')
        plt.title('Shot Success Rate by Distance')
        plt.ylabel('Success Rate (%)')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Favorite shot types
    st.write("### Preferred Shot Types")
    favorite_shots = df_shots.groupby('shot_sub_type').agg({
        'shot_result': ['count', 'mean']
    }).round(3)
    
    favorite_shots.columns = ['Attempts', 'Success Rate']
    favorite_shots = favorite_shots.sort_values('Attempts', ascending=False)
    favorite_shots['Success Rate'] = (favorite_shots['Success Rate'] * 100).round(1).astype(str) + '%'
    
    st.dataframe(favorite_shots)

    # Most successful shot types (minimum 5 attempts)
    st.write("### Most Successful Shot Types (min. 5 attempts)")
    successful_shots = df_shots.groupby('shot_sub_type').filter(lambda x: len(x) >= 5)
    successful_shots = successful_shots.groupby('shot_sub_type').agg({
        'shot_result': ['count', 'mean']
    }).round(3)
    
    successful_shots.columns = ['Attempts', 'Success Rate']
    successful_shots = successful_shots.sort_values(('Success Rate'), ascending=False)
    successful_shots['Success Rate'] = (successful_shots['Success Rate'] * 100).round(1).astype(str) + '%'
    
    st.dataframe(successful_shots)

def calculate_shot_zones(shots_df):
    """Calculate shot distribution and success rates by zone."""
    def get_zone(row):
        x, y = row['x_coord'], row['y_coord']
        distance = np.sqrt((x - 50)**2 + (y - 25)**2)
        
        if distance < 5:
            return 'Paint'
        elif distance < 15:
            return 'Mid-range'
        else:
            return '3PT'
    
    shots_df['zone'] = shots_df.apply(get_zone, axis=1)
    zones = shots_df.groupby('zone').agg({
        'shot_result': ['count', 'mean']
    }).round(3)
    
    return zones.to_dict()

def identify_hot_zones(shots_df):
    """Identify areas where a player has high success rate."""
    if shots_df.empty:
        return {}
        
    # Create court zones grid
    x_bins = np.linspace(0, 100, 5)
    y_bins = np.linspace(0, 50, 3)
    
    shots_df['x_bin'] = pd.cut(shots_df['x_coord'], x_bins, labels=False)
    shots_df['y_bin'] = pd.cut(shots_df['y_coord'], y_bins, labels=False)
    
    hot_zones = shots_df.groupby(['x_bin', 'y_bin']).agg({
        'shot_result': ['count', 'mean']
    }).reset_index()
    
    # Filter for zones with high success rate (above 40%) and minimum attempts
    hot_zones = hot_zones[
        (hot_zones[('shot_result', 'mean')] > 0.4) & 
        (hot_zones[('shot_result', 'count')] >= 3)
    ]
    
    return hot_zones.to_dict()

def predict_next_shot(patterns, game_situation):
    """Predict the most likely next shot based on patterns and current game situation."""
    predictions = []
    
    for player, stats in patterns['player_patterns'].items():
        # Basic probability of player taking the shot
        shot_prob = stats['total_shots'] / sum(p['total_shots'] for p in patterns['player_patterns'].values())
        
        # Adjust for hot zones
        hot_zone_bonus = 0.1 if stats['hot_zones'] else 0
        
        # Adjust for game situation
        if game_situation['score_diff'] < -5:
            # Team is behind - prefer high percentage shots and shooters
            situational_bonus = stats['success_rate'] * 0.2
        else:
            situational_bonus = 0
            
        final_prob = shot_prob + hot_zone_bonus + situational_bonus
        
        most_likely_zone = max(stats['shot_zones'].items(), key=lambda x: x[1][('shot_result', 'count')])[0]
        preferred_type = max(stats['preferred_types'].items(), key=lambda x: x[1])[0]
        
        predictions.append({
            'player': player,
            'probability': final_prob,
            'likely_zone': most_likely_zone,
            'shot_type': preferred_type,
            'expected_success_rate': stats['success_rate']
        })
    
    return sorted(predictions, key=lambda x: x['probability'], reverse=True)

def display_shot_analysis(lineup_players, team_name, game_situation):
    """Display shot analysis and predictions."""
    
    patterns = analyze_shot_patterns(lineup_players, team_name, 
                                   game_situation['game_time'], 
                                   game_situation['quarter'])
    
    if not patterns:
        st.warning("No historical shot data available for this lineup combination.")
        return
    
    # Predict next shot
    shot_predictions = predict_next_shot(patterns, game_situation)
    
    # Display Predictions
    st.subheader("🎯 Shot Predictions")
    
    # Top 3 most likely shooters
    cols = st.columns(3)
    for i, pred in enumerate(shot_predictions[:3]):
        with cols[i]:
            st.metric(
                f"#{i+1} Most Likely Shooter",
                pred['player'],
                f"{pred['probability']*100:.1f}%"
            )
            st.write(f"Likely zone: {pred['likely_zone']}")
            st.write(f"Shot type: {pred['shot_type']}")
            st.write(f"Success rate: {pred['expected_success_rate']*100:.1f}%")
    
    # Shot Chart
    st.subheader("📊 Team Shot Distribution")
    
    # Create shot chart visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot court
    draw_court(ax)
    
    # Plot shot locations with heatmap
    for player, stats in patterns['player_patterns'].items():
        player_shots = pd.DataFrame(stats['hot_zones'])
        if not player_shots.empty:
            sns.kdeplot(
                data=player_shots,
                x='x_coord',
                y='y_coord',
                levels=10,
                thresh=.2,
                alpha=.5
            )
    
    st.pyplot(fig)
    
    # Time-based Analysis
    st.subheader("⏱️ Time-based Shot Patterns")
    time_patterns = patterns['time_patterns']
    
    # Show when team is most likely to shoot
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        data=time_patterns,
        x='game_time',
        y=('shot_result', 'count'),
        hue='period'
    )
    st.pyplot(fig2)

def display_in_game_page():
    st.title("🏀 In-Game Shot Prediction")
    
    # Game Situation Input
    st.subheader("Current Game Situation")
    
    cols = st.columns(4)
    with cols[0]:
        teams = fetch_teams()
        team = st.selectbox("Team", teams)
    
    with cols[1]:
        quarter = st.selectbox("Quarter", [1, 2, 3, 4])
    
    with cols[2]:
        game_time = st.text_input("Time Remaining", "10:00")
    
    with cols[3]:
        score_diff = st.number_input("Score Difference", -50, 50, 0)
    
    # Lineup Selection
    st.subheader("Current Lineup")
    team_players = fetch_team_players(team)
    lineup = st.multiselect("Select 5 players", team_players, max_selections=5)
    
    if len(lineup) == 5:
        game_situation = {
            'quarter': quarter,
            'game_time': game_time,
            'score_diff': score_diff
        }
        
        display_shot_analysis(lineup, team, game_situation)

def analyze_shot_patterns(player_name):
    """Analyze shot patterns for a player."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        x_coord, 
        y_coord, 
        shot_result,
        action_type,
        shot_sub_type,
        period,
        game_id
    FROM Shots 
    WHERE player_name = ?;
    """
    df_shots = pd.read_sql_query(query, conn, params=(player_name,))
    conn.close()

    if df_shots.empty:
        st.warning(f"No shot data found for {player_name}.")
        return

    # Calculate shooting percentages
    total_shots = len(df_shots)
    made_shots = df_shots['shot_result'].sum()
    shooting_pct = (made_shots / total_shots * 100) if total_shots > 0 else 0

    # Shot type and subtype distribution
    shot_analysis = df_shots.groupby(['action_type', 'shot_sub_type']).agg({
        'shot_result': ['count', 'mean']
    }).round(3)
    
    shot_analysis.columns = ['Attempts', 'Success Rate']
    shot_analysis['Success Rate'] = (shot_analysis['Success Rate'] * 100).round(1).astype(str) + '%'

    # Display overall statistics
    st.write("### Overall Shooting Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Shots", total_shots)
    with col2:
        st.metric("Made Shots", made_shots)
    with col3:
        st.metric("Field Goal %", f"{shooting_pct:.1f}%")

    # Shot distribution
    st.write("### Shot Type Distribution")
    st.dataframe(shot_analysis)

    # Favorite shot types
    st.write("### Preferred Shot Types")
    favorite_shots = df_shots.groupby('shot_sub_type').agg({
        'shot_result': ['count', 'mean']
    }).round(3)
    
    favorite_shots.columns = ['Attempts', 'Success Rate']
    favorite_shots = favorite_shots.sort_values('Attempts', ascending=False)
    favorite_shots['Success Rate'] = (favorite_shots['Success Rate'] * 100).round(1).astype(str) + '%'
    
    st.dataframe(favorite_shots)

    # Most successful shot types (minimum 5 attempts)
    st.write("### Most Successful Shot Types (min. 5 attempts)")
    successful_shots = df_shots.groupby('shot_sub_type').filter(lambda x: len(x) >= 5)
    successful_shots = successful_shots.groupby('shot_sub_type').agg({
        'shot_result': ['count', 'mean']
    }).round(3)
    
    successful_shots.columns = ['Attempts', 'Success Rate']
    successful_shots = successful_shots.sort_values(('Success Rate'), ascending=False)
    successful_shots['Success Rate'] = (successful_shots['Success Rate'] * 100).round(1).astype(str) + '%'
    
    st.dataframe(successful_shots)

        # Add opponent analysis in an expander
    with st.expander("Analysis by Opponent", expanded=False):
        conn = sqlite3.connect(db_path)
        query = """
        WITH PlayerTeam AS (
            SELECT s.game_id, s.team_id, t.name as player_team
            FROM Shots s
            JOIN Teams t ON s.game_id = t.game_id AND s.team_id = t.tm
            WHERE s.player_name = ?
            GROUP BY s.game_id, s.team_id
        )
        SELECT 
            s.shot_result,
            s.action_type,
            s.shot_sub_type,
            s.x_coord,
            s.y_coord,
            t.name as opponent_team
        FROM Shots s
        JOIN PlayerTeam pt ON s.game_id = pt.game_id
        JOIN Teams t ON t.game_id = s.game_id 
            AND t.tm = CASE 
                WHEN pt.team_id = 1 THEN 2 
                ELSE 1 
            END
        WHERE s.player_name = ?;
        """
        df_opponent = pd.read_sql_query(query, conn, params=(player_name, player_name))
        conn.close()

        if not df_opponent.empty:
            # Overall stats by opponent
            team_analysis = df_opponent.groupby('opponent_team').agg({
                'shot_result': ['count', 'mean']
            }).round(3)
            
            team_analysis.columns = ['Total Shots', 'Success Rate']
            team_analysis = team_analysis.sort_values('Total Shots', ascending=False)
            team_analysis['Success Rate'] = (team_analysis['Success Rate'] * 100).round(1)
            
            # Visualization of success rates against all opponents
            fig_success = px.bar(
                team_analysis.reset_index(),
                x='opponent_team',
                y='Success Rate',
                title=f"Shot Success Rate Against Different Teams - {player_name}",
                labels={'opponent_team': 'Opponent Team', 'Success Rate': 'Success Rate (%)'}
            )
            st.plotly_chart(fig_success)

            # Shot type distribution
            shot_type_dist = df_opponent.groupby(['opponent_team', 'shot_sub_type']).size().unstack(fill_value=0)
            fig_dist = px.bar(
                shot_type_dist.reset_index(),
                x='opponent_team',
                y=shot_type_dist.columns,
                title=f"Shot Type Distribution Against Teams - {player_name}",
                barmode='stack'
            )
            st.plotly_chart(fig_dist)
            
def create_shot_chart(df_shots, title):
    """Create a shot chart using plotly"""
    # Create basketball court
    fig = go.Figure()

    # Add court outline
    fig.add_shape(type="rect",
                  x0=-250, y0=-47.5, x1=250, y1=422.5,
                  line=dict(color="black", width=1))
    
    # Add three point line
    fig.add_shape(type="circle",
                  x0=-220, y0=-47.5, x1=220, y1=392.5,
                  line=dict(color="black", width=1))
    
    # Add free throw circle
    fig.add_shape(type="circle",
                  x0=-60, y0=140, x1=60, y1=260,
                  line=dict(color="black", width=1))

    # Plot shots
    made_shots = df_shots[df_shots['shot_result'] == 1]
    missed_shots = df_shots[df_shots['shot_result'] == 0]

    fig.add_trace(go.Scatter(
        x=made_shots['x_coord'],
        y=made_shots['y_coord'],
        mode='markers',
        name='Made',
        marker=dict(color='green', size=8, symbol='circle'),
    ))

    fig.add_trace(go.Scatter(
        x=missed_shots['x_coord'],
        y=missed_shots['y_coord'],
        mode='markers',
        name='Missed',
        marker=dict(color='red', size=8, symbol='x'),
    ))

    fig.update_layout(
        title=title,
        showlegend=True,
        xaxis=dict(range=[-250, 250], showgrid=False),
        yaxis=dict(range=[-47.5, 422.5], showgrid=False),
        yaxis_scaleanchor="x",
    )

    return fig

def get_players_on_court(df_pbp, game_id):
    """Track which players are on court based on substitutions."""
    conn = sqlite3.connect(db_path)
    
    # Get starting lineups (players marked as starters)
    starters_query = """
    SELECT team_id, json_player_id as player_id
    FROM Players
    WHERE game_id = ? AND starter = 1;
    """
    df_starters = pd.read_sql_query(starters_query, conn, params=(game_id,))
    conn.close()

    # Initialize dictionaries to track players on court for each team
    team1_players = set(df_starters[df_starters['team_id'] == 1]['player_id'])
    team2_players = set(df_starters[df_starters['team_id'] == 2]['player_id'])
    
    # List to store players on court for each action
    players_by_action = []
    
    # Process each action
    for _, row in df_pbp.iterrows():
        if row['action_type'] == 'substitution':
            team_id = row['team_id']
            player_id = row['player_id']
            sub_type = row['sub_type']
            
            if team_id == 1:
                players = team1_players
            else:
                players = team2_players
                
            if sub_type == 'in':
                players.add(player_id)
            elif sub_type == 'out':
                players.discard(player_id)
        
        # Record current players on court
        players_by_action.append({
            'team1_players': sorted(list(team1_players)),
            'team2_players': sorted(list(team2_players))
        })
    
    return players_by_action

def fetch_pbp_data(game_id):
    """Fetch play by play data for a specific game."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT 
        action_number,
        period,
        game_time,
        action_type,
        sub_type,
        team_id,
        player_id,
        success,
        current_score_team1,
        current_score_team2,
        lead
    FROM PlayByPlay
    WHERE game_id = ?
    ORDER BY action_number ASC;
    """
    try:
        df = pd.read_sql_query(query, conn, params=(game_id,))
    except Exception as e:
        st.error(f"Error fetching play-by-play data: {str(e)}")
        df = pd.DataFrame()
    finally:
        conn.close()
    return df

def get_player_names_dict(game_id):
    """Create a dictionary mapping (team_id, player_id) tuples to player names."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT team_id, 
           json_player_id as player_id, 
           first_name || ' ' || last_name as player_name
    FROM Players
    WHERE game_id = ?;
    """
    df_players = pd.read_sql_query(query, conn, params=(game_id,))
    conn.close()
    
    # Create dictionary with (team_id, player_id) as key
    return {(row['team_id'], row['player_id']): row['player_name'] 
            for _, row in df_players.iterrows()}

def analyze_five_player_segments(game_id):
    """Analyze segments where specific combinations of five players played together."""
    conn = sqlite3.connect(db_path)
    
    # Get starting lineups with team_id
    starters_query = """
    SELECT team_id, json_player_id as player_id
    FROM Players
    WHERE game_id = ? AND starter = 1
    ORDER BY team_id;
    """
    df_starters = pd.read_sql_query(starters_query, conn, params=(game_id,))
    
    # Get all substitutions with action numbers and lead
    subs_query = """
    SELECT action_number, team_id, player_id, sub_type, lead
    FROM PlayByPlay
    WHERE game_id = ? AND action_type = 'substitution'
    ORDER BY action_number;
    """
    df_subs = pd.read_sql_query(subs_query, conn, params=(game_id,))
    
    # Get all action numbers and leads
    leads_query = """
    SELECT action_number, lead
    FROM PlayByPlay
    WHERE game_id = ?
    ORDER BY action_number;
    """
    df_leads = pd.read_sql_query(leads_query, conn, params=(game_id,))
    
    # Get last action number and lead
    last_action_query = """
    SELECT MAX(action_number) as last_action, lead as final_lead
    FROM PlayByPlay
    WHERE game_id = ?;
    """
    last_action_data = pd.read_sql_query(last_action_query, conn, params=(game_id,)).iloc[0]
    last_action = last_action_data['last_action']
    
    conn.close()

    # Get player names dictionary
    player_names = get_player_names_dict(game_id)

    # Initialize segments tracking
    segments = []
    
    # Initialize current fives for both teams using (team_id, player_id) tuples
    current_five = {
        1: set((1, pid) for pid in df_starters[df_starters['team_id'] == 1]['player_id']),
        2: set((2, pid) for pid in df_starters[df_starters['team_id'] == 2]['player_id'])
    }
    
    # Add initial segment (starters)
    if len(current_five[1]) == 5 and len(current_five[2]) == 5:
        initial_lead = df_leads.iloc[0]['lead'] if not df_leads.empty else 0
        segments.append({
            'start_action': 1,
            'end_action': None,
            'team1_five': [player_names.get((1, pid), f"Player {pid} (Team 1)") 
                          for _, pid in sorted(current_five[1])],
            'team2_five': [player_names.get((2, pid), f"Player {pid} (Team 2)") 
                          for _, pid in sorted(current_five[2])],
            'start_lead': initial_lead,
            'end_lead': None
        })
    
    # Process substitutions
    pending_subs = {1: False, 2: False}
    
    for _, sub in df_subs.iterrows():
        team_id = sub['team_id']
        player_id = sub['player_id']
        sub_type = sub['sub_type']
        action_num = sub['action_number']
        current_lead = sub['lead']
        
        # Update the current five using (team_id, player_id) tuple
        if sub_type == 'in':
            current_five[team_id].add((team_id, player_id))
            pending_subs[team_id] = True
        elif sub_type == 'out':
            current_five[team_id].discard((team_id, player_id))
            pending_subs[team_id] = True
        
        # Check if we have complete fives after substitutions
        if len(current_five[team_id]) == 5 and pending_subs[team_id]:
            pending_subs[team_id] = False
            
            # If both teams have complete fives, create new segment
            if len(current_five[1]) == 5 and len(current_five[2]) == 5:
                # Close previous segment
                if segments:
                    segments[-1]['end_action'] = action_num - 1
                    segments[-1]['end_lead'] = current_lead
                
                # Start new segment
                segments.append({
                    'start_action': action_num,
                    'end_action': None,
                    'team1_five': [player_names.get((1, pid), f"Player {pid} (Team 1)") 
                                 for _, pid in sorted(current_five[1])],
                    'team2_five': [player_names.get((2, pid), f"Player {pid} (Team 2)") 
                                 for _, pid in sorted(current_five[2])],
                    'start_lead': current_lead,
                    'end_lead': None
                })
    
    # Close the last segment
    if segments:
        segments[-1]['end_action'] = last_action
        segments[-1]['end_lead'] = last_action_data['final_lead']
    
    return segments

def display_five_player_segments():
    st.title("📊 Five Player Segments Analysis")

    # Add timestamp and user info
    st.markdown(f"*Analysis generated on: 2025-03-25 21:41:55 UTC*")
    st.markdown(f"*Generated by: Dodga010*")

    st.subheader("🏀 Select Match")
    match_dict = fetch_matches()
    selected_match_name = st.selectbox("Select a match:", list(match_dict.keys()))
    selected_match = match_dict[selected_match_name]
    
    if selected_match:
        # Get five player segments
        segments = analyze_five_player_segments(selected_match)
        
        if not segments:
            st.warning("No complete five-player segments found for this game.")
            return
        
        # Convert segments to DataFrame
        df_segments = pd.DataFrame(segments)
        
        # Calculate duration and plus/minus for each segment
        df_segments['Duration'] = df_segments['end_action'] - df_segments['start_action'] + 1
        df_segments['Plus_Minus'] = df_segments['end_lead'] - df_segments['start_lead']
        
        # Format five player combinations as strings
        df_segments['Team 1 Five'] = df_segments['team1_five'].apply(lambda x: '\n'.join(x))
        df_segments['Team 2 Five'] = df_segments['team2_five'].apply(lambda x: '\n'.join(x))
        
        # Display segments
        st.subheader("🏃‍♂️ Five Player Segments")
        
        display_df = df_segments[[
            'start_action',
            'end_action',
            'Duration',
            'Team 1 Five',
            'Team 2 Five',
            'start_lead',
            'end_lead',
            'Plus_Minus'
        ]].copy()
        
        st.dataframe(display_df, hide_index=True)
        
        # Display summary statistics for each team
        st.subheader("📊 Team Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Team 1 (Home) Lineups")
            
            # Aggregate stats for Team 1 lineups
            team1_stats = (df_segments.groupby('Team 1 Five')
                         .agg({
                             'Duration': 'sum',
                             'Plus_Minus': 'sum'
                         })
                         .sort_values('Duration', ascending=False))
            
            team1_stats = team1_stats.reset_index()
            team1_stats.columns = ['Five Players', 'Actions Played', 'Plus/Minus']
            st.dataframe(team1_stats, hide_index=True)
        
        with col2:
            st.write("### Team 2 (Away) Lineups")
            
            # Aggregate stats for Team 2 lineups
            team2_stats = (df_segments.groupby('Team 2 Five')
                         .agg({
                             'Duration': 'sum',
                             'Plus_Minus': lambda x: -x.sum()  # Invert plus/minus for away team
                         })
                         .sort_values('Duration', ascending=False))
            
            team2_stats = team2_stats.reset_index()
            team2_stats.columns = ['Five Players', 'Actions Played', 'Plus/Minus']
            st.dataframe(team2_stats, hide_index=True)
        
        # Add most effective lineups
        st.subheader("🌟 Most Effective Lineups")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Team 1 (Home) - By Plus/Minus")
            team1_effective = team1_stats.sort_values('Plus/Minus', ascending=False)
            st.dataframe(team1_effective, hide_index=True)
        
        with col2:
            st.write("### Team 2 (Away) - By Plus/Minus")
            team2_effective = team2_stats.sort_values('Plus/Minus', ascending=False)
            st.dataframe(team2_effective, hide_index=True)

def get_teams():
    """Get list of all teams from the database."""
    conn = sqlite3.connect(db_path)
    query = """
    SELECT DISTINCT team_id, name 
    FROM Teams 
    ORDER BY team_id;
    """
    teams = pd.read_sql_query(query, conn)
    conn.close()
    return dict(zip(teams['team_id'], teams['name']))

from collections import defaultdict

from datetime import datetime

def analyze_all_team_lineups():
    """Analyze all lineups across all games."""
    conn = sqlite3.connect(db_path)
    
    # Get all games
    games_query = "SELECT DISTINCT game_id FROM PlayByPlay;"
    games = pd.read_sql_query(games_query, conn)
    
    # Get all teams
    teams_query = """
    SELECT DISTINCT name 
    FROM Teams 
    ORDER BY name;
    """
    teams = pd.read_sql_query(teams_query, conn)["name"].tolist()
    conn.close()

    all_lineup_stats = []
    all_players = set()
    # Enhanced player impact tracking with possessions
    player_impact_stats = defaultdict(lambda: {
        'total_plus_minus': 0, 
        'total_actions': 0, 
        'total_possessions': 0,
        'lineups': 0,
        'weighted_plus_minus': 0
    })
    lineup_games = defaultdict(set)
    lineup_teams = defaultdict(set)
    
    for game_id in games['game_id']:
        segments = analyze_five_player_segments(game_id)
        if segments:
            df_segments = pd.DataFrame(segments)
            
            # Get team names for this game
            conn = sqlite3.connect(db_path)
            team_query = """
            SELECT tm, name 
            FROM Teams 
            WHERE game_id = ? 
            ORDER BY tm;
            """
            game_teams = pd.read_sql_query(team_query, conn, params=(game_id,))
            conn.close()
            
            team_dict = dict(zip(game_teams['tm'], game_teams['name']))
            
            # Calculate Duration
            df_segments['Duration'] = df_segments['end_action'] - df_segments['start_action'] + 1
            
            # Process both home and away lineups
            for team_num in [1, 2]:
                team_name = team_dict.get(team_num, f"Team {team_num}")
                team_stats = df_segments.copy()
                
                # Create lineup string
                team_stats['lineup_str'] = team_stats[f'team{team_num}_five'].apply(
                    lambda x: ' | '.join(sorted(x)) if isinstance(x, list) else ''
                )
                
                # Calculate plus/minus
                team_stats['net_points'] = team_stats['end_lead'] - team_stats['start_lead']
                if team_num == 2:
                    team_stats['net_points'] = -team_stats['net_points']
                
                # Calculate estimated possessions for each segment
                team_stats['possessions'] = (team_stats['Duration'] / 2.5).round(1)
                
                # Group by lineup for this game
                game_lineup_stats = team_stats.groupby('lineup_str').agg({
                    'Duration': 'sum',
                    'net_points': 'sum',
                    'possessions': 'sum'
                }).reset_index()
                
                game_lineup_stats['team_name'] = team_name
                game_lineup_stats['game_id'] = game_id
                
                # Update tracking
                for _, row in team_stats.iterrows():
                    if row['lineup_str']:  # Only process valid lineups
                        lineup_str = row['lineup_str']
                        lineup_games[lineup_str].add(game_id)
                        lineup_teams[lineup_str].add(team_name)
                        
                        for player in row[f'team{team_num}_five']:
                            if isinstance(player, str):  # Only process valid player names
                                all_players.add(player)
                                # Track raw stats
                                player_impact_stats[player]['total_plus_minus'] += row['net_points']
                                player_impact_stats[player]['total_actions'] += row['Duration']
                                player_impact_stats[player]['total_possessions'] += row['possessions']
                                player_impact_stats[player]['lineups'] += 1
                                # Track possession-weighted plus/minus
                                player_impact_stats[player]['weighted_plus_minus'] += row['net_points'] * row['possessions']
                
                all_lineup_stats.append(game_lineup_stats)
    
    if not all_lineup_stats:
        return None, [], {}, [], len(games)
    
    # Combine all games stats
    all_stats = pd.concat(all_lineup_stats, ignore_index=True)
    
    # Calculate aggregate statistics
    lineup_stats = all_stats.groupby(['lineup_str', 'team_name']).agg({
        'Duration': 'sum',
        'net_points': 'sum',
        'possessions': 'sum'
    }).reset_index()
    
    # Add games played and team info
    lineup_stats['Games Played'] = lineup_stats['lineup_str'].apply(lambda x: len(lineup_games[x]))
    lineup_stats['Teams'] = lineup_stats['lineup_str'].apply(lambda x: ' | '.join(sorted(lineup_teams[x])))
    
    # Rename net_points to Plus/Minus
    lineup_stats = lineup_stats.rename(columns={'net_points': 'Plus/Minus'})
    
    # Calculate additional metrics
    lineup_stats['Reliability'] = round(lineup_stats['Duration'] / lineup_stats['Duration'].max() * 100, 1)
    lineup_stats['Avg Plus/Minus per Game'] = round(lineup_stats['Plus/Minus'] / lineup_stats['Games Played'], 2)
    
    # Use pre-calculated possessions instead of re-calculating
    lineup_stats['Estimated Possessions'] = lineup_stats['possessions'].round(1)
    
    # Calculate Plus/Minus per 100 possessions
    lineup_stats['Plus/Minus per 100'] = round(lineup_stats['Plus/Minus'] / lineup_stats['Estimated Possessions'] * 100, 2)
    
    # Finalize player impact stats with weighted metrics
    player_stats = []
    for player, stats in player_impact_stats.items():
        if stats['total_possessions'] > 0:
            # Calculate metrics without using .round() method
            raw_pm_per_100 = round((stats['total_plus_minus'] / stats['total_possessions'] * 100), 2)
            weighted_pm_per_100 = round((stats['weighted_plus_minus'] / stats['total_possessions']), 2)
            impact_score = round((weighted_pm_per_100 * (stats['total_possessions'] / 100)), 2)
            
            player_stats.append({
                'player_name': player,
                'total_actions': stats['total_actions'],
                'total_possessions': round(stats['total_possessions'], 1),
                'total_plus_minus': round(stats['total_plus_minus'], 1),
                'total_lineups': stats['lineups'],
                'raw_plus_minus_per_100': raw_pm_per_100,
                'weighted_plus_minus_per_100': weighted_pm_per_100,
                'impact_score': impact_score
            })
    
    # Convert player stats to DataFrame and sort
    player_impact_df = pd.DataFrame(player_stats)
    if not player_impact_df.empty:
        player_impact_df = player_impact_df.sort_values('weighted_plus_minus_per_100', ascending=False)
    
    # Rename columns for clarity
    lineup_stats = lineup_stats.rename(columns={
        'lineup_str': 'Lineup',
        'team_name': 'Primary Team',
        'Duration': 'Total Actions'
    })
    
    # Remove the raw possessions column
    lineup_stats = lineup_stats.drop(columns=['possessions'])
    
    # Arrange columns in desired order
    lineup_stats = lineup_stats[[
        'Lineup', 
        'Primary Team', 
        'Teams',
        'Total Actions',
        'Estimated Possessions',
        'Games Played',
        'Plus/Minus',
        'Avg Plus/Minus per Game',
        'Plus/Minus per 100',
        'Reliability'
    ]]
    
    return lineup_stats, sorted(list(all_players)), player_impact_df, teams, len(games)

def analyze_lineup_patterns(lineup_stats, player_impact_stats):
    """Analyze patterns in lineup performance to identify good and bad combinations."""
    
    # Convert lineup strings to player combinations
    def get_player_pairs(lineup):
        players = lineup.split(' | ')
        return list(combinations(players, 2))
    
    # Track pair statistics
    pair_stats = defaultdict(lambda: {
        'total_plus_minus': 0,
        'total_actions': 0,
        'games_together': 0,
        'lineups_together': 0,
        'avg_reliability': 0
    })
    
    # Analyze each lineup
    for _, row in lineup_stats.iterrows():
        pairs = get_player_pairs(row['Lineup'])
        for pair in pairs:
            pair_key = tuple(sorted(pair))
            pair_stats[pair_key]['total_plus_minus'] += row['Plus/Minus']
            pair_stats[pair_key]['total_actions'] += row['Total Actions']
            pair_stats[pair_key]['games_together'] += row['Games Played']
            pair_stats[pair_key]['lineups_together'] += 1
            pair_stats[pair_key]['avg_reliability'] += row['Reliability']
    
    # Calculate pair effectiveness
    pair_analysis = []
    for pair, stats in pair_stats.items():
        if stats['total_actions'] > 0:
            avg_plus_minus_per_100 = (stats['total_plus_minus'] / stats['total_actions'] * 100)
            avg_reliability = stats['avg_reliability'] / stats['lineups_together']
            
            pair_analysis.append({
                'Player 1': pair[0],
                'Player 2': pair[1],
                'Games Together': stats['games_together'],
                'Total Actions': stats['total_actions'],
                'Plus/Minus per 100': round(avg_plus_minus_per_100, 2),
                'Avg Reliability': round(avg_reliability, 1), 
                'Number of Lineups': stats['lineups_together'],
                'Recommendation': 'Strong Pair' if avg_plus_minus_per_100 > 3 and avg_reliability > 35
                                else 'Avoid Pairing' if avg_plus_minus_per_100 < -3 and stats['games_together'] > 3
                                else 'Neutral'
            })
    
    return pd.DataFrame(pair_analysis)

def identify_lineup_patterns(lineup_stats):
    """Identify patterns in successful and unsuccessful lineups."""
    
    # Calculate average performance metrics
    avg_plus_minus = lineup_stats['Plus/Minus per 100'].mean()
    std_plus_minus = lineup_stats['Plus/Minus per 100'].std()
    
    patterns = {
        'strong_lineups': lineup_stats[
            (lineup_stats['Plus/Minus per 100'] > (avg_plus_minus + std_plus_minus)) &
            (lineup_stats['Games Played'] >= 3) &
            (lineup_stats['Reliability'] > 50)
        ],
        'weak_lineups': lineup_stats[
            (lineup_stats['Plus/Minus per 100'] < (avg_plus_minus - std_plus_minus)) &
            (lineup_stats['Games Played'] >= 3) &
            (lineup_stats['Reliability'] > 50)
        ],
        'high_potential': lineup_stats[
            (lineup_stats['Plus/Minus per 100'] > avg_plus_minus) &
            (lineup_stats['Games Played'] < 3) &
            (lineup_stats['Reliability'] > 70)
        ]
    }
    
    return patterns

def display_pattern_analysis(lineup_stats, player_impact_stats):
    """Display pattern analysis in the Streamlit interface."""
    st.subheader("🔍 Pattern Analysis")
    
    # Analyze player pairs
    pair_analysis = analyze_lineup_patterns(lineup_stats, player_impact_stats)
    
    # Display strong pairs
    st.write("### 💪 Strong Player Combinations")
    strong_pairs = pair_analysis[pair_analysis['Recommendation'] == 'Strong Pair'].sort_values(
        'Plus/Minus per 100', ascending=False
    )
    if not strong_pairs.empty:
        st.dataframe(strong_pairs)
    else:
        st.info("No strong pairs identified yet.")
    
    # Display pairs to avoid
    st.write("### ⚠️ Combinations to Avoid")
    weak_pairs = pair_analysis[pair_analysis['Recommendation'] == 'Avoid Pairing'].sort_values(
        'Plus/Minus per 100'
    )
    if not weak_pairs.empty:
        st.dataframe(weak_pairs)
    else:
        st.info("No clearly negative pairs identified.")
    
    # Get lineup patterns
    patterns = identify_lineup_patterns(lineup_stats)
    
    # Display strong lineups
    st.write("### 🏆 Most Effective Lineups")
    if not patterns['strong_lineups'].empty:
        st.dataframe(patterns['strong_lineups'])
    else:
        st.info("No consistently strong lineups identified yet.")
    
    # Display lineups to avoid
    st.write("### ⛔ Least Effective Lineups")
    if not patterns['weak_lineups'].empty:
        st.dataframe(patterns['weak_lineups'])
    else:
        st.info("No consistently weak lineups identified yet.")
    
    # Display high potential lineups
    st.write("### 🌟 High Potential Lineups (Limited Sample)")
    if not patterns['high_potential'].empty:
        st.dataframe(patterns['high_potential'])
    else:
        st.info("No high potential lineups identified yet.")
    
    # Add this to your existing display_team_analysis function
    st.subheader("📊 Pattern Analysis")
    
    # Create visualization of pair effectiveness
    fig = px.scatter(pair_analysis,
                    x='Avg Reliability',
                    y='Plus/Minus per 100',
                    size='Games Together',
                    color='Recommendation',
                    hover_data=['Player 1', 'Player 2', 'Number of Lineups'],
                    title='Player Pair Effectiveness')
    
    st.plotly_chart(fig)
    
def analyze_player_impact(lineup_stats):
    """Analyze how each player affects different lineups."""
    
    # Create a dictionary to store player impact metrics
    player_impacts = defaultdict(lambda: {
        'total_lineups': 0,
        'positive_impact_lineups': 0,
        'negative_impact_lineups': 0,
        'total_plus_minus': 0,
        'total_actions': 0,
        'lineup_effects': [],
        'worst_combinations': [],
        'best_combinations': []
    })
    
    # Analyze each lineup
    for _, row in lineup_stats.iterrows():
        players = row['Lineup'].split(' | ')
        plus_minus_per_100 = row['Plus/Minus per 100']
        reliability = row['Reliability']
        games = row['Games Played']
        
        # Only consider lineups with sufficient sample size
        if games >= 3 and reliability >= 30:
            # Analyze impact of each player in the lineup
            for player in players:
                player_impacts[player]['total_lineups'] += 1
                player_impacts[player]['total_plus_minus'] += plus_minus_per_100
                player_impacts[player]['total_actions'] += row['Total Actions']
                
                # Record lineup effect
                other_players = [p for p in players if p != player]
                effect = {
                    'other_players': other_players,
                    'plus_minus': plus_minus_per_100,
                    'games': games,
                    'reliability': reliability
                }
                player_impacts[player]['lineup_effects'].append(effect)
                
                # Track positive/negative impact
                if plus_minus_per_100 > 0:
                    player_impacts[player]['positive_impact_lineups'] += 1
                elif plus_minus_per_100 < 0:
                    player_impacts[player]['negative_impact_lineups'] += 1
    
    # Calculate final metrics and find problematic combinations
    impact_analysis = []
    for player, stats in player_impacts.items():
        if stats['total_lineups'] > 0:
            # Calculate average impact
            avg_impact = stats['total_plus_minus'] / stats['total_lineups']
            
            # Find worst and best combinations
            lineup_effects = stats['lineup_effects']
            sorted_effects = sorted(lineup_effects, key=lambda x: x['plus_minus'])
            
            worst_combos = sorted_effects[:3] if len(sorted_effects) >= 3 else sorted_effects
            best_combos = sorted_effects[-3:] if len(sorted_effects) >= 3 else []
            
            impact_analysis.append({
                'Player': player,
                'Avg Impact per 100': round(avg_impact, 2),
                'Total Lineups': stats['total_lineups'],
                'Positive Impact %': round(stats['positive_impact_lineups'] / stats['total_lineups'] * 100, 1),
                'Negative Impact %': round(stats['negative_impact_lineups'] / stats['total_lineups'] * 100, 1),
                'Total Actions': stats['total_actions'],
                'Worst Combinations': [
                    f"{', '.join(combo['other_players'])} ({combo['plus_minus']:.1f})"
                    for combo in worst_combos
                ],
                'Best Combinations': [
                    f"{', '.join(combo['other_players'])} ({combo['plus_minus']:.1f})"
                    for combo in reversed(best_combos)
                ]
            })
    
    return pd.DataFrame(impact_analysis)

def display_player_impact_analysis(lineup_stats):
    """Display detailed player impact analysis."""
    st.subheader("🔍 Player Impact Analysis")
    
    impact_df = analyze_player_impact(lineup_stats)
    
    # Sort by average impact
    impact_df_sorted = impact_df.sort_values('Avg Impact per 100', ascending=True)
    
    # Display problematic players
    st.write("### ⚠️ Players with Potential Negative Impact")
    negative_impact = impact_df_sorted[
        (impact_df_sorted['Avg Impact per 100'] < 0) & 
        (impact_df_sorted['Total Lineups'] >= 3)
    ]
    
    if not negative_impact.empty:
        for _, player in negative_impact.iterrows():
            st.write(f"**{player['Player']}**")
            st.write(f"- Average Impact per 100 possessions: {player['Avg Impact per 100']:.1f}")
            st.write(f"- Negative Impact in {player['Negative Impact %']}% of lineups")
            st.write("Worst lineup combinations:")
            for combo in player['Worst Combinations'][:3]:
                st.write(f"  • With {combo}")
            st.write("---")
    else:
        st.info("No consistently negative impact players identified.")
    
    # Visualization of player impact
    st.subheader("📊 Player Impact Distribution")
    
    fig = px.scatter(impact_df,
                    x='Total Lineups',
                    y='Avg Impact per 100',
                    size='Total Actions',
                    color='Positive Impact %',
                    hover_data=['Player', 'Negative Impact %'],
                    title='Player Impact Analysis')
    
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig)
    
    # Detailed statistics table
    st.write("### 📋 Detailed Player Statistics")
    detailed_stats = impact_df[[
        'Player', 
        'Avg Impact per 100', 
        'Total Lineups',
        'Positive Impact %',
        'Negative Impact %',
        'Total Actions'
    ]].sort_values('Total Lineups', ascending=False)
    
    st.dataframe(detailed_stats)
    
    # Player compatibility matrix
    st.write("### 🔄 Player Compatibility Analysis")
    compatibility_data = []
    
    for _, row in lineup_stats.iterrows():
        players = row['Lineup'].split(' | ')
        plus_minus = row['Plus/Minus per 100']
        games = row['Games Played']
        
        if games >= 3:  # Only consider lineups with sufficient sample size
            for p1, p2 in combinations(players, 2):
                compatibility_data.append({
                    'Player 1': p1,
                    'Player 2': p2,
                    'Plus/Minus': plus_minus,
                    'Games': games
                })
    
    if compatibility_data:
        compat_df = pd.DataFrame(compatibility_data)
        pivot_compat = compat_df.groupby(['Player 1', 'Player 2'])['Plus/Minus'].mean().reset_index()
        
        # Create heatmap
        players_list = sorted(list(set(pivot_compat['Player 1'].unique()) | set(pivot_compat['Player 2'].unique())))
        compat_matrix = pd.DataFrame(0, index=players_list, columns=players_list)
        
        for _, row in pivot_compat.iterrows():
            compat_matrix.loc[row['Player 1'], row['Player 2']] = row['Plus/Minus']
            compat_matrix.loc[row['Player 2'], row['Player 1']] = row['Plus/Minus']
        
        fig = px.imshow(compat_matrix,
                       labels=dict(x="Player 2", y="Player 1", color="Plus/Minus per 100"),
                       title="Player Compatibility Heatmap")
        st.plotly_chart(fig)


    
    
def display_team_analysis():
    """Display team lineup analysis with proper formatting."""
    st.title("📊 Season Lineup Analysis")
    
    # Format current date/time in UTC
    current_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    current_user = "Dodga010"  # Replace with your actual user system
    # Display metadata with proper formatting
    st.markdown(f"""
    *Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {current_time}*  
    *Current User's Login: {current_user}*
    """)
    
    # Get analysis data
    stats, all_players, player_impact, teams, total_games = analyze_all_team_lineups()
    
    if stats is None:
        st.warning("No lineup data available.")
        return
    
    st.info(f"Analyzing data from {total_games} total games")
    
    # Add filters in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_team = st.selectbox(
            "Filter by Team",
            ["All Teams"] + teams,
            help="Show lineups for a specific team"
        )
    
    with col2:
        min_actions = st.slider(
            "Minimum Actions",
            min_value=0,
            max_value=int(stats['Total Actions'].max()),
            value=20,
            help="Filter out lineups with fewer actions"
        )
    
    with col3:
        min_games = st.slider(
            "Minimum Games",
            min_value=1,
            max_value=int(stats['Games Played'].max()),
            value=2,
            help="Filter lineups based on minimum games played"
        )
    
    # Apply filters
    filtered_stats = stats[
        (stats['Total Actions'] >= min_actions) & 
        (stats['Games Played'] >= min_games)
    ]
    
    if selected_team != "All Teams":
        filtered_stats = filtered_stats[
            (filtered_stats['Primary Team'] == selected_team) | 
            (filtered_stats['Teams'].str.contains(selected_team))
        ]
    
    if filtered_stats.empty:
        st.warning("No lineups found matching the selected criteria.")
        return
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by",
        ["Plus/Minus per 100", "Total Actions", "Games Played", "Reliability"]
    )
    
    sorted_stats = filtered_stats.sort_values(sort_by, ascending=False)
    
    # Display main stats table
    st.subheader("🏀 Lineup Statistics")
    st.dataframe(sorted_stats, hide_index=True)
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Lineups",
            len(filtered_stats)
        )
    with col2:
        st.metric(
            "Avg Games per Lineup",
            f"{filtered_stats['Games Played'].mean():.1f}"
        )
    with col3:
        st.metric(
            "Avg Plus/Minus per 100",
            f"{filtered_stats['Plus/Minus per 100'].mean():.1f}"
        )
    with col4:
        st.metric(
            "Most Used Lineup Actions",
            int(filtered_stats['Total Actions'].max())
        )
    
    # Add effectiveness visualization
    st.subheader("📈 Lineup Effectiveness")
    
    # Create scatter plot
    fig = px.scatter(
        filtered_stats,
        x='Reliability',
        y='Plus/Minus per 100',
        size='Games Played',
        color='Primary Team',
        hover_data=['Lineup', 'Games Played', 'Total Actions'],
        title='Lineup Effectiveness vs Reliability'
    )
    
    st.plotly_chart(fig)
    
    # Add Player Impact Analysis
    st.header("🏀 Player Impact Analysis (Weighted by Possessions)")

    # Check if we have player impact data to display
    if isinstance(player_impact, pd.DataFrame) and not player_impact.empty:
        # Create tabs for different views - now with three tabs
        impact_tab1, impact_tab2, impact_tab3 = st.tabs(["Overall Impact", "Detail View", "Team Analysis"])
        
        # Tab 1: Overall Impact (unchanged)
        with impact_tab1:
            st.subheader("Top Players by Weighted Impact")
            
            # Display top 15 players by weighted impact
            top_players = player_impact.nlargest(15, 'weighted_plus_minus_per_100')
            
            # Format DataFrame for display
            display_cols = [
                'player_name', 'total_possessions', 'weighted_plus_minus_per_100', 'impact_score'
            ]
            
            # Create formatted DataFrame with renamed columns
            display_df = top_players[display_cols].copy()
            display_df.columns = [
                'Player', 'Possessions', 'Plus/Minus per 100 Poss.', 'Impact Score'
            ]
            
            # Display formatted table
            st.dataframe(
                display_df.style.format({
                    'Possessions': '{:.0f}',
                    'Plus/Minus per 100 Poss.': '{:.2f}',
                    'Impact Score': '{:.2f}'
                }),
                height=400
            )
            
            # Add visualization - bar chart of player impact
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot top 10 players
            plot_data = top_players.head(10)
            sns.barplot(
                x='weighted_plus_minus_per_100', 
                y='player_name',
                data=plot_data,
                ax=ax
            )
            
            ax.set_title('Top 10 Players by Weighted Plus/Minus per 100 Possessions')
            ax.set_xlabel('Plus/Minus per 100 Possessions (Weighted)')
            ax.set_ylabel('Player')
            
            # Add data labels
            for i, v in enumerate(plot_data['weighted_plus_minus_per_100']):
                ax.text(v + 0.1, i, f'{v:.2f}', va='center')
                
            st.pyplot(fig)
        
        # Tab 2: Detail View (unchanged)  
        with impact_tab2:
            st.subheader("Detailed Player Impact Data")
            
            # Create two columns for filters
            filter_col1, filter_col2 = st.columns(2)
            
            with filter_col1:
                # Add dropdown for player selection instead of text search
                if not player_impact.empty:
                    player_list = ["All Players"] + sorted(player_impact["player_name"].unique().tolist())
                    selected_player = st.selectbox("Select a player:", player_list)
                else:
                    selected_player = "All Players"
            
            with filter_col2:
                # Add minimum possessions filter
                min_possessions = st.slider("Minimum Possessions:", 
                                          min_value=0, 
                                          max_value=int(player_impact['total_possessions'].max() if not player_impact.empty else 0), 
                                          value=50)
            
            # Apply filters
            filtered_players = player_impact.copy()
            if selected_player != "All Players":
                filtered_players = filtered_players[filtered_players['player_name'] == selected_player]
            filtered_players = filtered_players[filtered_players['total_possessions'] >= min_possessions]
            
            # Display filtered data
            if not filtered_players.empty:
                # All columns for detailed view
                detail_cols = [
                    'player_name', 'total_possessions', 'total_plus_minus', 
                    'raw_plus_minus_per_100', 'weighted_plus_minus_per_100', 'impact_score', 'total_lineups'
                ]
                
                # Renamed columns for display
                detail_display = filtered_players[detail_cols].copy()
                detail_display.columns = [
                    'Player', 'Possessions', 'Total +/-', 
                    'Raw +/- per 100', 'Weighted +/- per 100', 'Impact Score', 'Lineups'
                ]
                
                st.dataframe(
                    detail_display.style.format({
                        'Possessions': '{:.0f}',
                        'Total +/-': '{:.1f}',
                        'Raw +/- per 100': '{:.2f}',
                        'Weighted +/- per 100': '{:.2f}',
                        'Impact Score': '{:.2f}',
                        'Lineups': '{:.0f}'
                    }),
                    height=500
                )
                
                # Show detailed player information if a specific player is selected
                if selected_player != "All Players":
                    st.subheader(f"📊 {selected_player} Details")
                    
                    # Create 3 columns for player metrics
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    # Get the single player data
                    player_data = filtered_players.iloc[0]
                    
                    with metric_col1:
                        st.metric(
                            "Impact Score",
                            f"{player_data['impact_score']:.2f}"
                        )
                    with metric_col2:
                        st.metric(
                            "Weighted +/- per 100",
                            f"{player_data['weighted_plus_minus_per_100']:.2f}"
                        )
                    with metric_col3:
                        st.metric(
                            "Total Possessions",
                            f"{int(player_data['total_possessions'])}"
                        )
                    
                    # Show team-specific impact for this player
                    st.subheader("🏀 Team Impact Breakdown")
                    
                    # Find lineups containing this player
                    player_lineups = stats[stats['Lineup'].str.contains(selected_player)].copy()
                    
                    if not player_lineups.empty:
                        # Group by team to get team-specific stats
                        team_impact = []
                        
                        # Process Primary Team stats
                        team_groups = player_lineups.groupby('Primary Team')
                        
                        for team, team_data in team_groups:
                            # Calculate weighted metrics for this team
                            total_poss = team_data['Estimated Possessions'].sum()
                            weighted_pm = sum(team_data['Plus/Minus'] * team_data['Estimated Possessions']) / total_poss if total_poss > 0 else 0
                            
                            team_impact.append({
                                'Team': team,
                                'Games': team_data['Games Played'].sum(),
                                'Possessions': total_poss,
                                'Lineups': len(team_data),
                                'Plus/Minus': team_data['Plus/Minus'].sum(),
                                'Weighted PM per 100': round(weighted_pm * 100, 2)
                            })
                        
                        # Convert to DataFrame and display
                        team_impact_df = pd.DataFrame(team_impact)
                        st.dataframe(
                            team_impact_df.style.format({
                                'Games': '{:.0f}',
                                'Possessions': '{:.0f}',
                                'Lineups': '{:.0f}',
                                'Plus/Minus': '{:.1f}',
                                'Weighted PM per 100': '{:.2f}'
                            }),
                            height=300
                        )
                        
                        # Show the specific lineups this player is part of
                        st.subheader("Best Lineups with " + selected_player)
                        best_lineups = player_lineups.sort_values('Plus/Minus per 100', ascending=False).head(5)
                        
                        # Display the top lineups in a clean format
                        for idx, lineup in best_lineups.iterrows():
                            st.markdown(f"""
                            **Lineup:** {lineup['Lineup']}  
                            **Team:** {lineup['Primary Team']}  
                            **+/- per 100:** {lineup['Plus/Minus per 100']:.2f} | **Possessions:** {lineup['Estimated Possessions']:.0f} | **Games:** {lineup['Games Played']}
                            """)
                            st.divider()
                    else:
                        st.info(f"No lineup data found for {selected_player}")
            else:
                st.warning("No players match the selected filters.")
                
            # Explanation of metrics
            with st.expander("About These Metrics"):
                st.markdown("""
                ### Understanding Player Impact Metrics
                
                - **Possessions**: Total estimated possessions the player was on court
                - **Total +/-**: Raw plus-minus across all lineups
                - **Raw +/- per 100**: Simple plus-minus per 100 possessions
                - **Weighted +/- per 100**: Plus-minus per 100 possessions weighted by possession time
                - **Impact Score**: Weighted impact accounting for both efficiency and volume
                - **Lineups**: Number of different 5-player combinations the player was part of
                
                The **weighted metrics** give more importance to longer lineup stints, providing a more accurate picture of player impact.
                
                The **Team Impact Breakdown** shows how the player performs with different teams, calculated by finding all lineups containing the player and grouping them by team.
                """)
        
                    # NEW: Tab 3 - Team Analysis
            with impact_tab3:
                st.subheader("Team-Specific Analysis")
                
                # Team selection dropdown
                analysis_team = st.selectbox(
                    "Select Team for Analysis:",
                    teams,
                    key="team_analysis_selector"
                )
                
                # Get all lineups for the selected team
                team_lineups = stats[
                    (stats['Primary Team'] == analysis_team) | 
                    (stats['Teams'].str.contains(analysis_team))
                ].copy()
                
                if team_lineups.empty:
                    st.warning(f"No lineup data available for {analysis_team}")
                else:
                    # Team overview metrics
                    st.subheader(f"📊 {analysis_team} Overview")
                    
                    # Calculate team summary metrics
                    team_metrics_col1, team_metrics_col2, team_metrics_col3, team_metrics_col4 = st.columns(4)
                    
                    with team_metrics_col1:
                        st.metric(
                            "Total Lineups",
                            len(team_lineups)
                        )
                    with team_metrics_col2:
                        st.metric(
                            "Avg Plus/Minus per 100",
                            f"{team_lineups['Plus/Minus per 100'].mean():.2f}"
                        )
                    with team_metrics_col3:
                        avg_actions = team_lineups['Total Actions'].mean()
                        st.metric(
                            "Avg Actions per Lineup",
                            f"{avg_actions:.1f}"
                        )
                    with team_metrics_col4:
                        avg_poss = team_lineups['Estimated Possessions'].mean()
                        st.metric(
                            "Avg Possessions per Lineup",
                            f"{avg_poss:.1f}"
                        )
                    
                    # Best lineups for this team - with minimum possessions filter
                    st.subheader("Best Performing Lineups")

                    # Add a filter for minimum possessions
                    min_lineup_possessions = st.slider(
                        "Minimum Possessions for Lineup:",
                        min_value=10,
                        max_value=int(team_lineups['Estimated Possessions'].max()) if not team_lineups.empty else 100,
                        value=50,  # Default to 50 possessions minimum
                        help="Filter out lineups with fewer possessions to ensure statistical significance"
                    )

                    # Apply possession filter and sort by Plus/Minus per 100
                    filtered_team_lineups = team_lineups[team_lineups['Estimated Possessions'] >= min_lineup_possessions]

                    if filtered_team_lineups.empty:
                        st.warning(f"No lineups with at least {min_lineup_possessions} possessions. Try lowering the minimum.")
                    else:
                        # Sort by Plus/Minus per 100 and take top 5
                        best_team_lineups = filtered_team_lineups.sort_values('Plus/Minus per 100', ascending=False).head(5)
                        
                        # Create a bar chart of top lineups
                        fig, ax = plt.subplots(figsize=(12, 6))
                        
                        # Create shorter lineup labels for display that include possession count
                        best_team_lineups['Short Lineup'] = best_team_lineups.apply(
                            lambda x: f"{x['Lineup'].replace(' | ', '-')} ({int(x['Estimated Possessions'])} poss)",
                            axis=1
                        )
                        
                        # Sort for better visualization (tallest bars first)
                        best_team_lineups = best_team_lineups.sort_values('Plus/Minus per 100')
                        
                        # Create horizontal bar chart with colored bars based on plus/minus value
                        bars = sns.barplot(
                            x='Plus/Minus per 100',
                            y='Short Lineup',
                            data=best_team_lineups,
                            ax=ax,
                            palette='RdYlGn_r'  # Red for negative, green for positive
                        )
                        
                        # Style the chart
                        ax.set_title(f'Top 5 Lineups for {analysis_team} (Minimum {min_lineup_possessions} Possessions)')
                        ax.set_xlabel('Plus/Minus per 100 Possessions')
                        ax.set_ylabel('')  # Remove y-axis label as it's clear these are lineups
                        
                        # Add data labels
                        for i, v in enumerate(best_team_lineups['Plus/Minus per 100']):
                            ax.text(v + 0.5, i, f'{v:.2f}', va='center')
                        
                        # Set a vertical line at 0 for reference
                        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                        
                        st.pyplot(fig)
                        
                        # Also display the lineup data in a table with more details
                        st.subheader("Detailed Stats for Top Lineups")
                        
                        # Select columns for display
                        display_cols = ['Lineup', 'Estimated Possessions', 'Games Played', 'Plus/Minus', 'Plus/Minus per 100']
                        
                        # Format the table data
                        lineup_table = best_team_lineups[display_cols].copy()
                        
                        st.dataframe(
                            lineup_table.style.format({
                                'Estimated Possessions': '{:.0f}',
                                'Games Played': '{:.0f}',
                                'Plus/Minus': '{:.1f}',
                                'Plus/Minus per 100': '{:.2f}'
                            }),
                            height=250
                        )
                    
                    # Find players in this team's lineups
                    team_players = set()
                    
                    for lineup in team_lineups['Lineup']:
                        players = lineup.split(' | ')
                        for player in players:
                            team_players.add(player)
                    
                    # Get impact data for these players
                    team_player_impact = player_impact[player_impact['player_name'].isin(team_players)].copy()
                    
                    # Player performance in this team
                    st.subheader(f"Player Performance in {analysis_team}")
                    
                    if not team_player_impact.empty:
                        # Sort by weighted plus/minus
                        team_player_impact = team_player_impact.sort_values('weighted_plus_minus_per_100', ascending=False)
                        
                        # Add minimum possessions filter for players
                        min_player_possessions = st.slider(
                            "Minimum Player Possessions:",
                            min_value=10,
                            max_value=int(team_player_impact['total_possessions'].max()),
                            value=100,  # Default to 100 possessions minimum
                            help="Filter out players with limited playing time"
                        )
                        
                        # Apply filter
                        filtered_player_impact = team_player_impact[team_player_impact['total_possessions'] >= min_player_possessions]
                        
                        if filtered_player_impact.empty:
                            st.warning(f"No players with at least {min_player_possessions} possessions. Try lowering the minimum.")
                        else:
                            # Display all players from this team
                            player_display_cols = [
                                'player_name', 'total_possessions', 'weighted_plus_minus_per_100', 'impact_score', 'total_lineups'
                            ]
                            
                            player_display_df = filtered_player_impact[player_display_cols].copy()
                            player_display_df.columns = [
                                'Player', 'Possessions', 'Plus/Minus per 100', 'Impact Score', 'Lineups'
                            ]
                            
                            st.dataframe(
                                player_display_df.style.format({
                                    'Possessions': '{:.0f}',
                                    'Plus/Minus per 100': '{:.2f}',
                                    'Impact Score': '{:.2f}',
                                    'Lineups': '{:.0f}'
                                }),
                                height=400
                            )
                            
                            # Player impact distribution visualization
                            st.subheader("Player Impact Distribution")
                            
                            # Create two columns for different visualizations
                            vis_col1, vis_col2 = st.columns(2)
                            
                            with vis_col1:
                                # Create scatter plot of player impact vs possessions
                                # Fix for negative impact scores in size parameter
                                filtered_player_impact['abs_impact_score'] = filtered_player_impact['impact_score'].abs() + 5  # Add constant to ensure visibility
                                
                                fig_scatter = px.scatter(
                                    filtered_player_impact,
                                    x='total_possessions',
                                    y='weighted_plus_minus_per_100',
                                    size='abs_impact_score',  # Use absolute values for size
                                    hover_name='player_name',
                                    color='weighted_plus_minus_per_100',  # Use weighted plus/minus for color
                                    color_continuous_scale='RdYlGn',  # Red to Green color scale
                                    title=f'Player Impact vs. Playing Time ({analysis_team})',
                                    labels={
                                        'total_possessions': 'Total Possessions',
                                        'weighted_plus_minus_per_100': 'Weighted Plus/Minus per 100',
                                        'player_name': 'Player'
                                    }
                                )
                                # Add reference line at 0
                                fig_scatter.add_hline(y=0, line_width=1, line_color="black", line_dash="dash")
                                st.plotly_chart(fig_scatter)
                            
                            with vis_col2:
                                # Create a horizontal bar chart for top players
                                top_count = min(10, len(filtered_player_impact))
                                top_team_players = filtered_player_impact.head(top_count)
                                
                                # For bar chart, sort ascending so highest value is at top
                                top_team_players = top_team_players.sort_values('weighted_plus_minus_per_100')
                                
                                fig_bar = px.bar(
                                    top_team_players,
                                    y='player_name',
                                    x='weighted_plus_minus_per_100',
                                    orientation='h',
                                    title=f'Top {top_count} Players by Impact ({analysis_team})',
                                    labels={
                                        'player_name': 'Player',
                                        'weighted_plus_minus_per_100': 'Weighted Plus/Minus per 100'
                                    },
                                    color='weighted_plus_minus_per_100',
                                    color_continuous_scale='RdYlGn'  # Red to Green color scale
                                )
                                # Add reference line at 0
                                fig_bar.add_vline(x=0, line_width=1, line_color="black", line_dash="dash")
                                st.plotly_chart(fig_bar)
                            
                            # Add context for interpreting the data
                            with st.expander("Understanding Team Analysis"):
                                st.markdown("""
                                ### How to Interpret Team Analysis
                                
                                **Team Overview**: Summary metrics showing how the team performs overall.
                                
                                **Best Performing Lineups**: Shows the top lineups filtered by a minimum number of possessions. Higher plus/minus values indicate better performance.
                                
                                **Player Impact Distribution**:
                                - **Impact vs. Playing Time**: Shows each player's efficiency (y-axis) vs. volume (x-axis).
                                - **Bar Chart**: Shows the weighted plus/minus per 100 possessions for the top players.
                                
                                Players in the top right of the scatter plot are your most valuable players - they combine high efficiency with significant playing time.
                                """)
                    else:
                        st.warning("No player impact data available for this team")

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def analyze_four_factors_wins(db_path="database.db"):
    """Analyze how often teams with better four factor stats win their games."""
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Query to get game data with four factors for each team
    query = """
    SELECT 
        t1.game_id,
        t1.name AS team1,
        t2.name AS team2,
        (t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) AS score1,
        (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score) AS score2,
        
        -- Calculate four factors for team 1
        (t1.field_goals_made + 0.5 * t1.three_pointers_made) * 100.0 / NULLIF(t1.field_goals_attempted, 0) AS efg1,
        t1.turnovers * 100.0 / NULLIF((t1.field_goals_attempted + 0.44 * t1.free_throws_attempted), 0) AS tov1,
        t1.rebounds_offensive * 100.0 / NULLIF((t1.rebounds_offensive + t2.rebounds_defensive), 0) AS orb1,
        t1.free_throws_attempted * 100.0 / NULLIF(t1.field_goals_attempted, 0) AS ftr1,
        
        -- Calculate four factors for team 2
        (t2.field_goals_made + 0.5 * t2.three_pointers_made) * 100.0 / NULLIF(t2.field_goals_attempted, 0) AS efg2, 
        t2.turnovers * 100.0 / NULLIF((t2.field_goals_attempted + 0.44 * t2.free_throws_attempted), 0) AS tov2,
        t2.rebounds_offensive * 100.0 / NULLIF((t2.rebounds_offensive + t1.rebounds_defensive), 0) AS orb2,
        t2.free_throws_attempted * 100.0 / NULLIF(t2.field_goals_attempted, 0) AS ftr2
        
    FROM Teams t1
    JOIN Teams t2 ON t1.game_id = t2.game_id 
    WHERE t1.tm = 1 AND t2.tm = 2
    """
    
    # Execute query and load into DataFrame
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Drop games with null values in any four factor stat
    df = df.dropna()
    
    # Determine winner (1 for team1, 2 for team2)
    df['winner'] = (df['score1'] < df['score2']).astype(int) + 1
    
    # For each four factor, determine which team had the better stat
    # Note: For TOV%, lower is better, so we invert the comparison
    df['better_efg'] = (df['efg1'] < df['efg2']).astype(int) + 1  
    df['better_tov'] = (df['tov1'] > df['tov2']).astype(int) + 1  # Lower TOV% is better
    df['better_orb'] = (df['orb1'] < df['orb2']).astype(int) + 1
    df['better_ftr'] = (df['ftr1'] < df['ftr2']).astype(int) + 1
    
    # Calculate results for each factor
    results = {}
    
    factor_names = {
        'efg': 'Shooting (eFG%)',
        'tov': 'Turnovers (TOV%)',
        'orb': 'Offensive Rebounding (ORB%)', 
        'ftr': 'Free Throw Rate (FTR)'
    }
    
    for factor, factor_name in factor_names.items():
        # Count games where team with better factor won
        better_col = f'better_{factor}'
        win_count = sum(df[better_col] == df['winner'])
        total_count = len(df)
        win_pct = win_count / total_count * 100 if total_count > 0 else 0
        
        results[factor] = {
            'Factor': factor_name,
            'Win Count': win_count,
            'Total Games': total_count,
            'Win Percentage': win_pct
        }
    
    # Convert to DataFrame
    results_df = pd.DataFrame(list(results.values()))
    
    # Sort by win percentage
    results_df = results_df.sort_values('Win Percentage', ascending=False)
    
    return df, results_df

def display_four_factors_analysis():
    """Display the four factors win analysis in Streamlit."""
    st.title("🏀 Four Factors Win Analysis")
    st.write("""
    ## How often do teams with better Four Factor stats win their games?
    
    This analysis examines how well Dean Oliver's Four Factors of Basketball Success
    predict game outcomes. For each factor, we calculate how often the team with 
    the better stat won the game.
    """)
    
    # Get the data
    df, results_df = analyze_four_factors_wins()
    
    if df.empty:
        st.warning("No data available for analysis.")
        return
    
    # Create bar chart for win percentages
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use nicer color palette
    bar_colors = sns.color_palette("viridis", len(results_df))
    
    # Create bars
    bars = ax.bar(
        results_df['Factor'], 
        results_df['Win Percentage'], 
        color=bar_colors
    )
    
    # Add percentage labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 1,
            f'{height:.1f}%', 
            ha='center', va='bottom', 
            fontweight='bold'
        )
    
    # Add styling
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    ax.text(0.1, 51, 'Random chance (50%)', fontsize=9, color='gray')
    ax.set_ylabel('Win Percentage (%)')
    ax.set_title('Win Percentage When Team Has Better Four Factors Stats')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Display the chart
    st.pyplot(fig)
    
    # Display metrics
    st.subheader("Win Percentages by Factor")
    
    cols = st.columns(len(results_df))
    for i, (_, row) in enumerate(results_df.iterrows()):
        with cols[i]:
            st.metric(
                label=row['Factor'],
                value=f"{row['Win Percentage']:.1f}%",
                delta=f"{row['Win Count']}/{row['Total Games']} games"
            )
    
    # Display detailed table
    st.subheader("Detailed Results")
    
    # Format win percentage for display
    display_df = results_df

import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime

# Define SQLite database path
db_path = os.path.join(os.path.dirname(__file__), "database.db")

def analyze_basketball_stats_wins():
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Query to get match data with four factors and other stats for both teams in each game
    query = """
    WITH team_matches AS (
        SELECT 
            t1.game_id,
            t1.name AS team1_name,
            t2.name AS team2_name,
            (t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) AS team1_score,
            (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score) AS team2_score,
            -- Team 1 Four Factors
            ROUND((t1.field_goals_made + 0.5 * t1.three_pointers_made) * 100.0 / t1.field_goals_attempted, 2) AS team1_eFG_percentage,
            ROUND(t1.turnovers * 100.0 / (t1.field_goals_attempted + 0.44 * t1.free_throws_attempted), 2) AS team1_TOV_percentage,
            ROUND(t1.rebounds_offensive * 100.0 / (t1.rebounds_offensive + t1.rebounds_defensive), 2) AS team1_ORB_percentage,
            ROUND(t1.free_throws_attempted * 100.0 / t1.field_goals_attempted, 2) AS team1_FTR_percentage,
            -- Team 2 Four Factors
            ROUND((t2.field_goals_made + 0.5 * t2.three_pointers_made) * 100.0 / t2.field_goals_attempted, 2) AS team2_eFG_percentage,
            ROUND(t2.turnovers * 100.0 / (t2.field_goals_attempted + 0.44 * t2.free_throws_attempted), 2) AS team2_TOV_percentage,
            ROUND(t2.rebounds_offensive * 100.0 / (t2.rebounds_offensive + t2.rebounds_defensive), 2) AS team2_ORB_percentage,
            ROUND(t2.free_throws_attempted * 100.0 / t2.field_goals_attempted, 2) AS team2_FTR_percentage,
            
            -- Additional Stats - Team 1
            t1.assists AS team1_assists,
            t1.steals AS team1_steals,
            t1.blocks AS team1_blocks,
            t1.rebounds_total AS team1_rebounds,
            t1.three_pointers_made AS team1_3PM,
            t1.field_goal_percentage AS team1_FG_percentage,
            t1.three_point_percentage AS team1_3P_percentage,
            t1.free_throw_percentage AS team1_FT_percentage,
            t1.fouls_total AS team1_fouls,
            t1.biggest_scoring_run AS team1_scoring_run,
            t1.points_in_paint AS team1_points_in_paint,
            t1.points_from_turnovers AS team1_points_off_TO,
            t1.points_fast_break AS team1_fast_break,
            
            -- Additional Stats - Team 2
            t2.assists AS team2_assists,
            t2.steals AS team2_steals,
            t2.blocks AS team2_blocks,
            t2.rebounds_total AS team2_rebounds,
            t2.three_pointers_made AS team2_3PM,
            t2.field_goal_percentage AS team2_FG_percentage,
            t2.three_point_percentage AS team2_3P_percentage,
            t2.free_throw_percentage AS team2_FT_percentage,
            t2.fouls_total AS team2_fouls,
            t2.biggest_scoring_run AS team2_scoring_run,
            t2.points_in_paint AS team2_points_in_paint,
            t2.points_from_turnovers AS team2_points_off_TO,
            t2.points_fast_break AS team2_fast_break
        FROM Teams t1
        JOIN Teams t2 ON t1.game_id = t2.game_id AND t1.tm = 1 AND t2.tm = 2
    )
    SELECT * FROM team_matches
    """
    
    # Execute query and load data
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Check if we have data
    if df.empty:
        return None
        
    # Determine winner for each game
    df['winner'] = 1  # Team 1 is default winner
    df.loc[df['team2_score'] > df['team1_score'], 'winner'] = 2  # Team 2 is winner if they scored more
    
    # For games with ties (if any), consider them as no winner
    df.loc[df['team1_score'] == df['team2_score'], 'winner'] = 0
    
    # ------------- FOUR FACTORS ANALYSIS -------------
    
    # Determine which team had the advantage in each factor
    # For eFG% (higher is better)
    df['better_eFG'] = 1  # Team 1 by default
    df.loc[df['team2_eFG_percentage'] > df['team1_eFG_percentage'], 'better_eFG'] = 2
    df.loc[df['team2_eFG_percentage'] == df['team1_eFG_percentage'], 'better_eFG'] = 0  # Tie
    
    # For TOV% (lower is better)
    df['better_TOV'] = 1  # Team 1 by default
    df.loc[df['team2_TOV_percentage'] < df['team1_TOV_percentage'], 'better_TOV'] = 2
    df.loc[df['team2_TOV_percentage'] == df['team1_TOV_percentage'], 'better_TOV'] = 0  # Tie
    
    # For ORB% (higher is better)
    df['better_ORB'] = 1  # Team 1 by default
    df.loc[df['team2_ORB_percentage'] > df['team1_ORB_percentage'], 'better_ORB'] = 2
    df.loc[df['team2_ORB_percentage'] == df['team1_ORB_percentage'], 'better_ORB'] = 0  # Tie
    
    # For FTR% (higher is better)
    df['better_FTR'] = 1  # Team 1 by default
    df.loc[df['team2_FTR_percentage'] > df['team1_FTR_percentage'], 'better_FTR'] = 2
    df.loc[df['team2_FTR_percentage'] == df['team1_FTR_percentage'], 'better_FTR'] = 0  # Tie
    
    # ------------- ADDITIONAL STATS ANALYSIS -------------
    
    # For Assists (higher is better)
    df['better_AST'] = 1  # Team 1 by default
    df.loc[df['team2_assists'] > df['team1_assists'], 'better_AST'] = 2
    df.loc[df['team2_assists'] == df['team1_assists'], 'better_AST'] = 0  # Tie
    
    # For Steals (higher is better)
    df['better_STL'] = 1  # Team 1 by default
    df.loc[df['team2_steals'] > df['team1_steals'], 'better_STL'] = 2
    df.loc[df['team2_steals'] == df['team1_steals'], 'better_STL'] = 0  # Tie
    
    # For Blocks (higher is better)
    df['better_BLK'] = 1  # Team 1 by default
    df.loc[df['team2_blocks'] > df['team1_blocks'], 'better_BLK'] = 2
    df.loc[df['team2_blocks'] == df['team1_blocks'], 'better_BLK'] = 0  # Tie
    
    # For Total Rebounds (higher is better)
    df['better_REB'] = 1  # Team 1 by default
    df.loc[df['team2_rebounds'] > df['team1_rebounds'], 'better_REB'] = 2
    df.loc[df['team2_rebounds'] == df['team1_rebounds'], 'better_REB'] = 0  # Tie
    
    # For 3-Pointers Made (higher is better)
    df['better_3PM'] = 1  # Team 1 by default
    df.loc[df['team2_3PM'] > df['team1_3PM'], 'better_3PM'] = 2
    df.loc[df['team2_3PM'] == df['team1_3PM'], 'better_3PM'] = 0  # Tie
    
    # For Field Goal Percentage (higher is better)
    df['better_FG_PCT'] = 1  # Team 1 by default
    df.loc[df['team2_FG_percentage'] > df['team1_FG_percentage'], 'better_FG_PCT'] = 2
    df.loc[df['team2_FG_percentage'] == df['team1_FG_percentage'], 'better_FG_PCT'] = 0  # Tie
    
    # For 3-Point Percentage (higher is better)
    df['better_3P_PCT'] = 1  # Team 1 by default
    df.loc[df['team2_3P_percentage'] > df['team1_3P_percentage'], 'better_3P_PCT'] = 2
    df.loc[df['team2_3P_percentage'] == df['team1_3P_percentage'], 'better_3P_PCT'] = 0  # Tie
    
    # For Free Throw Percentage (higher is better)
    df['better_FT_PCT'] = 1  # Team 1 by default
    df.loc[df['team2_FT_percentage'] > df['team1_FT_percentage'], 'better_FT_PCT'] = 2
    df.loc[df['team2_FT_percentage'] == df['team1_FT_percentage'], 'better_FT_PCT'] = 0  # Tie
    
    # For Fouls (lower is better)
    df['better_FOULS'] = 1  # Team 1 by default
    df.loc[df['team2_fouls'] < df['team1_fouls'], 'better_FOULS'] = 2
    df.loc[df['team2_fouls'] == df['team1_fouls'], 'better_FOULS'] = 0  # Tie
    
    # For Biggest Scoring Run (higher is better)
    df['better_RUN'] = 1  # Team 1 by default
    df.loc[df['team2_scoring_run'] > df['team1_scoring_run'], 'better_RUN'] = 2
    df.loc[df['team2_scoring_run'] == df['team1_scoring_run'], 'better_RUN'] = 0  # Tie
    
    # For Points in Paint (higher is better)
    df['better_PAINT'] = 1  # Team 1 by default
    df.loc[df['team2_points_in_paint'] > df['team1_points_in_paint'], 'better_PAINT'] = 2
    df.loc[df['team2_points_in_paint'] == df['team1_points_in_paint'], 'better_PAINT'] = 0  # Tie
    
    # For Points off Turnovers (higher is better)
    df['better_PTS_OFF_TO'] = 1  # Team 1 by default
    df.loc[df['team2_points_off_TO'] > df['team1_points_off_TO'], 'better_PTS_OFF_TO'] = 2
    df.loc[df['team2_points_off_TO'] == df['team1_points_off_TO'], 'better_PTS_OFF_TO'] = 0  # Tie
    
    # For Fast Break Points (higher is better)
    df['better_FAST_BRK'] = 1  # Team 1 by default
    df.loc[df['team2_fast_break'] > df['team1_fast_break'], 'better_FAST_BRK'] = 2
    df.loc[df['team2_fast_break'] == df['team1_fast_break'], 'better_FAST_BRK'] = 0  # Tie
    
    # Calculate number of advantages for each team in four factors
    four_factor_columns = ['better_eFG', 'better_TOV', 'better_ORB', 'better_FTR']
    
    # Count four factors advantages for team1
    df['team1_ff_advantages'] = sum((df[col] == 1) for col in four_factor_columns)
    
    # Count four factors advantages for team2
    df['team2_ff_advantages'] = sum((df[col] == 2) for col in four_factor_columns)
    
    # Calculate win rate when a team has better stats in each factor
    results = {}
    
    # Total games
    total_games = len(df)
    
    # Process four factors
    factor_results = {}
    
    for factor, factor_name in zip(
        four_factor_columns,
        ['Effective FG%', 'Turnover Rate', 'Offensive Rebounding', 'Free Throw Rate']
    ):
        # Games where team1 was better in this factor
        team1_better = df[df[factor] == 1]
        team1_won_count = sum(team1_better['winner'] == 1)
        team1_better_count = len(team1_better)
        
        # Games where team2 was better in this factor
        team2_better = df[df[factor] == 2]
        team2_won_count = sum(team2_better['winner'] == 2)
        team2_better_count = len(team2_better)
        
        # Win percentages
        team1_win_pct = team1_won_count / team1_better_count if team1_better_count > 0 else 0
        team2_win_pct = team2_won_count / team2_better_count if team2_better_count > 0 else 0
        
        # Combined win percentage for teams with advantage in this factor
        combined_wins = team1_won_count + team2_won_count
        combined_games = team1_better_count + team2_better_count
        combined_win_pct = combined_wins / combined_games if combined_games > 0 else 0
        
        factor_results[factor_name] = {
            'win_percentage': combined_win_pct * 100,
            'games_with_advantage': combined_games,
            'wins_with_advantage': combined_wins
        }
    
    # Calculate win rates based on how many four factors a team had advantage in
    advantage_counts = {}
    
    for i in range(5):  # 0-4 advantages
        # Games where team1 had exactly i advantages
        team1_with_i = df[df['team1_ff_advantages'] == i]
        team1_wins_with_i = sum(team1_with_i['winner'] == 1)
        
        # Games where team2 had exactly i advantages
        team2_with_i = df[df['team2_ff_advantages'] == i]
        team2_wins_with_i = sum(team2_with_i['winner'] == 2)
        
        # Combined stats
        total_games_with_i = len(team1_with_i) + len(team2_with_i)
        total_wins_with_i = team1_wins_with_i + team2_wins_with_i
        win_pct_with_i = total_wins_with_i / total_games_with_i if total_games_with_i > 0 else 0
        
        advantage_counts[i] = {
            'games': total_games_with_i,
            'wins': total_wins_with_i,
            'win_percentage': win_pct_with_i * 100
        }
    
    # Process additional stats
    additional_stat_columns = [
        'better_AST', 'better_STL', 'better_BLK', 'better_REB',
        'better_3PM', 'better_FG_PCT', 'better_3P_PCT', 'better_FT_PCT',
        'better_FOULS', 'better_RUN', 'better_PAINT', 'better_PTS_OFF_TO', 
        'better_FAST_BRK'
    ]
    
    additional_stat_names = [
        'Assists', 'Steals', 'Blocks', 'Total Rebounds',
        '3-Pointers Made', 'FG Percentage', '3P Percentage', 'FT Percentage',
        'Fewer Fouls', 'Biggest Scoring Run', 'Points in Paint', 'Points Off Turnovers',
        'Fast Break Points'
    ]
    
    additional_results = {}
    
    for factor, factor_name in zip(additional_stat_columns, additional_stat_names):
        # Games where team1 was better in this stat
        team1_better = df[df[factor] == 1]
        team1_won_count = sum(team1_better['winner'] == 1)
        team1_better_count = len(team1_better)
        
        # Games where team2 was better in this stat
        team2_better = df[df[factor] == 2]
        team2_won_count = sum(team2_better['winner'] == 2)
        team2_better_count = len(team2_better)
        
        # Win percentages
        team1_win_pct = team1_won_count / team1_better_count if team1_better_count > 0 else 0
        team2_win_pct = team2_won_count / team2_better_count if team2_better_count > 0 else 0
        
        # Combined win percentage for teams with advantage in this stat
        combined_wins = team1_won_count + team2_won_count
        combined_games = team1_better_count + team2_better_count
        combined_win_pct = combined_wins / combined_games if combined_games > 0 else 0
        
        additional_results[factor_name] = {
            'win_percentage': combined_win_pct * 100,
            'games_with_advantage': combined_games,
            'wins_with_advantage': combined_wins
        }
    
    return {
        'factor_results': factor_results,
        'advantage_counts': advantage_counts,
        'additional_results': additional_results,
        'raw_data': df
    }

def display_basketball_stats_win_analysis():
    st.title("🏀 Basketball Stats Win Analysis")
    
    # Add timestamp
    st.markdown(f"*Analysis generated on: 2025-03-28 20:41:03*")
    st.markdown(f"*Generated by: Dodga010nice*")
    
    # Run the analysis
    analysis_results = analyze_basketball_stats_wins()
    
    if not analysis_results:
        st.error("No data available for analysis.")
        return
    
    df = analysis_results['raw_data']
    factor_results = analysis_results['factor_results']
    advantage_counts = analysis_results['advantage_counts']
    additional_results = analysis_results['additional_results']
    
    # Display summary of games analyzed
    st.write(f"### Analysis based on {len(df)} games")
    
    # Create tabs for different types of analysis
    tab1, tab2, tab3 = st.tabs(["Four Factors", "Additional Stats", "Combined Analysis"])
    
    with tab1:
        st.write("## Four Factors Analysis")
        
        # Display win percentage for each factor
        st.write("### Win Percentage When Having Advantage in Each Factor")
        
        # Create DataFrame for factor results
        factor_df = pd.DataFrame({
            'Factor': list(factor_results.keys()),
            'Win %': [factor_results[factor]['win_percentage'] for factor in factor_results],
            'Games': [factor_results[factor]['games_with_advantage'] for factor in factor_results],
            'Wins': [factor_results[factor]['wins_with_advantage'] for factor in factor_results]
        })
        
        # Sort by win percentage
        factor_df = factor_df.sort_values('Win %', ascending=False).reset_index(drop=True)
        
        # Display as table
        st.dataframe(
            factor_df.style.format({
                'Win %': '{:.1f}%',
                'Games': '{:.0f}',
                'Wins': '{:.0f}'
            })
        )
        
        # Create bar chart for factor win percentages
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x='Factor', 
            y='Win %', 
            data=factor_df,
            palette='viridis',
            ax=ax
        )
        
        # Add data labels
        for i, row in factor_df.iterrows():
            ax.text(
                i, 
                row['Win %'] + 1, 
                f"{row['Win %']:.1f}%", 
                ha='center',
                fontweight='bold'
            )
        
        ax.set_xlabel("Four Factors")
        ax.set_ylabel("Win Percentage")
        ax.set_title("Win Percentage When Having Advantage in Each Factor")
        plt.xticks(rotation=0)
        
        st.pyplot(fig)
        
        # Display win percentage by number of advantages
        st.write("### Win Percentage by Number of Four Factor Advantages")
        
        # Create DataFrame for advantage counts
        advantage_df = pd.DataFrame({
            'Advantages': list(advantage_counts.keys()),
            'Win %': [advantage_counts[i]['win_percentage'] for i in range(5)],
            'Games': [advantage_counts[i]['games'] for i in range(5)],
            'Wins': [advantage_counts[i]['wins'] for i in range(5)],
        })
        
        # Display as table
        st.dataframe(
            advantage_df.style.format({
                'Win %': '{:.1f}%',
                'Games': '{:.0f}',
                'Wins': '{:.0f}'
            })
        )
        
        # Create line chart for advantage win percentages
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            x='Advantages', 
            y='Win %', 
            data=advantage_df,
            marker='o',
            linewidth=2,
            markersize=10,
            color='blue',
            ax=ax
        )
        
        # Add data labels
        for i, row in advantage_df.iterrows():
            ax.text(
                row['Advantages'], 
                row['Win %'] + 1, 
                f"{row['Win %']:.1f}%", 
                ha='center',
                fontweight='bold'
            )
        
        ax.set_xlabel("Number of Four Factor Advantages")
        ax.set_ylabel("Win Percentage")
        ax.set_title("Win Percentage by Number of Four Factor Advantages")
        ax.set_xticks(range(5))
        ax.set_xlim(-0.5, 4.5)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Show details of what the four factors are
        with st.expander("What are the Four Factors?", expanded=False):
            st.write("""
            The "Four Factors" of basketball success were identified by Dean Oliver as the key statistics that determine the outcome of basketball games:
            
            1. **Effective Field Goal Percentage (eFG%)**: Accounts for the fact that 3-point field goals are worth 50% more than 2-point field goals.
               - Formula: (FG + 0.5 * 3P) / FGA
               - Higher is better
            
            2. **Turnover Rate (TOV%)**: The percentage of possessions that end in a turnover.
               - Formula: TO / (FGA + 0.44 * FTA + TO)
               - Lower is better
            
            3. **Offensive Rebounding Percentage (ORB%)**: The percentage of available offensive rebounds a team gets.
               - Formula: ORB / (ORB + Opponent DRB)
               - Higher is better
            
            4. **Free Throw Rate (FTR)**: How often a team gets to the free throw line relative to field goal attempts.
               - Formula: FTA / FGA
               - Higher is better
            """)
    
    with tab2:
        st.write("## Additional Stats Analysis")
        
        # Display win percentage for each additional stat
        st.write("### Win Percentage When Having Advantage in Each Statistic")
        
        # Create DataFrame for additional stat results
        add_stat_df = pd.DataFrame({
            'Statistic': list(additional_results.keys()),
            'Win %': [additional_results[stat]['win_percentage'] for stat in additional_results],
            'Games': [additional_results[stat]['games_with_advantage'] for stat in additional_results],
            'Wins': [additional_results[stat]['wins_with_advantage'] for stat in additional_results]
        })
        
        # Sort by win percentage
        add_stat_df = add_stat_df.sort_values('Win %', ascending=False).reset_index(drop=True)
        
        # Display as table
        st.dataframe(
            add_stat_df.style.format({
                'Win %': '{:.1f}%',
                'Games': '{:.0f}',
                'Wins': '{:.0f}'
            })
        )
        
        # Create bar chart for additional stats win percentages
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = sns.color_palette("coolwarm", len(add_stat_df))
        bars = sns.barplot(
            x='Statistic', 
            y='Win %', 
            data=add_stat_df,
            palette=colors,
            ax=ax
        )
        
        # Add data labels
        for i, row in add_stat_df.iterrows():
            ax.text(
                i, 
                row['Win %'] + 0.5, 
                f"{row['Win %']:.1f}%", 
                ha='center',
                fontweight='bold',
                fontsize=9
            )
        
        ax.set_xlabel("Statistics")
        ax.set_ylabel("Win Percentage")
        ax.set_title("Win Percentage When Having Advantage in Each Statistic")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Create a heatmap showing the correlation between different stats and winning
        st.write("### Correlation Between Stats and Winning")
        
        # Create a correlation matrix between having advantage in each stat and winning
        all_columns = ['winner'] + ['better_' + col for col in ['eFG', 'TOV', 'ORB', 'FTR', 'AST', 'STL', 'BLK', 'REB', '3PM', 'FG_PCT', '3P_PCT', 'FT_PCT', 'FOULS', 'RUN', 'PAINT', 'PTS_OFF_TO', 'FAST_BRK']]
        
        # Create binary columns for team1 winning and having advantage in each stat
        binary_df = pd.DataFrame()
        binary_df['won_game'] = (df['winner'] == 1).astype(int)
        
        for col in all_columns[1:]:
            binary_df[col] = (df[col] == 1).astype(int)
        
        # Calculate correlation
        corr_matrix = binary_df.corr()
        
        # Display heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        heatmap = sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            cmap='coolwarm', 
            fmt='.2f',
            linewidths=0.5,
            vmin=-1, 
            vmax=1,
            center=0,
            square=True,
            ax=ax
        )
        
        plt.title('Correlation Between Stats and Winning')
        plt.tight_layout()
        
        st.pyplot(fig)
    
    with tab3:
        st.write("## Combined Analysis")
        
        # Combine four factors and additional stats
        all_stats_results = {**factor_results, **additional_results}
        
        # Create DataFrame for all stats results
        all_stats_df = pd.DataFrame({
            'Statistic': list(all_stats_results.keys()),
            'Win %': [all_stats_results[stat]['win_percentage'] for stat in all_stats_results],
            'Games': [all_stats_results[stat]['games_with_advantage'] for stat in all_stats_results],
            'Wins': [all_stats_results[stat]['wins_with_advantage'] for stat in all_stats_results]
        })
        
        # Sort by win percentage
        all_stats_df = all_stats_df.sort_values('Win %', ascending=False).reset_index(drop=True)
        
        # Add a column to identify if it's a four factor
        all_stats_df['Category'] = 'Other Stat'
        all_stats_df.loc[all_stats_df['Statistic'].isin(factor_results.keys()), 'Category'] = 'Four Factor'
        
        # Display as table
        st.write("### All Stats Ranked by Win Percentage")
        st.dataframe(
            all_stats_df.style.format({
                'Win %': '{:.1f}%',
                'Games': '{:.0f}',
                'Wins': '{:.0f}'
            })
        )
        
        # Create bar chart for all stats win percentages
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = sns.barplot(
            x='Statistic', 
            y='Win %', 
            data=all_stats_df,
            hue='Category',
            palette={'Four Factor': '#3366cc', 'Other Stat': '#ff9900'},
            ax=ax
        )
        
        # Add data labels
        for i, row in all_stats_df.iterrows():
            ax.text(
                i, 
                row['Win %'] + 0.5, 
                f"{row['Win %']:.1f}%", 
                ha='center',
                fontweight='bold',
                fontsize=9,
                rotation=0
            )
        
        ax.set_xlabel("Statistics")
        ax.set_ylabel("Win Percentage")
        ax.set_title("All Stats Ranked by Win Percentage")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Category')
        plt.tight_layout()
        
        st.pyplot(fig)
        
        # Display top 5 most important stats for winning
        st.write("### Top 5 Most Important Stats for Winning")
        top_stats = all_stats_df.head(5)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        cols = [col1, col2, col3, col4, col5]
        
        for i, (_, row) in enumerate(top_stats.iterrows()):
            with cols[i]:
                st.metric(
                    f"#{i+1}: {row['Statistic']}", 
                    f"{row['Win %']:.1f}%",
                    f"{row['Category']}"
                )
        
        # Example game analysis section
        st.write("### Example: Analyzing a Game with all Stats")
        
        # Pick a recent game randomly
        sample_game = df.sample(1).iloc[0]
        team1_name = sample_game['team1_name']
        team2_name = sample_game['team2_name']
        team1_score = int(sample_game['team1_score'])
        team2_score = int(sample_game['team2_score'])
        winner = team1_name if sample_game['winner'] == 1 else team2_name
        
        st.write(f"#### {team1_name} ({team1_score}) vs {team2_name} ({team2_score})")
        st.write(f"Winner: **{winner}**")
        
        # Create DataFrames to compare stats side by side
        four_factors_comparison = pd.DataFrame({
            'Four Factors': ['Effective FG%', 'Turnover Rate', 'Offensive Rebounding', 'Free Throw Rate'],
            team1_name: [
                f"{sample_game['team1_eFG_percentage']:.1f}%",
                f"{sample_game['team1_TOV_percentage']:.1f}%",
                f"{sample_game['team1_ORB_percentage']:.1f}%",
                f"{sample_game['team1_FTR_percentage']:.1f}%"
            ],
            team2_name: [
                f"{sample_game['team2_eFG_percentage']:.1f}%",
                f"{sample_game['team2_TOV_percentage']:.1f}%",
                f"{sample_game['team2_ORB_percentage']:.1f}%",
                f"{sample_game['team2_FTR_percentage']:.1f}%"
            ],
            'Advantage': [
                team1_name if sample_game['better_eFG'] == 1 else team2_name,
                team1_name if sample_game['better_TOV'] == 1 else team2_name,
                team1_name if sample_game['better_ORB'] == 1 else team2_name,
                team1_name if sample_game['better_FTR'] == 1 else team2_name
            ]
        })
        
        # Display the four factors comparison
        st.write("##### Four Factors Comparison")
        st.dataframe(four_factors_comparison)

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime

# Define SQLite database path
db_path = os.path.join(os.path.dirname(__file__), "database.db")

def analyze_team_comparison(team1_name, team2_name):
    """
    Analyze strengths and weaknesses between two teams based on:
    - Head-to-head matchups
    - Overall season stats
    - Four factors comparison
    - Key statistical categories
    
    Parameters:
    -----------
    team1_name : str
        Name of first team to compare
    team2_name : str
        Name of second team to compare
    
    Returns:
    --------
    dict
        Dictionary containing all comparison results
    """
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Results dictionary
    results = {
        'head_to_head': {},
        'season_stats': {},
        'four_factors': {},
        'strengths_weaknesses': {}
    }
    
    # 1. Head-to-head matchups
    h2h_query = """
    WITH team1_matches AS (
        SELECT 
            t1.game_id,
            t1.name AS team1_name,
            t2.name AS team2_name,
            (t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) AS team1_score,
            (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score) AS team2_score
        FROM Teams t1
        JOIN Teams t2 ON t1.game_id = t2.game_id AND t1.tm != t2.tm
        WHERE t1.name = ? AND t2.name = ?
    ),
    team2_matches AS (
        SELECT 
            t1.game_id,
            t2.name AS team1_name,
            t1.name AS team2_name,
            (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score) AS team1_score,
            (t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) AS team2_score
        FROM Teams t1
        JOIN Teams t2 ON t1.game_id = t2.game_id AND t1.tm != t2.tm
        WHERE t1.name = ? AND t2.name = ?
    )
    SELECT * FROM team1_matches
    UNION
    SELECT * FROM team2_matches
    ORDER BY game_id
    """
    
    h2h_df = pd.read_sql_query(h2h_query, conn, params=(team1_name, team2_name, team2_name, team1_name))
    
    # Calculate head-to-head record
    if not h2h_df.empty:
        team1_wins = sum((h2h_df['team1_name'] == team1_name) & 
                          (h2h_df['team1_score'] > h2h_df['team2_score']) | 
                          (h2h_df['team2_name'] == team1_name) & 
                          (h2h_df['team2_score'] > h2h_df['team1_score']))
                          
        team2_wins = sum((h2h_df['team1_name'] == team2_name) & 
                          (h2h_df['team1_score'] > h2h_df['team2_score']) | 
                          (h2h_df['team2_name'] == team2_name) & 
                          (h2h_df['team2_score'] > h2h_df['team1_score']))
        
        results['head_to_head'] = {
            'team1_wins': int(team1_wins),
            'team2_wins': int(team2_wins),
            'total_games': len(h2h_df),
            'matchup_details': h2h_df.to_dict('records')
        }
    else:
        results['head_to_head'] = {
            'team1_wins': 0,
            'team2_wins': 0,
            'total_games': 0,
            'matchup_details': []
        }
    
    # 2. Overall season stats
    season_query = """
    SELECT 
        name AS team_name,
        COUNT(DISTINCT game_id) AS games_played,
        AVG(p1_score + p2_score + p3_score + p4_score) AS avg_points,
        AVG(field_goals_made) AS avg_fg_made,
        AVG(field_goals_attempted) AS avg_fg_attempted,
        AVG(field_goal_percentage) AS avg_fg_pct,
        AVG(three_pointers_made) AS avg_3p_made,
        AVG(three_pointers_attempted) AS avg_3p_attempted,
        AVG(three_point_percentage) AS avg_3p_pct,
        AVG(free_throws_made) AS avg_ft_made,
        AVG(free_throws_attempted) AS avg_ft_attempted,
        AVG(free_throw_percentage) AS avg_ft_pct,
        AVG(rebounds_total) AS avg_rebounds,
        AVG(rebounds_offensive) AS avg_off_rebounds,
        AVG(rebounds_defensive) AS avg_def_rebounds,
        AVG(assists) AS avg_assists,
        AVG(turnovers) AS avg_turnovers,
        AVG(steals) AS avg_steals,
        AVG(blocks) AS avg_blocks,
        AVG(fouls_total) AS avg_fouls,
        AVG(points_in_paint) AS avg_paint_points,
        AVG(points_from_turnovers) AS avg_pts_off_to,
        AVG(points_second_chance) AS avg_second_chance,
        AVG(bench_points) AS avg_bench_points
    FROM Teams
    WHERE name IN (?, ?)
    GROUP BY name
    """
    
    season_df = pd.read_sql_query(season_query, conn, params=(team1_name, team2_name))
    
    if len(season_df) == 2:
        # Convert to dict for easier access
        team1_stats = season_df[season_df['team_name'] == team1_name].iloc[0].to_dict()
        team2_stats = season_df[season_df['team_name'] == team2_name].iloc[0].to_dict()
        
        # Identify stat differences
        stat_comparisons = {}
        for stat in season_df.columns:
            if stat != 'team_name':
                team1_val = team1_stats[stat]
                team2_val = team2_stats[stat]
                diff = team1_val - team2_val
                
                # Determine which team is better in this stat
                # For most stats, higher is better, except for turnovers and fouls
                is_higher_better = True
                if 'turnovers' in stat or 'fouls' in stat:
                    is_higher_better = False
                
                better_team = team1_name if (diff > 0 and is_higher_better) or (diff < 0 and not is_higher_better) else team2_name
                
                stat_comparisons[stat] = {
                    'team1_value': float(team1_val),
                    'team2_value': float(team2_val),
                    'difference': float(diff),
                    'better_team': better_team,
                    'is_significant': abs(diff) > (0.1 * max(team1_val, team2_val))  # 10% difference threshold
                }
        
        results['season_stats'] = {
            'team1_stats': team1_stats,
            'team2_stats': team2_stats,
            'comparisons': stat_comparisons
        }
    else:
        results['season_stats'] = {
            'team1_stats': {},
            'team2_stats': {},
            'comparisons': {}
        }
    
    # 3. Four factors comparison
    four_factors_query = """
    SELECT 
        name AS team_name,
        AVG((field_goals_made + 0.5 * three_pointers_made) * 100.0 / field_goals_attempted) AS avg_efg_pct,
        AVG(turnovers * 100.0 / (field_goals_attempted + 0.44 * free_throws_attempted)) AS avg_tov_pct,
        AVG(rebounds_offensive * 100.0 / (rebounds_offensive + rebounds_defensive)) AS avg_orb_pct,
        AVG(free_throws_attempted * 100.0 / field_goals_attempted) AS avg_ftr_pct
    FROM Teams
    WHERE name IN (?, ?)
    GROUP BY name
    """
    
    four_factors_df = pd.read_sql_query(four_factors_query, conn, params=(team1_name, team2_name))
    
    if len(four_factors_df) == 2:
        # Convert to dict for easier access
        team1_factors = four_factors_df[four_factors_df['team_name'] == team1_name].iloc[0].to_dict()
        team2_factors = four_factors_df[four_factors_df['team_name'] == team2_name].iloc[0].to_dict()
        
        # Compare four factors
        factor_comparisons = {}
        for factor in ['avg_efg_pct', 'avg_tov_pct', 'avg_orb_pct', 'avg_ftr_pct']:
            team1_val = team1_factors[factor]
            team2_val = team2_factors[factor]
            diff = team1_val - team2_val
            
            # Determine which team is better in this factor
            # For turnover percentage, lower is better
            is_higher_better = factor != 'avg_tov_pct'
            
            better_team = team1_name if (diff > 0 and is_higher_better) or (diff < 0 and not is_higher_better) else team2_name
            
            factor_comparisons[factor] = {
                'team1_value': float(team1_val),
                'team2_value': float(team2_val),
                'difference': float(diff),
                'better_team': better_team,
                'is_significant': abs(diff) > (0.1 * max(team1_val, team2_val))  # 10% difference threshold
            }
        
        results['four_factors'] = {
            'team1_factors': team1_factors,
            'team2_factors': team2_factors,
            'comparisons': factor_comparisons
        }
    else:
        results['four_factors'] = {
            'team1_factors': {},
            'team2_factors': {},
            'comparisons': {}
        }
    
    # 4. Identify strengths and weaknesses
    if results['season_stats']['comparisons']:
        # Find areas where each team excels (significant advantage)
        team1_strengths = {}
        team2_strengths = {}
        
        for stat, comparison in results['season_stats']['comparisons'].items():
            if comparison['is_significant']:
                if comparison['better_team'] == team1_name:
                    team1_strengths[stat] = comparison
                else:
                    team2_strengths[stat] = comparison
        
        # Sort strengths by significance (absolute difference / average value)
        for team_strengths in [team1_strengths, team2_strengths]:
            for stat, data in team_strengths.items():
                avg_val = (data['team1_value'] + data['team2_value']) / 2
                if avg_val != 0:
                    data['significance_ratio'] = abs(data['difference']) / avg_val
                else:
                    data['significance_ratio'] = 0
        
        # Sort by significance ratio
        team1_strengths = {k: v for k, v in sorted(
            team1_strengths.items(), 
            key=lambda item: item[1]['significance_ratio'], 
            reverse=True
        )}
        
        team2_strengths = {k: v for k, v in sorted(
            team2_strengths.items(), 
            key=lambda item: item[1]['significance_ratio'], 
            reverse=True
        )}
        
        results['strengths_weaknesses'] = {
            'team1_strengths': team1_strengths,
            'team2_strengths': team2_strengths
        }
    
    # Close database connection
    conn.close()
    
    return results

def display_team_comparison_analysis():
    """Display team comparison analysis in Streamlit"""
    st.title("🏀 Team Comparison Analysis")
    
    # Current date/time
    st.markdown(f"*Analysis generated on: 2025-03-28 21:09:04*")
    st.markdown(f"*Analysis by: Dodga010*")
    
    # Team selection
    teams = fetch_teams()
    
    if not teams:
        st.error("No team data available.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("Select First Team", teams, index=0, key="team1")
    
    with col2:
        # Filter out team1 from options for team2
        team2_options = [team for team in teams if team != team1]
        team2 = st.selectbox("Select Second Team", team2_options, index=0, key="team2")
    
    if team1 and team2 and team1 != team2:
        # Run comparison analysis
        with st.spinner(f"Analyzing {team1} vs {team2}..."):
            comparison_results = analyze_team_comparison(team1, team2)
        
        # 1. Head-to-head record
        st.header("📊 Head-to-Head Results")
        
        h2h_data = comparison_results['head_to_head']
        total_games = h2h_data['total_games']
        
        if total_games > 0:
            team1_wins = h2h_data['team1_wins']
            team2_wins = h2h_data['team2_wins']
            
            # Create columns for the H2H record
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(team1, team1_wins)
            
            with col2:
                st.metric("Games Played", total_games)
            
            with col3:
                st.metric(team2, team2_wins)
            
            # Display matchup details
            st.subheader("Game Details")
            
            # Create a more readable dataframe for display
            games_df = pd.DataFrame(h2h_data['matchup_details'])
            
            # Format for display
            display_games = []
            for _, game in games_df.iterrows():
                if game['team1_name'] == team1:
                    team1_score = game['team1_score']
                    team2_score = game['team2_score']
                else:
                    team1_score = game['team2_score']
                    team2_score = game['team1_score']
                
                winner = team1 if team1_score > team2_score else team2
                
                display_games.append({
                    'Game ID': game['game_id'],
                    f'{team1} Score': int(team1_score),
                    f'{team2} Score': int(team2_score),
                    'Winner': winner,
                    'Margin': abs(int(team1_score) - int(team2_score))
                })
            
            st.table(pd.DataFrame(display_games))
        else:
            st.info(f"No head-to-head games found between {team1} and {team2}.")
        
        # 2. Season Stats Comparison
        st.header("📈 Season Statistics Comparison")
        
        season_stats = comparison_results['season_stats']
        
        if season_stats['comparisons']:
            # Create a dataframe for the key stats
            key_stats = [
                'avg_points', 'avg_fg_pct', 'avg_3p_pct', 'avg_ft_pct',
                'avg_rebounds', 'avg_assists', 'avg_turnovers', 'avg_steals', 'avg_blocks'
            ]
            
            stat_display_names = {
                'avg_points': 'Points',
                'avg_fg_pct': 'FG%',
                'avg_3p_pct': '3P%',
                'avg_ft_pct': 'FT%',
                'avg_rebounds': 'Rebounds',
                'avg_assists': 'Assists',
                'avg_turnovers': 'Turnovers',
                'avg_steals': 'Steals',
                'avg_blocks': 'Blocks'
            }
            
            stats_df = pd.DataFrame({
                'Statistic': [stat_display_names.get(stat, stat) for stat in key_stats],
                team1: [round(season_stats['team1_stats'][stat], 1) for stat in key_stats],
                team2: [round(season_stats['team2_stats'][stat], 1) for stat in key_stats],
                'Difference': [round(season_stats['comparisons'][stat]['difference'], 1) for stat in key_stats],
                'Better Team': [season_stats['comparisons'][stat]['better_team'] for stat in key_stats]
            })
            
            # Display stats table
            st.table(stats_df.set_index('Statistic'))
            
            # Create a radar chart for visual comparison
            fig = create_radar_chart(season_stats, team1, team2, key_stats, stat_display_names)
            st.pyplot(fig)
            
            # Additional important stats section
            st.subheader("Additional Key Statistics")
            
            additional_stats = [
                'avg_paint_points', 'avg_pts_off_to', 'avg_second_chance', 'avg_bench_points'
            ]
            
            additional_display_names = {
                'avg_paint_points': 'Points in Paint',
                'avg_pts_off_to': 'Points off Turnovers',
                'avg_second_chance': 'Second Chance Points',
                'avg_bench_points': 'Bench Points'
            }
            
            additional_df = pd.DataFrame({
                'Statistic': [additional_display_names.get(stat, stat) for stat in additional_stats],
                team1: [round(season_stats['team1_stats'][stat], 1) for stat in additional_stats],
                team2: [round(season_stats['team2_stats'][stat], 1) for stat in additional_stats],
                'Difference': [round(season_stats['comparisons'][stat]['difference'], 1) for stat in additional_stats],
                'Better Team': [season_stats['comparisons'][stat]['better_team'] for stat in additional_stats]
            })
            
            # Display additional stats table
            st.table(additional_df.set_index('Statistic'))
        
        else:
            st.info("Season statistics comparison not available.")
        
        # 3. Four Factors Analysis
        st.header("🔍 Four Factors Analysis")
        
        four_factors = comparison_results['four_factors']
        
        if four_factors['comparisons']:
            factor_display_names = {
                'avg_efg_pct': 'Effective Field Goal %',
                'avg_tov_pct': 'Turnover %',
                'avg_orb_pct': 'Offensive Rebound %',
                'avg_ftr_pct': 'Free Throw Rate'
            }
            
            factors_df = pd.DataFrame({
                'Factor': list(factor_display_names.values()),
                team1: [round(four_factors['team1_factors'][factor], 1) for factor in factor_display_names.keys()],
                team2: [round(four_factors['team2_factors'][factor], 1) for factor in factor_display_names.keys()],
                'Difference': [round(four_factors['comparisons'][factor]['difference'], 1) for factor in factor_display_names.keys()],
                'Better Team': [four_factors['comparisons'][factor]['better_team'] for factor in factor_display_names.keys()],
                'Is Significant': [four_factors['comparisons'][factor]['is_significant'] for factor in factor_display_names.keys()]
            })
            
            # Display four factors table
            st.table(factors_df.set_index('Factor'))
            
            # Create bar chart for visual comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(factor_display_names))
            width = 0.35
            
            team1_values = [four_factors['team1_factors'][factor] for factor in factor_display_names.keys()]
            team2_values = [four_factors['team2_factors'][factor] for factor in factor_display_names.keys()]
            
            rects1 = ax.bar(x - width/2, team1_values, width, label=team1)
            rects2 = ax.bar(x + width/2, team2_values, width, label=team2)
            
            ax.set_ylabel('Value')
            ax.set_title('Four Factors Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(list(factor_display_names.values()))
            ax.legend()
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
        
        else:
            st.info("Four factors comparison not available.")
        
        # 4. Strengths and Weaknesses Analysis
        st.header("💪 Team Strengths Analysis")
        
        strengths = comparison_results['strengths_weaknesses']
        
        if strengths:
            col1, col2 = st.columns(2)
            
            # Team 1 Strengths
            with col1:
                st.subheader(f"{team1} Strengths")
                
                if strengths['team1_strengths']:
                    strength_items = []
                    for stat, data in list(strengths['team1_strengths'].items())[:5]:  # Top 5 strengths
                        # Format stat name for display
                        display_stat = stat.replace('avg_', '').replace('_', ' ').title()
                        
                        # Calculate percentage difference
                        pct_diff = (data['difference'] / data['team2_value'] * 100) if data['team2_value'] != 0 else 0
                        
                        strength_items.append({
                            'Statistic': display_stat,
                            'Value': round(data['team1_value'], 2),
                            'Advantage': f"{round(abs(pct_diff), 1)}% better"
                        })
                    
                    st.table(pd.DataFrame(strength_items).set_index('Statistic'))
                else:
                    st.info(f"No significant statistical advantages found for {team1}.")
            
            # Team 2 Strengths
            with col2:
                st.subheader(f"{team2} Strengths")
                
                if strengths['team2_strengths']:
                    strength_items = []
                    for stat, data in list(strengths['team2_strengths'].items())[:5]:  # Top 5 strengths
                        # Format stat name for display
                        display_stat = stat.replace('avg_', '').replace('_', ' ').title()
                        
                        # Calculate percentage difference
                        pct_diff = (data['difference'] / data['team1_value'] * 100) if data['team1_value'] != 0 else 0
                        
                        strength_items.append({
                            'Statistic': display_stat,
                            'Value': round(data['team2_value'], 2),
                            'Advantage': f"{round(abs(pct_diff), 1)}% better"
                        })
                    
                    st.table(pd.DataFrame(strength_items).set_index('Statistic'))
                else:
                    st.info(f"No significant statistical advantages found for {team2}.")
        
        # 5. Matchup Analysis and Prediction
        st.header("🔮 Matchup Analysis")
        
        # Count advantages in important categories
        if season_stats['comparisons']:
            team1_advantages = 0
            team2_advantages = 0
            important_stats = [
                'avg_points', 'avg_fg_pct', 'avg_3p_pct',
                'avg_rebounds', 'avg_assists', 'avg_steals'
            ]
            
            for stat in important_stats:
                if stat in season_stats['comparisons']:
                    if season_stats['comparisons'][stat]['better_team'] == team1:
                        team1_advantages += 1
                    else:
                        team2_advantages += 1
            
            # Four factors advantages
            for factor in four_factors['comparisons']:
                if four_factors['comparisons'][factor]['better_team'] == team1:
                    team1_advantages += 1
                else:
                    team2_advantages += 1
            
            # Head-to-head record
            if h2h_data['team1_wins'] > h2h_data['team2_wins']:
                team1_advantages += 1
            elif h2h_data['team2_wins'] > h2h_data['team1_wins']:
                team2_advantages += 1
            
            # Display prediction
            st.subheader("Matchup Advantages")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(team1, team1_advantages)
            
            with col2:
                st.metric(team2, team2_advantages)
            
            favored_team = team1 if team1_advantages > team2_advantages else team2
            advantage_diff = abs(team1_advantages - team2_advantages)
            
            if advantage_diff >= 5:
                st.write(f"**Significant advantage for {favored_team}** with {advantage_diff} more statistical edges.")
            elif advantage_diff >= 3:
                st.write(f"**Moderate advantage for {favored_team}** with {advantage_diff} more statistical edges.")
            elif advantage_diff >= 1:
                st.write(f"**Slight advantage for {favored_team}** with {advantage_diff} more statistical edges.")
            else:
                st.write("**Even matchup** with both teams having similar statistical profiles.")

def create_radar_chart(season_stats, team1, team2, key_stats, stat_display_names):
    """Create a radar chart comparing the two teams"""
    # Normalize the values
    max_values = {}
    for stat in key_stats:
        max_values[stat] = max(season_stats['team1_stats'][stat], season_stats['team2_stats'][stat])
    
    # For stats where lower is better (like turnovers), invert the normalization
    inverted_stats = ['avg_turnovers']
    
    team1_values = []
    team2_values = []
    for stat in key_stats:
        if max_values[stat] > 0:
            if stat in inverted_stats:
                team1_values.append(1 - (season_stats['team1_stats'][stat] / max_values[stat]))
                team2_values.append(1 - (season_stats['team2_stats'][stat] / max_values[stat]))
            else:
                team1_values.append(season_stats['team1_stats'][stat] / max_values[stat])
                team2_values.append(season_stats['team2_stats'][stat] / max_values[stat])
        else:
            team1_values.append(0)
            team2_values.append(0)
    
    # Set up the radar chart
    labels = [stat_display_names.get(stat, stat) for stat in key_stats]
    
    # Create the angles for each statistic (ensure same number of points as values)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    
    # Close the polygon by appending the first value to the end
    team1_values.append(team1_values[0])
    team2_values.append(team2_values[0])
    angles.append(angles[0])  # Close the loop
    labels.append(labels[0])  # Add the first label to close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw the chart - now angles and values have the same length
    ax.plot(angles, team1_values, 'o-', linewidth=2, label=team1, color='blue')
    ax.fill(angles, team1_values, alpha=0.1, color='blue')
    
    ax.plot(angles, team2_values, 'o-', linewidth=2, label=team2, color='red')
    ax.fill(angles, team2_values, alpha=0.1, color='red')
    
    # Set labels - make sure to use only the original labels (not the duplicated one)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels[:-1])
    
    # Add legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    ax.set_title(f"Team Stats Comparison: {team1} vs {team2}")
    
    return fig

def fetch_teams():
    """Fetch all team names from database"""
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT name FROM Teams ORDER BY name"
    teams = pd.read_sql_query(query, conn)["name"].tolist()
    conn.close()
    return teams

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime

# Define SQLite database path
db_path = os.path.join(os.path.dirname(__file__), "database.db")

def analyze_advanced_metrics(team1_name, team2_name):
    """
    Calculate and compare advanced basketball metrics between two teams
    
    Parameters:
    -----------
    team1_name : str
        Name of first team to compare
    team2_name : str
        Name of second team to compare
    
    Returns:
    --------
    dict
        Dictionary containing all advanced metrics comparison
    """
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Query raw data needed for calculations
    query = """
    SELECT 
        name AS team_name,
        AVG(p1_score + p2_score + p3_score + p4_score) AS avg_points,
        AVG(field_goals_made) AS avg_fg_made,
        AVG(field_goals_attempted) AS avg_fg_attempted,
        AVG(field_goal_percentage) AS avg_fg_pct,
        AVG(three_pointers_made) AS avg_3p_made,
        AVG(three_pointers_attempted) AS avg_3p_attempted,
        AVG(three_point_percentage) AS avg_3p_pct,
        AVG(free_throws_made) AS avg_ft_made,
        AVG(free_throws_attempted) AS avg_ft_attempted,
        AVG(rebounds_total) AS avg_rebounds,
        AVG(rebounds_offensive) AS avg_off_rebounds,
        AVG(rebounds_defensive) AS avg_def_rebounds,
        AVG(assists) AS avg_assists,
        AVG(turnovers) AS avg_turnovers,
        AVG(steals) AS avg_steals,
        AVG(blocks) AS avg_blocks,
        AVG(fouls_total) AS avg_fouls,
        AVG(field_goals_made + 0.5 * three_pointers_made) AS avg_efg_numerator,
        COUNT(DISTINCT game_id) AS games_played
    FROM Teams
    WHERE name IN (?, ?)
    GROUP BY name
    """
    
    df = pd.read_sql_query(query, conn, params=(team1_name, team2_name))
    
    # Get average opponent stats for each team
    opponent_query = """
    WITH team_games AS (
        SELECT 
            t1.game_id,
            t1.name AS team_name,
            t2.name AS opponent_name,
            t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score AS opponent_score,
            t2.field_goals_made AS opponent_fg_made,
            t2.field_goals_attempted AS opponent_fg_attempted,
            t2.three_pointers_made AS opponent_3p_made,
            t2.three_pointers_attempted AS opponent_3p_attempted,
            t2.rebounds_total AS opponent_rebounds,
            t2.rebounds_offensive AS opponent_off_rebounds,
            t2.rebounds_defensive AS opponent_def_rebounds,
            t2.assists AS opponent_assists,
            t2.turnovers AS opponent_turnovers,
            t2.steals AS opponent_steals,
            t2.blocks AS opponent_blocks
        FROM Teams t1
        JOIN Teams t2 ON t1.game_id = t2.game_id AND t1.tm != t2.tm
        WHERE t1.name IN (?, ?)
    )
    SELECT 
        team_name,
        AVG(opponent_score) AS avg_opponent_points,
        AVG(opponent_fg_made) AS avg_opponent_fg_made,
        AVG(opponent_fg_attempted) AS avg_opponent_fg_attempted,
        AVG(opponent_3p_made) AS avg_opponent_3p_made,
        AVG(opponent_3p_attempted) AS avg_opponent_3p_attempted,
        AVG(opponent_rebounds) AS avg_opponent_rebounds,
        AVG(opponent_off_rebounds) AS avg_opponent_off_rebounds,
        AVG(opponent_def_rebounds) AS avg_opponent_def_rebounds,
        AVG(opponent_assists) AS avg_opponent_assists,
        AVG(opponent_turnovers) AS avg_opponent_turnovers,
        AVG(opponent_steals) AS avg_opponent_steals,
        AVG(opponent_blocks) AS avg_opponent_blocks
    FROM team_games
    GROUP BY team_name
    """
    
    opponent_df = pd.read_sql_query(opponent_query, conn, params=(team1_name, team2_name))
    
    # Close connection
    conn.close()
    
    if df.empty or len(df) < 2 or opponent_df.empty or len(opponent_df) < 2:
        return None
    
    # Convert DFs to dictionaries for easier access
    team1_stats = df[df['team_name'] == team1_name].iloc[0].to_dict()
    team2_stats = df[df['team_name'] == team2_name].iloc[0].to_dict()
    
    team1_opp = opponent_df[opponent_df['team_name'] == team1_name].iloc[0].to_dict()
    team2_opp = opponent_df[opponent_df['team_name'] == team2_name].iloc[0].to_dict()
    
    # Calculate advanced metrics
    team1_metrics = calculate_team_metrics(team1_stats, team1_opp)
    team2_metrics = calculate_team_metrics(team2_stats, team2_opp)
    
    # Compare metrics and determine advantages
    comparisons = {}
    
    # Loop through all metrics in team1_metrics (should be same as team2_metrics)
    for metric, value in team1_metrics.items():
        team2_value = team2_metrics[metric]
        diff = value - team2_value
        
        # Determine if higher is better (most metrics - higher is better, except for defensive metrics)
        is_higher_better = True
        if metric in ['defensive_rating', 'opponent_efg', 'turnover_rate']:
            is_higher_better = False
        
        better_team = team1_name if (diff > 0 and is_higher_better) or (diff < 0 and not is_higher_better) else team2_name
        
        comparisons[metric] = {
            'team1_value': value,
            'team2_value': team2_value,
            'difference': diff,
            'better_team': better_team,
            'is_significant': abs(diff) > (0.05 * max(abs(value), abs(team2_value), 0.0001)) # 5% difference threshold
        }
    
    return {
        'team1_name': team1_name,
        'team1_metrics': team1_metrics,
        'team2_name': team2_name,
        'team2_metrics': team2_metrics,
        'comparisons': comparisons
    }

def calculate_team_metrics(team_stats, opponent_stats):
    """Calculate advanced basketball metrics for a team"""
    
    # Estimated pace calculation (per 40 minutes)
    avg_pace = (team_stats['avg_fg_attempted'] + 0.44 * team_stats['avg_ft_attempted'] + team_stats['avg_turnovers'] 
              + opponent_stats['avg_opponent_fg_attempted'] + 0.44 * opponent_stats['avg_opponent_fg_attempted'] 
              + opponent_stats['avg_opponent_turnovers'])
    
    # Offensive and defensive ratings (points per 100 possessions)
    possessions = team_stats['avg_fg_attempted'] + team_stats['avg_turnovers'] + 0.44 * team_stats['avg_ft_attempted'] - team_stats['avg_off_rebounds']
    
    offensive_rating = 100 * team_stats['avg_points'] / possessions if possessions > 0 else 0
    defensive_rating = 100 * opponent_stats['avg_opponent_points'] / possessions if possessions > 0 else 0
    
    # Net rating
    net_rating = offensive_rating - defensive_rating
    
    # True shooting percentage
    true_shooting_pct = team_stats['avg_points'] / (2 * (team_stats['avg_fg_attempted'] + 0.44 * team_stats['avg_ft_attempted'])) if (team_stats['avg_fg_attempted'] + 0.44 * team_stats['avg_ft_attempted']) > 0 else 0
    
    # Effective field goal percentage
    efg_pct = team_stats['avg_efg_numerator'] / team_stats['avg_fg_attempted'] if team_stats['avg_fg_attempted'] > 0 else 0
    
    # Opponent effective field goal percentage
    opponent_efg = (opponent_stats['avg_opponent_fg_made'] + 0.5 * opponent_stats['avg_opponent_3p_made']) / opponent_stats['avg_opponent_fg_attempted'] if opponent_stats['avg_opponent_fg_attempted'] > 0 else 0
    
    # Assist ratio = Assists / (FGA + FTA + Turnovers)
    assist_ratio = 100 * team_stats['avg_assists'] / (team_stats['avg_fg_attempted'] + team_stats['avg_ft_attempted'] + team_stats['avg_turnovers']) if (team_stats['avg_fg_attempted'] + team_stats['avg_ft_attempted'] + team_stats['avg_turnovers']) > 0 else 0
    
    # Turnover rate
    turnover_rate = 100 * team_stats['avg_turnovers'] / (team_stats['avg_fg_attempted'] + 0.44 * team_stats['avg_ft_attempted'] + team_stats['avg_turnovers']) if (team_stats['avg_fg_attempted'] + 0.44 * team_stats['avg_ft_attempted'] + team_stats['avg_turnovers']) > 0 else 0
    
    # Assist to turnover ratio
    ast_to_ratio = team_stats['avg_assists'] / team_stats['avg_turnovers'] if team_stats['avg_turnovers'] > 0 else 0
    
    # Defensive rebound percentage
    def_reb_pct = 100 * team_stats['avg_def_rebounds'] / (team_stats['avg_def_rebounds'] + opponent_stats['avg_opponent_off_rebounds']) if (team_stats['avg_def_rebounds'] + opponent_stats['avg_opponent_off_rebounds']) > 0 else 0
    
    # Offensive rebound percentage
    off_reb_pct = 100 * team_stats['avg_off_rebounds'] / (team_stats['avg_off_rebounds'] + opponent_stats['avg_opponent_def_rebounds']) if (team_stats['avg_off_rebounds'] + opponent_stats['avg_opponent_def_rebounds']) > 0 else 0
    
    # Total rebound percentage
    total_reb_pct = 100 * team_stats['avg_rebounds'] / (team_stats['avg_rebounds'] + opponent_stats['avg_opponent_rebounds']) if (team_stats['avg_rebounds'] + opponent_stats['avg_opponent_rebounds']) > 0 else 0
    
    # Steal percentage
    steal_pct = 100 * team_stats['avg_steals'] / possessions if possessions > 0 else 0
    
    # Block percentage
    block_pct = 100 * team_stats['avg_blocks'] / opponent_stats['avg_opponent_fg_attempted'] if opponent_stats['avg_opponent_fg_attempted'] > 0 else 0
    
    # Points per shot
    pts_per_shot = team_stats['avg_points'] / team_stats['avg_fg_attempted'] if team_stats['avg_fg_attempted'] > 0 else 0
    
    return {
        'pace': avg_pace,
        'offensive_rating': offensive_rating,
        'defensive_rating': defensive_rating,
        'net_rating': net_rating,
        'true_shooting_pct': true_shooting_pct * 100,  # Convert to percentage
        'efg_pct': efg_pct * 100,  # Convert to percentage
        'opponent_efg': opponent_efg * 100,  # Convert to percentage
        'assist_ratio': assist_ratio,
        'turnover_rate': turnover_rate,
        'ast_to_ratio': ast_to_ratio,
        'def_reb_pct': def_reb_pct,
        'off_reb_pct': off_reb_pct,
        'total_reb_pct': total_reb_pct,
        'steal_pct': steal_pct,
        'block_pct': block_pct,
        'pts_per_shot': pts_per_shot
    }

def display_advanced_metrics_analysis():
    """Display advanced metrics analysis in Streamlit"""
    st.header("📊 Advanced Metrics Analysis")
    
    # Get teams for comparison (assuming these are already selected)
    team1 = st.session_state.get("team1")
    team2 = st.session_state.get("team2")
    
    if not team1 or not team2:
        st.warning("Please select teams for comparison first.")
        return
    
    # Run advanced metrics analysis
    with st.spinner("Calculating advanced metrics..."):
        metrics_results = analyze_advanced_metrics(team1, team2)
    
    if not metrics_results:
        st.warning("Insufficient data to calculate advanced metrics.")
        return
    
    # Display metrics in categories
    metric_categories = {
        "Efficiency Metrics": [
            ("offensive_rating", "Offensive Rating", "Points per 100 possessions"),
            ("defensive_rating", "Defensive Rating", "Points allowed per 100 possessions"),
            ("net_rating", "Net Rating", "Off. Rating - Def. Rating"),
            ("true_shooting_pct", "True Shooting %", "Accounts for FG, 3PT, FT"),
            ("efg_pct", "Effective FG%", "Adjusts for 3PT value"),
            ("pts_per_shot", "Points Per Shot", "Total points / FGA")
        ],
        "Possession Metrics": [
            ("pace", "Pace", "Estimated possessions per 40 minutes"),
            ("assist_ratio", "Assist Ratio", "% of possessions ending in assist"),
            ("turnover_rate", "Turnover Rate", "% of possessions ending in turnover"),
            ("ast_to_ratio", "AST/TO Ratio", "Assists per turnover")
        ],
        "Rebounding Metrics": [
            ("def_reb_pct", "Defensive Rebound %", "% of available def. rebounds secured"),
            ("off_reb_pct", "Offensive Rebound %", "% of available off. rebounds secured"),
            ("total_reb_pct", "Total Rebound %", "% of all available rebounds secured")
        ],
        "Defensive Metrics": [
            ("opponent_efg", "Opponent eFG%", "Opponent's effective field goal %"),
            ("steal_pct", "Steal %", "% of opponent possessions ending in steal"),
            ("block_pct", "Block %", "% of opponent FGA blocked")
        ]
    }
    
    # For each category, create a section
    for category, metrics in metric_categories.items():
        st.subheader(category)
        
        # Create columns for metric name, team1, team2, better team
        columns = ["Metric", team1, team2, "Better Team"]
        data = []
        
        for metric_id, metric_name, tooltip in metrics:
            if metric_id in metrics_results['comparisons']:
                comparison = metrics_results['comparisons'][metric_id]
                team1_value = comparison['team1_value']
                team2_value = comparison['team2_value']
                better_team = comparison['better_team']
                
                # Format numbers appropriately
                if metric_id in ['offensive_rating', 'defensive_rating', 'net_rating']:
                    team1_formatted = f"{team1_value:.1f}"
                    team2_formatted = f"{team2_value:.1f}"
                else:
                    team1_formatted = f"{team1_value:.1f}%"
                    team2_formatted = f"{team2_value:.1f}%"
                
                data.append([
                    f"{metric_name} ({tooltip})",
                    team1_formatted,
                    team2_formatted,
                    better_team
                ])
        
        # Create DataFrame and display as table
        if data:
            df = pd.DataFrame(data, columns=columns)
            st.table(df.set_index("Metric"))
        else:
            st.info(f"No {category.lower()} data available.")
    
    # Create visualization for key metrics
    st.subheader("🔍 Key Metrics Comparison")
    
    key_metrics = [
        ('offensive_rating', 'Offensive Rating'),
        ('defensive_rating', 'Defensive Rating'),
        ('net_rating', 'Net Rating'),
        ('true_shooting_pct', 'True Shooting %'),
        ('efg_pct', 'Effective FG%'),
        ('ast_to_ratio', 'AST/TO Ratio')
    ]
    
    # Prepare data for bar chart
    metric_names = [name for _, name in key_metrics]
    team1_values = [metrics_results['team1_metrics'][metric] for metric, _ in key_metrics]
    team2_values = [metrics_results['team2_metrics'][metric] for metric, _ in key_metrics]
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'Metric': metric_names,
        team1: team1_values,
        team2: team2_values
    })
    
    # Melt DataFrame for easier plotting
    plot_df_melted = pd.melt(plot_df, id_vars=['Metric'], var_name='Team', value_name='Value')
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=plot_df_melted, x='Metric', y='Value', hue='Team', palette=['blue', 'red'])
    plt.xticks(rotation=45)
    plt.title('Key Advanced Metrics Comparison')
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Interpretation and insights
    st.subheader("📝 Analysis Summary")
    
    # Count significant advantages
    team1_advantages = sum(1 for comp in metrics_results['comparisons'].values() 
                          if comp['better_team'] == team1 and comp['is_significant'])
    team2_advantages = sum(1 for comp in metrics_results['comparisons'].values() 
                          if comp['better_team'] == team2 and comp['is_significant'])
    
    st.write(f"**{team1}** has significant advantages in **{team1_advantages}** advanced metrics.")
    st.write(f"**{team2}** has significant advantages in **{team2_advantages}** advanced metrics.")
    
    # Highlight particular strengths
    st.write("### Key Strengths")
    
    # Offensive strengths
    offensive_metrics = ['offensive_rating', 'true_shooting_pct', 'efg_pct', 'pts_per_shot']
    team1_off_adv = [metric for metric in offensive_metrics if metric in metrics_results['comparisons'] and
                     metrics_results['comparisons'][metric]['better_team'] == team1 and
                     metrics_results['comparisons'][metric]['is_significant']]
    
    team2_off_adv = [metric for metric in offensive_metrics if metric in metrics_results['comparisons'] and
                     metrics_results['comparisons'][metric]['better_team'] == team2 and
                     metrics_results['comparisons'][metric]['is_significant']]
    
    # Defensive strengths
    defensive_metrics = ['defensive_rating', 'opponent_efg', 'steal_pct', 'block_pct']
    team1_def_adv = [metric for metric in defensive_metrics if metric in metrics_results['comparisons'] and
                     metrics_results['comparisons'][metric]['better_team'] == team1 and
                     metrics_results['comparisons'][metric]['is_significant']]
    
    team2_def_adv = [metric for metric in defensive_metrics if metric in metrics_results['comparisons'] and
                     metrics_results['comparisons'][metric]['better_team'] == team2 and
                     metrics_results['comparisons'][metric]['is_significant']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{team1}**")
        if team1_off_adv:
            st.write("*Offensive:* " + ", ".join(team1_off_adv))
        if team1_def_adv:
            st.write("*Defensive:* " + ", ".join(team1_def_adv))
        if not team1_off_adv and not team1_def_adv:
            st.write("No significant strengths identified.")
    
    with col2:
        st.write(f"**{team2}**")
        if team2_off_adv:
            st.write("*Offensive:* " + ", ".join(team2_off_adv))
        if team2_def_adv:
            st.write("*Defensive:* " + ", ".join(team2_def_adv))
        if not team2_off_adv and not team2_def_adv:
            st.write("No significant strengths identified.")

def analyze_shot_distribution_comparison(team1_name, team2_name):
    """Analyze and compare shot distributions between two teams"""
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Get shot data for both teams
    query = """
    WITH team_ids AS (
        SELECT DISTINCT game_id, tm 
        FROM Teams 
        WHERE name = ?
    )
    SELECT 
        s.x_coord, s.y_coord, s.shot_result, s.action_type,
        CASE 
            WHEN s.action_type = '2pt' THEN 2
            WHEN s.action_type = '3pt' THEN 3
            ELSE 0
        END AS point_value
    FROM Shots s
    JOIN team_ids t ON s.game_id = t.game_id AND s.team_id = t.tm
    """
    
    team1_shots = pd.read_sql_query(query, conn, params=(team1_name,))
    team2_shots = pd.read_sql_query(query, conn, params=(team2_name,))
    
    conn.close()
    
    if team1_shots.empty or team2_shots.empty:
        return None
    
    # Process shot data
    results = {
        'team1_name': team1_name,
        'team2_name': team2_name,
        'team1_shots': len(team1_shots),
        'team2_shots': len(team2_shots)
    }
    
    # Calculate shot zones
    zone_definitions = {
        'rim': lambda x, y: np.sqrt((x - 25)**2 + (y - 5.25)**2) <= 4,  # Within 4 units of basket
        'paint': lambda x, y: y <= 19 and not np.sqrt((x - 25)**2 + (y - 5.25)**2) <= 4,  # In paint but not rim
        'midrange': lambda x, y: y > 19 and not is_three_point(x, y),  # Not in paint and not 3PT
        'corner_three': lambda x, y: is_three_point(x, y) and (x <= 10 or x >= 40),  # Corner 3
        'above_break_three': lambda x, y: is_three_point(x, y) and not (x <= 10 or x >= 40)  # Other 3s
    }
    
    # Classify shots into zones
    for team_shots, team_key in [(team1_shots, 'team1'), (team2_shots, 'team2')]:
        zone_counts = {zone: 0 for zone in zone_definitions}
        zone_makes = {zone: 0 for zone in zone_definitions}
        zone_points = {zone: 0 for zone in zone_definitions}
        
        for _, shot in team_shots.iterrows():
            for zone, condition in zone_definitions.items():
                if condition(shot['x_coord'], shot['y_coord']):
                    zone_counts[zone] += 1
                    if shot['shot_result'] == 1:
                        zone_makes[zone] += 1
                        zone_points[zone] += shot['point_value']
                    break
        
        total_shots = len(team_shots)
        
        # Calculate percentages and efficiency
        zone_percentages = {zone: (count / total_shots * 100) if total_shots > 0 else 0 
                           for zone, count in zone_counts.items()}
        
        zone_efficiency = {zone: (points / count) if count > 0 else 0 
                         for zone, count, points in zip(zone_counts.keys(), 
                                                      zone_counts.values(), 
                                                      zone_points.values())}
        
        zone_accuracy = {zone: (makes / count * 100) if count > 0 else 0 
                       for zone, count, makes in zip(zone_counts.keys(), 
                                                  zone_counts.values(), 
                                                  zone_makes.values())}
        
        results[f'{team_key}_zones'] = {
            'counts': zone_counts,
            'percentages': zone_percentages,
            'efficiency': zone_efficiency,
            'accuracy': zone_accuracy
        }
    
    # Compare zone distributions and efficiency
    zone_comparisons = {}
    for zone in zone_definitions:
        team1_pct = results['team1_zones']['percentages'][zone]
        team2_pct = results['team2_zones']['percentages'][zone]
        team1_eff = results['team1_zones']['efficiency'][zone]
        team2_eff = results['team2_zones']['efficiency'][zone]
        
        zone_comparisons[zone] = {
            'team1_percentage': team1_pct,
            'team2_percentage': team2_pct,
            'percentage_diff': team1_pct - team2_pct,
            'team1_efficiency': team1_eff,
            'team2_efficiency': team2_eff,
            'efficiency_diff': team1_eff - team2_eff
        }
    
    results['zone_comparisons'] = zone_comparisons
    
    return results

def is_three_point(x, y):
    """Determine if a shot is a three-pointer based on coordinates"""
    # Distance from center of court
    distance = np.sqrt((x - 25)**2 + (y - 25)**2)
    # Approximately 23.75 feet (basketball 3pt line) in our coordinate system
    return distance >= 22

def display_shot_distribution_analysis():
    """Display shot distribution analysis in Streamlit"""
    st.header("🎯 Shot Distribution Analysis")
    
    # Get teams for comparison (assuming these are already selected)
    team1 = st.session_state.get("team1")
    team2 = st.session_state.get("team2")
    
    if not team1 or not team2:
        st.warning("Please select teams for comparison first.")
        return
    
    # Run shot distribution analysis
    with st.spinner("Analyzing shot distributions..."):
        shot_results = analyze_shot_distribution_comparison(team1, team2)
    
    if not shot_results:
        st.warning("Insufficient shot data to analyze distribution.")
        return
    
    # Display shot distribution comparison
    st.subheader(f"Shot Zone Breakdown: {team1} vs {team2}")
    
    # Create dataframe for zone comparison
    zone_names = {
        'rim': 'At Rim',
        'paint': 'Paint (Non-Rim)',
        'midrange': 'Mid-Range',
        'corner_three': 'Corner 3',
        'above_break_three': 'Above Break 3'
    }
    
    zone_data = []
    for zone_id, zone_name in zone_names.items():
        if zone_id in shot_results['zone_comparisons']:
            comp = shot_results['zone_comparisons'][zone_id]
            zone_data.append({
                'Zone': zone_name,
                f'{team1} %': f"{comp['team1_percentage']:.1f}%",
                f'{team1} PPP': f"{comp['team1_efficiency']:.2f}",
                f'{team2} %': f"{comp['team2_percentage']:.1f}%",
                f'{team2} PPP': f"{comp['team2_efficiency']:.2f}",
                'Distribution Diff': f"{comp['percentage_diff']:.1f}%",
                'Efficiency Diff': f"{comp['efficiency_diff']:.2f}"
            })
    
    zone_df = pd.DataFrame(zone_data)
    st.table(zone_df.set_index('Zone'))
    
    # Create visualization
    st.subheader("Zone Distribution Comparison")
    
    # Prepare data for plotting
    zones = list(zone_names.values())
    team1_pct = [shot_results['zone_comparisons'][zone]['team1_percentage'] for zone in zone_names]
    team2_pct = [shot_results['zone_comparisons'][zone]['team2_percentage'] for zone in zone_names]
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(zones))
    width = 0.35
    
    # Plot bars
    ax.bar(x - width/2, team1_pct, width, label=team1, color='blue', alpha=0.7)
    ax.bar(x + width/2, team2_pct, width, label=team2, color='red', alpha=0.7)
    
    # Customize chart
    ax.set_ylabel('% of Total Shots')
    ax.set_title('Shot Distribution by Zone')
    ax.set_xticks(x)
    ax.set_xticklabels(zones)
    ax.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    # Create efficiency visualization
    st.subheader("Shooting Efficiency by Zone")
    
    # Prepare efficiency data
    team1_eff = [shot_results['zone_comparisons'][zone]['team1_efficiency'] for zone in zone_names]
    team2_eff = [shot_results['zone_comparisons'][zone]['team2_efficiency'] for zone in zone_names]
    
    # Set up figure
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    ax2.bar(x - width/2, team1_eff, width, label=team1, color='blue', alpha=0.7)
    ax2.bar(x + width/2, team2_eff, width, label=team2, color='red', alpha=0.7)
    
    # Add horizontal line for league average (~1 point per possession)
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='League Avg')
    
    # Customize chart
    ax2.set_ylabel('Points Per Shot')
    ax2.set_title('Shooting Efficiency by Zone')
    ax2.set_xticks(x)
    ax2.set_xticklabels(zones)
    ax2.legend()
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    st.pyplot(fig2)
    
    # Analysis and insights
    st.subheader("📝 Shot Distribution Insights")
    
    # Identify strengths and preferences
    team1_favored_zones = sorted(zone_names.keys(), 
                               key=lambda z: shot_results['zone_comparisons'][z]['team1_percentage'],
                               reverse=True)[:2]
    
    team2_favored_zones = sorted(zone_names.keys(), 
                               key=lambda z: shot_results['zone_comparisons'][z]['team2_percentage'],
                               reverse=True)[:2]
    
    team1_efficient_zones = sorted(zone_names.keys(), 
                                 key=lambda z: shot_results['zone_comparisons'][z]['team1_efficiency'],
                                 reverse=True)[:2]
    
    team2_efficient_zones = sorted(zone_names.keys(), 
                                 key=lambda z: shot_results['zone_comparisons'][z]['team2_efficiency'],
                                 reverse=True)[:2]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{team1} Shot Profile**")
        st.write(f"Favorite Zones: {', '.join([zone_names[z] for z in team1_favored_zones])}")
        st.write(f"Most Efficient Zones: {', '.join([zone_names[z] for z in team1_efficient_zones])}")
        
        # Identify if team is three-point heavy or rim-heavy
        three_pt_pct = (shot_results['zone_comparisons']['corner_three']['team1_percentage'] + 
                       shot_results['zone_comparisons']['above_break_three']['team1_percentage'])
        
        rim_pct = shot_results['zone_comparisons']['rim']['team1_percentage']
        
        if three_pt_pct > 35:
            st.write("**Style:** Three-point heavy offense")
        elif rim_pct > 35:
            st.write("**Style:** Rim-attacking offense")
        elif shot_results['zone_comparisons']['midrange']['team1_percentage'] > 30:
            st.write("**Style:** Midrange-focused offense")
        else:
            st.write("**Style:** Balanced shot distribution")
    
    with col2:
        st.write(f"**{team2} Shot Profile**")
        st.write(f"Favorite Zones: {', '.join([zone_names[z] for z in team2_favored_zones])}")
        st.write(f"Most Efficient Zones: {', '.join([zone_names[z] for z in team2_efficient_zones])}")
        
        # Identify if team is three-point heavy or rim-heavy
        three_pt_pct = (shot_results['zone_comparisons']['corner_three']['team2_percentage'] + 
                       shot_results['zone_comparisons']['above_break_three']['team2_percentage'])
        
        rim_pct = shot_results['zone_comparisons']['rim']['team2_percentage']
        
        if three_pt_pct > 35:
            st.write("**Style:** Three-point heavy offense")
        elif rim_pct > 35:
            st.write("**Style:** Rim-attacking offense")
        elif shot_results['zone_comparisons']['midrange']['team2_percentage'] > 30:
            st.write("**Style:** Midrange-focused offense")
        else:
            st.write("**Style:** Balanced shot distribution")

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import streamlit as st
from datetime import datetime

# Define SQLite database path
db_path = os.path.join(os.path.dirname(__file__), "database.db")

def analyze_team_shooting(team_name):
    """
    Analyze team shooting patterns using the existing court image
    """
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Query to get shooting data for the specified team
    query = """
    SELECT 
        s.game_id,
        s.team_id,
        s.player_name,
        s.period,
        s.action_type,
        s.shot_result,
        s.x_coord,
        s.y_coord,
        t.tm, -- This indicates if team is home (1) or away (2)
        t.name
    FROM Shots s
    JOIN Teams t ON s.game_id = t.game_id AND s.team_id = t.tm
    WHERE t.name = ?
    """
    
    shots_df = pd.read_sql_query(query, conn, params=(team_name,))
    
    # Close connection
    conn.close()
    
    if shots_df.empty:
        return None
    
    # Scale coordinates for the court image (assuming the court is 280x261)
    shots_df['court_x'] = shots_df['x_coord'] * 2.8
    shots_df['court_y'] = shots_df['y_coord'] * 2.61
    
    # Check if a shot is actually a 3-pointer based on distance from basket
    # Basket position at approximately (6.2, 50) in the original coordinates
    # So in the scaled coordinates it would be around (17.36, 130.5)
    basket_x = 6.2 * 2.8  # = 17.36
    basket_y = 50 * 2.61  # = 130.5
    
    # Function to determine if a shot is a three-pointer based on distance
    def is_three_pointer(x, y):
        # Convert coordinates
        x_court = x
        y_court = y
        
        # Distance from basket
        distance = np.sqrt((x_court - basket_x)**2 + (y_court - basket_y)**2)
        
        # 3-point line is approximately at a radius of 23.75 feet from basket
        # In our coordinate system this would be around 67 units
        return distance > 67
    
    # Verify shot types (sometimes the database classification might not match the actual position)
    shots_df['is_three_based_on_position'] = shots_df.apply(
        lambda row: is_three_pointer(row['court_x'], row['court_y']), 
        axis=1
    )
    
    # For analysis, we'll use both the recorded type and the position-based type
    shots_df['shot_type'] = shots_df['action_type']
    
    # Calculate points for each shot
    shots_df['points'] = 0
    shots_df.loc[(shots_df['shot_type'] == '2pt') & (shots_df['shot_result'] == 1), 'points'] = 2
    shots_df.loc[(shots_df['shot_type'] == '3pt') & (shots_df['shot_result'] == 1), 'points'] = 3
    
    # Define shot zones based on position on court
    shots_df['zone'] = 'Other'
    
    # Restricted area (close to basket)
    distance_from_basket = np.sqrt((shots_df['court_x'] - basket_x)**2 + (shots_df['court_y'] - basket_y)**2)
    shots_df.loc[distance_from_basket < 12, 'zone'] = 'Restricted Area'
    
    # Paint area
    shots_df.loc[(distance_from_basket >= 12) & 
                (distance_from_basket < 25) & 
                (abs(shots_df['court_y'] - basket_y) < 42), 'zone'] = 'Paint'
    
    # Mid-range
    shots_df.loc[(distance_from_basket >= 25) & 
                (distance_from_basket < 67) & 
                (shots_df['shot_type'] == '2pt'), 'zone'] = 'Mid-Range'
    
    # Corner 3
    shots_df.loc[(shots_df['shot_type'] == '3pt') & 
                (abs(shots_df['court_y'] - 130.5) > 70), 'zone'] = 'Corner 3'
    
    # Above the break 3
    shots_df.loc[(shots_df['shot_type'] == '3pt') & 
                (abs(shots_df['court_y'] - 130.5) <= 70), 'zone'] = 'Above Break 3'
    
    # Calculate shot efficiency by zone
    zone_stats = shots_df.groupby('zone').agg(
        total_shots=pd.NamedAgg(column='shot_result', aggfunc='count'),
        made_shots=pd.NamedAgg(column='shot_result', aggfunc='sum'),
        points=pd.NamedAgg(column='points', aggfunc='sum'),
    ).reset_index()
    
    # Calculate percentages and points per shot
    zone_stats['fg_percentage'] = round(zone_stats['made_shots'] / zone_stats['total_shots'] * 100, 1)
    zone_stats['pps'] = round(zone_stats['points'] / zone_stats['total_shots'], 2)
    zone_stats['distribution'] = round(zone_stats['total_shots'] / len(shots_df) * 100, 1)
    
    # Identify hot and cold zones (relative to average)
    avg_fg = shots_df['shot_result'].mean() * 100
    zone_stats['hot_zone'] = zone_stats['fg_percentage'] > (avg_fg + 5)  # 5% above average
    zone_stats['cold_zone'] = zone_stats['fg_percentage'] < (avg_fg - 5)  # 5% below average
    
    # Group by player for player shooting analysis
    player_stats = shots_df.groupby('player_name').agg(
        total_shots=pd.NamedAgg(column='shot_result', aggfunc='count'),
        made_shots=pd.NamedAgg(column='shot_result', aggfunc='sum'),
        points=pd.NamedAgg(column='points', aggfunc='sum')
    ).reset_index()
    
    player_stats['fg_percentage'] = round(player_stats['made_shots'] / player_stats['total_shots'] * 100, 1)
    player_stats['pps'] = round(player_stats['points'] / player_stats['total_shots'], 2)
    
    # Sort by total shots
    player_stats = player_stats.sort_values('total_shots', ascending=False)
    
    # Calculate additional team shooting stats
    total_shots = len(shots_df)
    made_shots = shots_df['shot_result'].sum()
    fg_percentage = round(made_shots / total_shots * 100, 1) if total_shots > 0 else 0
    
    two_point_shots = shots_df[shots_df['shot_type'] == '2pt']
    two_point_made = two_point_shots['shot_result'].sum()
    two_point_total = len(two_point_shots)
    two_point_pct = round(two_point_made / two_point_total * 100, 1) if two_point_total > 0 else 0
    
    three_point_shots = shots_df[shots_df['shot_type'] == '3pt']
    three_point_made = three_point_shots['shot_result'].sum()
    three_point_total = len(three_point_shots)
    three_point_pct = round(three_point_made / three_point_total * 100, 1) if three_point_total > 0 else 0
    
    return {
        'raw_data': shots_df,
        'zone_stats': zone_stats,
        'player_stats': player_stats,
        'team_stats': {
            'total_shots': total_shots,
            'made_shots': made_shots,
            'fg_percentage': fg_percentage,
            'two_point_total': two_point_total,
            'two_point_made': two_point_made,
            'two_point_pct': two_point_pct,
            'three_point_total': three_point_total,
            'three_point_made': three_point_made,
            'three_point_pct': three_point_pct,
            'total_points': shots_df['points'].sum(),
            'pps': round(shots_df['points'].sum() / total_shots, 2) if total_shots > 0 else 0
        }
    }

def display_team_shooting_analysis():
    """Display team shooting analysis in Streamlit using court image"""
    st.title("🏀 Team Shooting Analysis")
    
    # Current date/time and user
    st.markdown(f"*Analysis generated on: 2025-03-28 21:47:56*")
    st.markdown(f"*Generated by: Dodga010*")
    
    # Team selection
    teams = fetch_teams()
    
    if not teams:
        st.error("No team data available.")
        return
    
    selected_team = st.selectbox("Select Team", teams, index=0)
    
    if selected_team:
        # Run shooting analysis
        with st.spinner(f"Analyzing shooting data for {selected_team}..."):
            shooting_results = analyze_team_shooting(selected_team)
        
        if shooting_results is None:
            st.error(f"No shooting data available for {selected_team}.")
            return
            
        # Display overall shooting stats
        st.header("📈 Team Shooting Overview")
        
        team_stats = shooting_results['team_stats']
        
        # Overall shooting metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Field Goal %", f"{team_stats['fg_percentage']}%")
        
        with col2:
            st.metric("2PT %", f"{team_stats['two_point_pct']}%")
        
        with col3:
            st.metric("3PT %", f"{team_stats['three_point_pct']}%")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Shots", team_stats['total_shots'])
        
        with col2:
            st.metric("Made Shots", team_stats['made_shots'])
        
        with col3:
            st.metric("Points Per Shot", team_stats['pps'])
        
        # Shot chart visualization
        st.subheader("Shot Distribution Chart")
        
        # Create shot chart using the image instead of drawing
        fig, ax = plt.subplots(figsize=(12, 11))
        
        # Check if the court image exists
        if os.path.exists("fiba_courtonly.jpg"):
            # Load court image
            court_img = mpimg.imread("fiba_courtonly.jpg")
            ax.imshow(court_img, extent=[0, 280, 0, 261], aspect="auto")
            
            # Plot shots
            shots_df = shooting_results['raw_data']
            made = shots_df[shots_df['shot_result'] == 1]
            missed = shots_df[shots_df['shot_result'] == 0]
            
            ax.scatter(made['court_x'], made['court_y'], 
                      c='green', s=50, alpha=0.7, marker='o', label='Made')
            ax.scatter(missed['court_x'], missed['court_y'], 
                      c='red', s=50, alpha=0.7, marker='x', label='Missed')
            
            # Add a shot density heatmap
            if len(shots_df) > 10:  # Only add heatmap if we have enough shots
                sns.kdeplot(x=shots_df['court_x'], y=shots_df['court_y'], 
                           fill=True, alpha=0.3, levels=10, 
                           cmap='YlOrRd', ax=ax)
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add title and legend
            ax.set_title(f"{selected_team} Shot Chart", fontsize=16)
            ax.legend(loc='upper left')
            
            st.pyplot(fig)
        else:
            st.error("Court image 'fiba_courtonly.jpg' not found.")
            
            # Create a simple scatter plot without court image as fallback
            fig, ax = plt.subplots(figsize=(10, 10))
            
            shots_df = shooting_results['raw_data']
            made = shots_df[shots_df['shot_result'] == 1]
            missed = shots_df[shots_df['shot_result'] == 0]
            
            ax.scatter(made['x_coord'], made['y_coord'], 
                      c='green', s=50, alpha=0.7, marker='o', label='Made')
            ax.scatter(missed['x_coord'], missed['y_coord'], 
                      c='red', s=50, alpha=0.7, marker='x', label='Missed')
            
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_title(f"{selected_team} Shot Chart (No Court Image)", fontsize=16)
            ax.legend()
            
            st.pyplot(fig)
        
        # Shot zone analysis
        st.subheader("Shooting Efficiency by Zone")
        
        zone_stats = shooting_results['zone_stats'].sort_values('total_shots', ascending=False)
        
        # Format for display
        zone_display = zone_stats.copy()
        zone_display['FG%'] = zone_display['fg_percentage'].astype(str) + '%'
        zone_display['Distribution'] = zone_display['distribution'].astype(str) + '%'
        
        # Add a column that indicates if it's a hot or cold zone
        zone_display['Zone Efficiency'] = 'Average'
        zone_display.loc[zone_display['hot_zone'] == True, 'Zone Efficiency'] = 'Hot Zone 🔥'
        zone_display.loc[zone_display['cold_zone'] == True, 'Zone Efficiency'] = 'Cold Zone ❄️'
        
        # Select and rename columns for display
        display_df = zone_display[['zone', 'total_shots', 'made_shots', 'FG%', 'pps', 'Distribution', 'Zone Efficiency']].rename(
            columns={
                'zone': 'Zone', 
                'total_shots': 'Attempts', 
                'made_shots': 'Made', 
                'pps': 'Points/Shot'
            }
        )
        
        # Display zone stats table
        st.dataframe(display_df)
        
        # Player shooting stats
        st.subheader("Individual Player Shooting")
        
        player_stats = shooting_results['player_stats'].head(10)  # Top 10 shooters
        player_display = player_stats.copy()
        player_display['FG%'] = player_display['fg_percentage'].astype(str) + '%'
        
        st.table(player_display[['player_name', 'total_shots', 'made_shots', 'FG%', 'pps']].rename(
            columns={
                'player_name': 'Player', 
                'total_shots': 'Attempts', 
                'made_shots': 'Made',
                'pps': 'Points/Shot'
            }
        ))
        
        # Shot distribution by type
        st.subheader("Shot Type Distribution")
        
        # Calculate shot type distribution
        shots_df = shooting_results['raw_data']
        shot_types = shots_df.groupby('shot_type').agg(
            count=pd.NamedAgg(column='shot_result', aggfunc='count'),
            made=pd.NamedAgg(column='shot_result', aggfunc='sum'),
            points=pd.NamedAgg(column='points', aggfunc='sum')
        ).reset_index()
        
        shot_types['percentage'] = round(shot_types['count'] / shot_types['count'].sum() * 100, 1)
        shot_types['efficiency'] = round(shot_types['made'] / shot_types['count'] * 100, 1)
        shot_types['pps'] = round(shot_types['points'] / shot_types['count'], 2)
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.pie(shot_types['percentage'], 
               labels=[f"{row['shot_type']} ({row['percentage']}%)" for _, row in shot_types.iterrows()],
               autopct='%1.1f%%',
               startangle=90,
               colors=['#ff9999','#66b3ff'])
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        ax.set_title(f"{selected_team} Shot Type Distribution")
        
        st.pyplot(fig)
        
        # Shot zone distribution by efficiency
        st.subheader("Shot Zone Efficiency")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Use zone stats
        zones = zone_stats['zone']
        pps_values = zone_stats['pps']
        
        # Sort by points per shot
        sorted_indices = pps_values.argsort()
        zones = zones.iloc[sorted_indices]
        pps_values = pps_values.iloc[sorted_indices]
        
        # Create horizontal bar chart
        bars = ax.barh(zones, pps_values, color='skyblue')
        
        # Add data labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.05, bar.get_y() + bar.get_height()/2,
                   f"{width:.2f}", ha='left', va='center')
        
        ax.set_xlabel('Points Per Shot')
        ax.set_title(f'{selected_team} Shooting Efficiency by Zone')
        
        # Add average PPS line
        avg_pps = team_stats['pps']
        ax.axvline(x=avg_pps, color='red', linestyle='--', label=f'Avg PPS: {avg_pps:.2f}')
        ax.legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display insights
        st.header("📋 Shooting Insights")
        
        # Generate insights
        insights = []
        
        # Insight 1: Overall shooting efficiency
        if team_stats['fg_percentage'] > 45:
            insights.append(f"**Efficient Shooting Team**: {selected_team} shoots {team_stats['fg_percentage']}% from the field, which is above average.")
        else:
            insights.append(f"**Below Average Shooting**: {selected_team} shoots {team_stats['fg_percentage']}% from the field, which is below the typical league average.")
        
        # Insight 2: 3-point shooting
        three_pt_ratio = team_stats['three_point_total'] / team_stats['total_shots'] * 100
        if three_pt_ratio > 40:
            insights.append(f"**Three-Point Heavy**: {round(three_pt_ratio, 1)}% of shots come from beyond the arc.")
            if team_stats['three_point_pct'] > 35:
                insights.append("**Effective 3PT Shooting**: Above average efficiency on high volume.")
            else:
                insights.append("**Inefficient Volume Shooting**: High 3PT volume with below average efficiency.")
        
        # Insight 3: Best zones
        if not zone_stats.empty:
            best_zone = zone_stats.sort_values('pps', ascending=False).iloc[0]
            if best_zone['total_shots'] >= 10:
                insights.append(f"**Most Efficient Zone**: {best_zone['zone']} ({best_zone['pps']} points per shot on {best_zone['total_shots']} attempts).")
        
        # Insight 4: Shot selection
        restricted_area_pct = zone_stats[zone_stats['zone'] == 'Restricted Area']['distribution'].sum() if 'Restricted Area' in zone_stats['zone'].values else 0
        if restricted_area_pct > 30:
            insights.append(f"**Paint-Dominant Team**: {round(restricted_area_pct, 1)}% of shots come from the restricted area.")
        
        mid_range_pct = zone_stats[zone_stats['zone'] == 'Mid-Range']['distribution'].sum() if 'Mid-Range' in zone_stats['zone'].values else 0
        if mid_range_pct > 30:
            insights.append(f"**Mid-Range Heavy**: {round(mid_range_pct, 1)}% of shots are mid-range jumpers.")
        
        # Insight 5: Best shooters
        if not player_stats.empty:
            best_shooter = player_stats[player_stats['total_shots'] >= 20].sort_values('fg_percentage', ascending=False).iloc[0] if not player_stats[player_stats['total_shots'] >= 20].empty else None
            if best_shooter is not None:
                insights.append(f"**Most Efficient Shooter**: {best_shooter['player_name']} ({best_shooter['fg_percentage']}% on {best_shooter['total_shots']} attempts).")
        
        # Display insights
        for insight in insights:
            st.write(insight)
        
        # Shot distribution over time (if we have multiple games)
        game_counts = shooting_results['raw_data']['game_id'].nunique()
        if game_counts > 1:
            st.subheader("Shot Selection Trends")
            
            # Group by game_id and shot_type
            game_shot_trends = shooting_results['raw_data'].groupby(['game_id', 'shot_type']).size().unstack().fillna(0)
            
            # Calculate percentage of each shot type per game
            game_shot_percentages = game_shot_trends.divide(game_shot_trends.sum(axis=1), axis=0) * 100
            
            # Create line chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for shot_type in game_shot_percentages.columns:
                ax.plot(game_shot_percentages.index, game_shot_percentages[shot_type], 
                       marker='o', label=shot_type)
            
            ax.set_xlabel('Game ID')
            ax.set_ylabel('Percentage of Shots')
            ax.set_title(f"{selected_team} Shot Type Distribution Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)

def fetch_teams():
    """Fetch all team names from database"""
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT name FROM Teams ORDER BY name"
    teams = pd.read_sql_query(query, conn)["name"].tolist()
    conn.close()
    return teams

def improve_shot_zone_analysis(shots_df):
    """
    Improved shot zone analysis that:
    1. Normalizes shots to one basket (by inverting coordinates)
    2. Verifies shot type classification
    3. Properly handles corner 3s from both ends of court
    
    Parameters:
    -----------
    shots_df : pandas DataFrame
        DataFrame containing shot data with columns:
        x_coord, y_coord, action_type, period, etc.
    
    Returns:
    --------
    shots_df : pandas DataFrame
        DataFrame with improved shot zone classifications
    """
    # Step 1: Normalize all shots to one basket
    # Shots with x_coord > 50 are taken on the other half of the court
    shots_df['normalized_x'] = shots_df['x_coord'].apply(lambda x: 100 - x if x > 50 else x)
    shots_df['normalized_y'] = shots_df['y_coord'].copy()
    
    # Step 2: Scale coordinates for the court image
    shots_df['court_x'] = shots_df['normalized_x'] * 2.8
    shots_df['court_y'] = shots_df['normalized_y'] * 2.61
    
    # Define basket position (normalized to left side)
    basket_x = 6.2 * 2.8  # = 17.36
    basket_y = 50 * 2.61  # = 130.5
    
    # Step 3: Calculate distance from basket using normalized coordinates
    shots_df['distance_from_basket'] = np.sqrt(
        (shots_df['court_x'] - basket_x)**2 + 
        (shots_df['court_y'] - basket_y)**2
    )
    
    # Step 4: Determine if shot is a 3-pointer based on distance
    # NBA 3-point line is approximately 23.75 feet, converted to our coordinate system
    shots_df['is_three_by_position'] = shots_df['distance_from_basket'] > 67
    
    # Step 5: Compare with recorded shot type for verification
    shots_df['recorded_is_three'] = shots_df['action_type'] == '3pt'
    shots_df['type_mismatch'] = shots_df['is_three_by_position'] != shots_df['recorded_is_three']
    
    # Count and log misclassifications
    misclassified_count = shots_df['type_mismatch'].sum()
    if misclassified_count > 0:
        print(f"Warning: {misclassified_count} shots have mismatched type classification")
        
    # Step 6: Enhanced zone definitions with better corner detection
    # Create a function to classify shots into zones
    def classify_zone(row):
        dist = row['distance_from_basket']
        x = row['court_x']
        y = row['court_y']
        is_three = row['is_three_by_position']
        
        # Record original side of court (left/right) before normalization
        original_side = 'right' if row['x_coord'] > 50 else 'left'
        
        # Restricted area (right at the basket)
        if dist < 12:
            return 'Restricted Area'
            
        # Paint area (not at rim but in the key)
        elif y <= 19 * 2.61 and dist >= 12:
            return 'Paint (Non-Rim)'
            
        # Mid-range (outside paint but inside 3pt line)
        elif not is_three:
            return 'Mid-Range'
            
        # Corner 3s - using normalized coordinates but preserving original side info
        # Corner three criteria: Shot beyond 3pt line AND near sideline (low y-value)
        elif is_three and x < 10 * 2.8:  # Left corner on normalized court
            corner_side = 'Right' if original_side == 'right' else 'Left'
            return f'{corner_side} Corner 3'
            
        elif is_three and x > 40 * 2.8:  # Right corner on normalized court
            corner_side = 'Left' if original_side == 'right' else 'Right'
            return f'{corner_side} Corner 3'
            
        # Above the break 3s (all other 3pt shots)
        elif is_three:
            return 'Above Break 3'
            
        # Fallback
        else:
            return 'Other'
    
    # Apply zone classification
    shots_df['zone'] = shots_df.apply(classify_zone, axis=1)
    
    # Step 7: Calculate shot efficiency by zone
    zone_stats = shots_df.groupby('zone').agg(
        total_shots=pd.NamedAgg(column='action_type', aggfunc='count'),
        made_shots=pd.NamedAgg(column='shot_result', aggfunc='sum'),
        points=pd.NamedAgg(column='points', aggfunc='sum')
    ).reset_index()
    
    # Calculate percentages and points per shot
    zone_stats['fg_percentage'] = round(zone_stats['made_shots'] / zone_stats['total_shots'] * 100, 1)
    zone_stats['pps'] = round(zone_stats['points'] / zone_stats['total_shots'], 2)
    zone_stats['distribution'] = round(zone_stats['total_shots'] / len(shots_df) * 100, 1)
    
    # Step 8: Identify hot and cold zones
    avg_fg = shots_df['shot_result'].mean() * 100
    zone_stats['hot_zone'] = zone_stats['fg_percentage'] > (avg_fg + 5)
    zone_stats['cold_zone'] = zone_stats['fg_percentage'] < (avg_fg - 5)
    
    return shots_df, zone_stats

def generate_improved_shot_chart(player_name=None, team_name=None):
    """
    Generate improved shot chart visualization for a player or team
    
    Parameters:
    -----------
    player_name : str, optional
        Name of the player to analyze
    team_name : str, optional
        Name of the team to analyze
    """
    # Verify that we have either player_name or team_name, but not both
    if not player_name and not team_name:
        raise ValueError("Must provide either player_name or team_name")
    if player_name and team_name:
        raise ValueError("Cannot provide both player_name and team_name")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Construct query based on whether we're analyzing a player or team
    if player_name:
        query = """
        SELECT 
            s.game_id,
            s.team_id,
            s.player_name,
            s.period,
            s.action_type,
            s.shot_result,
            s.x_coord,
            s.y_coord
        FROM Shots s
        WHERE s.player_name = ?;
        """
        params = (player_name,)
        title_entity = player_name
    else:  # team_name
        query = """
        SELECT 
            s.game_id,
            s.team_id,
            s.player_name,
            s.period,
            s.action_type,
            s.shot_result,
            s.x_coord,
            s.y_coord,
            t.tm, 
            t.name
        FROM Shots s
        JOIN Teams t ON s.game_id = t.game_id AND s.team_id = t.tm
        WHERE t.name = ?;
        """
        params = (team_name,)
        title_entity = team_name
    
    # Execute query
    shots_df = pd.read_sql_query(query, conn, params=params)
    
    # Close connection
    conn.close()
    
    if shots_df.empty:
        print(f"No shot data found for {title_entity}.")
        return None, None
    
    # Calculate points for each shot
    shots_df['points'] = 0
    shots_df.loc[(shots_df['action_type'] == '2pt') & (shots_df['shot_result'] == 1), 'points'] = 2
    shots_df.loc[(shots_df['action_type'] == '3pt') & (shots_df['shot_result'] == 1), 'points'] = 3
    
    # Apply improved shot zone analysis
    processed_shots, zone_stats = improve_shot_zone_analysis(shots_df)
    
    # Create shot chart visualization
    fig, ax = plt.subplots(figsize=(12, 11))
    
    # Check if the court image exists
    if os.path.exists("fiba_courtonly.jpg"):
        # Load court image
        court_img = mpimg.imread("fiba_courtonly.jpg")
        ax.imshow(court_img, extent=[0, 280, 0, 261], aspect="auto")
        
        # Plot shots
        made = processed_shots[processed_shots['shot_result'] == 1]
        missed = processed_shots[processed_shots['shot_result'] == 0]
        
        ax.scatter(made['court_x'], made['court_y'], 
                  c='green', s=50, alpha=0.7, marker='o', label='Made')
        ax.scatter(missed['court_x'], missed['court_y'], 
                  c='red', s=50, alpha=0.7, marker='x', label='Missed')
        
        # Add a shot density heatmap
        if len(processed_shots) > 10:  # Only add heatmap if we have enough shots
            sns.kdeplot(x=processed_shots['court_x'], y=processed_shots['court_y'], 
                      fill=True, alpha=0.3, levels=10, 
                      cmap='YlOrRd', ax=ax)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title and legend
        ax.set_title(f"{title_entity} Shot Chart with Improved Zone Analysis", fontsize=16)
        ax.legend(loc='upper left')
        
        # Add text showing the misclassification rate
        misclassified_count = processed_shots['type_mismatch'].sum()
        total_shots = len(processed_shots)
        if total_shots > 0:
            misclass_rate = misclassified_count / total_shots * 100
            ax.text(10, 10, f"Shot type misclassification rate: {misclass_rate:.1f}%", 
                   fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7))
    
    # Return processed data and figure for further analysis
    return processed_shots, zone_stats, fig

def analyze_team_o_d_ratings():
    """
    Calculate offensive and defensive ratings for all teams and display them on a scatter plot.
    
    This function:
    1. Retrieves team data from the database
    2. Calculates offensive and defensive ratings for each team
    3. Creates a quadrant visualization showing team performance
    4. Returns the visualization as a figure
    """
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Query to get team data needed for calculation
    query = """
    WITH team_games AS (
        SELECT 
            t1.game_id,
            t1.name AS team_name,
            t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score AS team_score,
            t1.field_goals_attempted AS team_fga,
            t1.turnovers AS team_to,
            t1.free_throws_attempted AS team_fta,
            t1.rebounds_offensive AS team_orb,
            
            t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score AS opp_score,
            t2.field_goals_attempted AS opp_fga,
            t2.turnovers AS opp_to,
            t2.free_throws_attempted AS opp_fta,
            t2.rebounds_offensive AS opp_orb
        FROM Teams t1
        JOIN Teams t2 ON t1.game_id = t2.game_id AND t1.tm != t2.tm
    )
    SELECT 
        team_name,
        AVG(team_score) AS avg_pts_scored,
        AVG(opp_score) AS avg_pts_allowed,
        AVG(team_fga + 0.44 * team_fta - team_orb + team_to) AS avg_poss_used,
        AVG(opp_fga + 0.44 * opp_fta - opp_orb + opp_to) AS avg_poss_allowed
    FROM team_games
    GROUP BY team_name
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return None
    
    # Calculate offensive and defensive ratings (points per 100 possessions)
    df['offensive_rating'] = 100 * df['avg_pts_scored'] / df['avg_poss_used']
    df['defensive_rating'] = 100 * df['avg_pts_allowed'] / df['avg_poss_allowed']
    
    # Calculate net rating
    df['net_rating'] = df['offensive_rating'] - df['defensive_rating']
    
    # Sort by net rating
    df = df.sort_values('net_rating', ascending=False)
    
    # Calculate league averages
    league_avg_ortg = df['offensive_rating'].mean()
    league_avg_drtg = df['defensive_rating'].mean()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a colormap based on net rating
    norm = plt.Normalize(df['net_rating'].min(), df['net_rating'].max())
    colors = plt.cm.RdYlGn(norm(df['net_rating']))
    
    # Create scatter plot
    scatter = ax.scatter(
        df['offensive_rating'], 
        df['defensive_rating'],
        c=colors,
        s=100,
        alpha=0.8,
        edgecolor='black',
        linewidth=1
    )
    
    # Add quadrant lines at league average
    ax.axvline(x=league_avg_ortg, color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=league_avg_drtg, color='gray', linestyle='--', alpha=0.7)
    
    # Label the quadrants - CORRECTED POSITIONING PER CLIENT SPECIFICATION
    # Top Right: Good Both, Bottom Right: Bad Defense, Bottom Left: Bad Both, Top Left: Bad Offense
    ax.text(
        df['offensive_rating'].max() - 15, 
        df['defensive_rating'].min() + 2, 
        "Good Offense\nGood Defense",  # Top Right
        fontsize=12,
        weight='bold',
        color='green'
    )
    ax.text(
        df['offensive_rating'].max() - 15, 
        df['defensive_rating'].max() - 5, 
        "Good Offense\nBad Defense",  # Bottom Right
        fontsize=12,
        weight='bold',
        color='orange'
    )
    ax.text(
        df['offensive_rating'].min() + 2, 
        df['defensive_rating'].max() - 5, 
        "Bad Offense\nBad Defense",  # Bottom Left
        fontsize=12,
        weight='bold',
        color='red'
    )
    ax.text(
        df['offensive_rating'].min() + 2, 
        df['defensive_rating'].min() + 2, 
        "Bad Offense\nGood Defense",  # Top Left
        fontsize=12,
        weight='bold',
        color='blue'
    )
    
    # Invert y-axis for defensive rating (lower is better)
    ax.invert_yaxis()
    
    # Add team labels to each point
    for i, row in df.iterrows():
        ax.annotate(
            row['team_name'], 
            (row['offensive_rating'], row['defensive_rating']),
            xytext=(5, 0),
            textcoords='offset points',
            fontsize=9,
            weight='bold'
        )
    
    # Set labels and title
    ax.set_xlabel('Offensive Rating (Points Scored per 100 Possessions)', fontsize=12)
    ax.set_ylabel('Defensive Rating (Points Allowed per 100 Possessions)', fontsize=12)
    ax.set_title('Team Performance: Offensive vs. Defensive Rating', fontsize=16)
    
    # Add a colorbar legend
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='RdYlGn'), ax=ax)
    cbar.set_label('Net Rating', rotation=270, labelpad=15)
    
    # Add grid lines
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Equal aspect ratio for visual clarity
    ax.set_aspect('equal')
    
    # Add annotations for league averages
    ax.text(
        league_avg_ortg + 1, 
        df['defensive_rating'].min() + 2,
        f"League Avg: {league_avg_ortg:.1f}", 
        rotation=90,
        verticalalignment='bottom'
    )
    ax.text(
        df['offensive_rating'].min() + 2, 
        league_avg_drtg + 1,
        f"League Avg: {league_avg_drtg:.1f}"
    )
    
    return fig, df

def display_team_rating_analysis():
    """Display team offensive and defensive rating analysis in Streamlit"""
    st.title("📊 Team Offensive vs. Defensive Rating Analysis")
    
    # Current date/time - UPDATED AS SPECIFIED
    st.markdown(f"*Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-03-29 12:15:38*")
    st.markdown(f"*Current User's Login: Dodga010*")
    
    # Run the analysis
    with st.spinner("Calculating team ratings..."):
        result = analyze_team_o_d_ratings()
        
    if result is None:
        st.error("No team data available.")
        return
        
    fig, df = result
    
    # Display the chart
    st.pyplot(fig)
    
    # Explanation of quadrants - EXACTLY AS SPECIFIED BY CLIENT
    st.subheader("Understanding the Quadrants")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Top Left**: Teams with bad offense but good defense.
        - Teams that rely on defense to win games.
        - Often struggle to score but limit opponent scoring.
        - Defense-first teams.
        """)
        
        st.markdown("""
        **Bottom Left**: Teams with bad offense and bad defense.
        - Typically rebuilding or struggling teams.
        - Negative net rating.
        - Tend to lose most games.
        """)
    
    with col2:
        st.markdown("""
        **Top Right**: Teams with good offense and good defense.
        - The league's elite teams.
        - Positive net rating, championship contenders.
        - Well-balanced teams that excel on both ends.
        """)
        
        st.markdown("""
        **Bottom Right**: Teams with good offense but bad defense.
        - High scoring teams that struggle to get stops.
        - Typically exciting to watch but inconsistent.
        - Often involved in high-scoring games.
        """)
    
    # Display team rankings table
    st.subheader("Team Ratings")
    
    # Format the dataframe for display
    display_df = df[['team_name', 'offensive_rating', 'defensive_rating', 'net_rating']].copy()
    display_df.columns = ['Team', 'Offensive Rating', 'Defensive Rating', 'Net Rating']
    display_df = display_df.sort_values('Net Rating', ascending=False).reset_index(drop=True)
    
    # Add ranking column
    display_df.index = display_df.index + 1
    display_df = display_df.rename_axis('Rank').reset_index()
    
    # Display with formatting
    st.dataframe(
        display_df.style.format({
            'Offensive Rating': '{:.1f}',
            'Defensive Rating': '{:.1f}',
            'Net Rating': '{:.1f}'
        }).background_gradient(subset=['Net Rating'], cmap='RdYlGn')
    )
    
    # Add efficiency insights
    st.subheader("Team Efficiency Insights")
    
    # Most efficient offensive team
    best_offense = df.loc[df['offensive_rating'].idxmax()]
    st.write(f"📈 **Best Offensive Team**: {best_offense['team_name']} ({best_offense['offensive_rating']:.1f} points per 100 possessions)")
    
    # Most efficient defensive team
    best_defense = df.loc[df['defensive_rating'].idxmin()]
    st.write(f"🛡️ **Best Defensive Team**: {best_defense['team_name']} ({best_defense['defensive_rating']:.1f} points allowed per 100 possessions)")
    
    # Best net rating
    best_net = df.loc[df['net_rating'].idxmax()]
    st.write(f"⭐ **Best Overall Team**: {best_net['team_name']} (Net Rating: {best_net['net_rating']:.1f})")
    
    # Most balanced team (closest to the origin in the normalized space)
    df['balance_score'] = abs(df['offensive_rating'] - df['offensive_rating'].mean()) + abs(df['defensive_rating'] - df['defensive_rating'].mean())
    most_balanced = df.loc[df['balance_score'].idxmin()]
    st.write(f"⚖️ **Most Balanced Team**: {most_balanced['team_name']} (closest to league average in both offense and defense)")
	
def analyze_team_performance_trends(team_name):
    """
    Analyze a team's performance against opponents of varying strength
    and compare with how other teams performed against the same opponents
    
    Parameters:
    -----------
    team_name : str
        Name of the team to analyze
    
    Returns:
    --------
    dict
        Dictionary with performance metrics against teams of different strengths
        including similarity comparisons with how other teams performed
    """
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # 1. Calculate team strength ratings for all teams
    strength_query = """
    WITH team_games AS (
        SELECT 
            t1.game_id,
            t1.name AS team_name,
            t2.name AS opponent_name,
            (t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) AS team_score,
            (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score) AS opponent_score,
            ((t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) - 
             (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score)) AS point_diff
        FROM Teams t1
        JOIN Teams t2 ON t1.game_id = t2.game_id AND t1.tm != t2.tm
    )
    SELECT 
        team_name,
        AVG(point_diff) AS avg_point_diff,
        COUNT(DISTINCT game_id) AS games_played
    FROM team_games
    GROUP BY team_name
    """
    
    team_strengths = pd.read_sql_query(strength_query, conn)
    
    # Calculate a simple strength rating based on point differential
    if not team_strengths.empty:
        team_strengths['strength_rating'] = team_strengths['avg_point_diff']
        
        # Normalize ratings to range from 1-10 for easier understanding
        min_rating = team_strengths['strength_rating'].min()
        max_rating = team_strengths['strength_rating'].max()
        range_rating = max_rating - min_rating
        
        if range_rating > 0:  # Avoid division by zero
            team_strengths['strength_rating_normalized'] = 1 + 9 * (team_strengths['strength_rating'] - min_rating) / range_rating
        else:
            team_strengths['strength_rating_normalized'] = 5.0  # Default if all teams have same rating
    else:
        conn.close()
        return None
    
    # 2. Get the selected team's games with performance metrics
    team_games_query = f"""
    WITH team_matches AS (
        SELECT 
            t1.game_id,
            t1.name AS team_name,
            t2.name AS opponent_name,
            (t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) AS team_score,
            (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score) AS opponent_score,
            ((t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) - 
             (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score)) AS point_diff,
            
            -- Team stats for that specific game
            t1.field_goal_percentage AS team_fg_pct,
            t1.three_point_percentage AS team_3pt_pct,
            t1.assists AS team_assists,
            t1.rebounds_total AS team_rebounds,
            t1.turnovers AS team_turnovers
        FROM Teams t1
        JOIN Teams t2 ON t1.game_id = t2.game_id AND t1.tm != t2.tm
        WHERE t1.name = ?
        ORDER BY t1.game_id DESC
    )
    SELECT * FROM team_matches
    LIMIT 15  -- Get last 15 games for analysis
    """
    
    # Get the selected team's recent games
    recent_games = pd.read_sql_query(team_games_query, conn, params=(team_name,))
    
    # 3. NEW: Query to get how ALL TEAMS performed against each opponent
    opponent_performance_query = """
    WITH all_games AS (
        SELECT 
            t1.game_id,
            t1.name AS team_name,
            t2.name AS opponent_name,
            (t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) AS team_score,
            (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score) AS opponent_score,
            ((t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) - 
             (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score)) AS point_diff,
            t1.field_goal_percentage AS team_fg_pct,
            t1.turnovers AS team_turnovers,
            t1.assists AS team_assists
        FROM Teams t1
        JOIN Teams t2 ON t1.game_id = t2.game_id AND t1.tm != t2.tm
    )
    SELECT 
        opponent_name,
        AVG(point_diff) AS avg_point_diff,
        AVG(team_fg_pct) AS avg_fg_pct,
        AVG(team_turnovers) AS avg_turnovers,
        AVG(team_assists) AS avg_assists,
        COUNT(*) AS games_count
    FROM all_games
    GROUP BY opponent_name
    """
    
    # Get how teams typically perform against each opponent
    typical_vs_opponent = pd.read_sql_query(opponent_performance_query, conn)
    conn.close()
    
    if recent_games.empty:
        return None
    
    # 4. Merge team strength data with recent games to analyze performance vs opponent strength
    results = {
        'team_name': team_name,
        'games_analyzed': len(recent_games),
        'team_strength': float(team_strengths[team_strengths['team_name'] == team_name]['strength_rating_normalized'].iloc[0]) 
                         if not team_strengths[team_strengths['team_name'] == team_name].empty else 5.0,
        'games': []
    }
    
    # Calculate average team performance metrics to use as baseline
    avg_performance = {
        'avg_point_diff': recent_games['point_diff'].mean(),
        'avg_fg_pct': recent_games['team_fg_pct'].mean(),
        'avg_3pt_pct': recent_games['team_3pt_pct'].mean(),
        'avg_assists': recent_games['team_assists'].mean(),
        'avg_rebounds': recent_games['team_rebounds'].mean(),
        'avg_turnovers': recent_games['team_turnovers'].mean()
    }
    results['avg_performance'] = avg_performance
    
    # Add opponent strength and compare with typical performance against that opponent
    for _, game in recent_games.iterrows():
        opponent_name = game['opponent_name']
        opponent_strength = float(team_strengths[team_strengths['team_name'] == opponent_name]['strength_rating_normalized'].iloc[0]) \
                         if not team_strengths[team_strengths['team_name'] == opponent_name].empty else 5.0
        
        # Get typical performance of all teams against this opponent
        opponent_typical = typical_vs_opponent[typical_vs_opponent['opponent_name'] == opponent_name]
        
        if not opponent_typical.empty:
            # Calculate how this performance compares to typical performance against this opponent
            typical_point_diff = opponent_typical['avg_point_diff'].iloc[0]
            typical_fg_pct = opponent_typical['avg_fg_pct'].iloc[0]
            typical_turnovers = opponent_typical['avg_turnovers'].iloc[0]
            typical_assists = opponent_typical['avg_assists'].iloc[0]
            
            # Calculate similarity score (how similar this team's performance was to typical performance)
            point_diff_similarity = 1 - min(abs(game['point_diff'] - typical_point_diff) / 20, 1)  # Normalize to 0-1
            fg_pct_similarity = 1 - min(abs(game['team_fg_pct'] - typical_fg_pct) / 20, 1)  # Normalize to 0-1
            turnover_similarity = 1 - min(abs(game['team_turnovers'] - typical_turnovers) / 10, 1)  # Normalize to 0-1
            assists_similarity = 1 - min(abs(game['team_assists'] - typical_assists) / 10, 1)  # Normalize to 0-1
            
            # Weight the factors (you can adjust these weights based on importance)
            similarity_score = (
                0.4 * point_diff_similarity +
                0.3 * fg_pct_similarity +
                0.15 * turnover_similarity +
                0.15 * assists_similarity
            ) * 10  # Scale to 0-10
            
            # Calculate if performance was better or worse than typical
            # Positive means better than typical, negative means worse
            performance_vs_typical = game['point_diff'] - typical_point_diff
        else:
            # If no data for this opponent, use neutral values
            similarity_score = 5.0
            performance_vs_typical = 0.0
            typical_point_diff = 0.0
        
        # Continue with existing performance rating calculation
        expected_pt_diff = results['team_strength'] - opponent_strength
        performance_vs_expected = game['point_diff'] - expected_pt_diff
        
        # Traditional performance rating as before
        if expected_pt_diff > 0:  # Team was expected to win
            if game['point_diff'] > 0:  # Team won as expected
                if game['point_diff'] > expected_pt_diff:  # Won by more than expected
                    performance_rating = min(10, 7 + 3 * (performance_vs_expected / 10))
                else:  # Won by less than expected
                    performance_rating = max(4, 7 - 3 * abs(performance_vs_expected) / 10)
            else:  # Team lost despite being expected to win
                performance_rating = max(1, 4 - 3 * abs(performance_vs_expected) / 15)
        else:  # Team was expected to lose
            if game['point_diff'] > 0:  # Team won despite being expected to lose
                performance_rating = min(10, 7 + 3 * (performance_vs_expected / 5))
            else:  # Team lost as expected
                if game['point_diff'] > expected_pt_diff:  # Lost by less than expected
                    performance_rating = max(4, 7 - 3 * abs(performance_vs_expected) / 15)
                else:  # Lost by more than expected
                    performance_rating = max(1, 4 - 3 * abs(performance_vs_expected) / 20)
        
        # Store both performance ratings and similarity metrics
        game_data = {
            'opponent_name': opponent_name,
            'opponent_strength': opponent_strength,
            'score': f"{int(game['team_score'])} - {int(game['opponent_score'])}",
            'point_diff': int(game['point_diff']),
            'expected_pt_diff': float(expected_pt_diff),
            'typical_pt_diff': float(typical_point_diff),
            'performance_vs_expected': float(performance_vs_expected),
            'performance_vs_typical': float(performance_vs_typical),
            'fg_pct': float(game['team_fg_pct']),
            'performance_rating': float(performance_rating),
            'similarity_score': float(similarity_score),
            'game_id': int(game['game_id'])
        }
        results['games'].append(game_data)
    
    # 5. Calculate performance trends against different opponent strengths
    strength_tiers = {
        'strong_opponents': [g for g in results['games'] if g['opponent_strength'] >= 7],
        'average_opponents': [g for g in results['games'] if 4 <= g['opponent_strength'] < 7],
        'weak_opponents': [g for g in results['games'] if g['opponent_strength'] < 4]
    }
    
    # Calculate average performance for each tier
    tier_performance = {}
    for tier_name, tier_games in strength_tiers.items():
        if tier_games:
            avg_performance = {
                'games_count': len(tier_games),
                'avg_point_diff': sum(g['point_diff'] for g in tier_games) / len(tier_games),
                'avg_performance_rating': sum(g['performance_rating'] for g in tier_games) / len(tier_games),
                'avg_similarity_score': sum(g['similarity_score'] for g in tier_games) / len(tier_games),
                'win_percentage': sum(1 for g in tier_games if g['point_diff'] > 0) / len(tier_games) * 100
            }
            tier_performance[tier_name] = avg_performance
        else:
            tier_performance[tier_name] = {'games_count': 0, 'avg_point_diff': 0, 'avg_performance_rating': 0, 'avg_similarity_score': 0, 'win_percentage': 0}
    
    results['strength_tier_performance'] = tier_performance
    
    # 6. Identify best, worst, most similar, and most unique performances
    if results['games']:
        results['best_performance'] = max(results['games'], key=lambda x: x['performance_rating'])
        results['worst_performance'] = min(results['games'], key=lambda x: x['performance_rating'])
        results['most_similar_to_typical'] = max(results['games'], key=lambda x: x['similarity_score'])
        results['most_unique_from_typical'] = min(results['games'], key=lambda x: x['similarity_score'])
    
    # 7. Calculate performance trend
    if len(results['games']) >= 5:
        recent_5_avg = sum(g['performance_rating'] for g in results['games'][:5]) / 5
        older_avg = sum(g['performance_rating'] for g in results['games'][5:]) / len(results['games'][5:]) if results['games'][5:] else 0
        results['trend'] = 'improving' if recent_5_avg > older_avg else 'declining' if recent_5_avg < older_avg else 'stable'
        results['trend_strength'] = abs(recent_5_avg - older_avg)
    else:
        results['trend'] = 'insufficient_data'
        results['trend_strength'] = 0
    
    return results

def display_team_performance_analysis():
    """Display team performance analysis in Streamlit"""
    st.title("📊 Team Performance Analysis")
    
    # Current date/time and user info
    st.markdown(f"*Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-04-02 16:45:10*")
    st.markdown(f"*Current User's Login: Dodga010*")
    
    # Team selection
    teams = fetch_teams()
    
    if not teams:
        st.error("No team data available.")
        return
    
    selected_team = st.selectbox("Select Team", teams, index=0)
    
    if selected_team:
        # Run team performance analysis
        with st.spinner(f"Analyzing performance trends for {selected_team}..."):
            analysis_results = analyze_team_performance_trends(selected_team)
        
        if not analysis_results:
            st.error(f"No performance data available for {selected_team}.")
            return
        
        # Display team strength overview
        st.header(f"Team Strength Analysis: {selected_team}")
        team_strength = analysis_results['team_strength']
        
        # Create a gauge chart for team strength
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=team_strength,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Team Strength Rating (1-10)"},
            gauge={
                'axis': {'range': [1, 10]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [1, 4], 'color': "lightcoral"},
                    {'range': [4, 7], 'color': "lightyellow"},
                    {'range': [7, 10], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': team_strength
                }
            }
        ))
        
        st.plotly_chart(fig)
        
        # Display recent games performance
        st.subheader("Recent Games Performance")
        
        # Create a data table for recent games
        recent_games_df = pd.DataFrame(analysis_results['games'])
        
        # Add win/loss column
        recent_games_df['Result'] = recent_games_df['point_diff'].apply(lambda x: 'Win' if x > 0 else 'Loss')
        
        # Format for display
        display_df = recent_games_df[['opponent_name', 'score', 'Result', 'opponent_strength', 'performance_rating']].copy()
        display_df.columns = ['Opponent', 'Score', 'Result', 'Opp. Strength (1-10)', 'Performance Rating (1-10)']
        
        # Color code wins and losses
        st.dataframe(
            display_df.style
                .apply(lambda x: ['color: green' if v == 'Win' else 'color: red' for v in x], 
                      subset=['Result'])
                .background_gradient(subset=['Performance Rating (1-10)'], cmap='RdYlGn')
                .background_gradient(subset=['Opp. Strength (1-10)'], cmap='Blues')
                .format({
                    'Opp. Strength (1-10)': '{:.1f}',
                    'Performance Rating (1-10)': '{:.1f}'
                })
        )
        
        # Performance against different opponent strengths
        st.subheader("Performance by Opponent Strength")
        
        # Create columns for the three tiers
        col1, col2, col3 = st.columns(3)
        
        # Display tier stats
        tiers = analysis_results['strength_tier_performance']
        
        with col1:
            st.write("### vs Strong Teams")
            strong_data = tiers['strong_opponents']
            if strong_data['games_count'] > 0:
                st.metric("Games Played", strong_data['games_count'])
                st.metric("Win %", f"{strong_data['win_percentage']:.1f}%")
                st.metric("Avg Point Diff", f"{strong_data['avg_point_diff']:.1f}")
                
                # Rating visualization
                st.progress(strong_data['avg_performance_rating'] / 10)
                st.write(f"Performance Rating: {strong_data['avg_performance_rating']:.1f}/10")
            else:
                st.write("No games against strong teams")
        
        with col2:
            st.write("### vs Average Teams")
            avg_data = tiers['average_opponents']
            if avg_data['games_count'] > 0:
                st.metric("Games Played", avg_data['games_count'])
                st.metric("Win %", f"{avg_data['win_percentage']:.1f}%")
                st.metric("Avg Point Diff", f"{avg_data['avg_point_diff']:.1f}")
                
                # Rating visualization
                st.progress(avg_data['avg_performance_rating'] / 10)
                st.write(f"Performance Rating: {avg_data['avg_performance_rating']:.1f}/10")
            else:
                st.write("No games against average teams")
        
        with col3:
            st.write("### vs Weak Teams")
            weak_data = tiers['weak_opponents']
            if weak_data['games_count'] > 0:
                st.metric("Games Played", weak_data['games_count'])
                st.metric("Win %", f"{weak_data['win_percentage']:.1f}%")
                st.metric("Avg Point Diff", f"{weak_data['avg_point_diff']:.1f}")
                
                # Rating visualization
                st.progress(weak_data['avg_performance_rating'] / 10)
                st.write(f"Performance Rating: {weak_data['avg_performance_rating']:.1f}/10")
            else:
                st.write("No games against weak teams")
        
        # Performance trend
        st.subheader("Performance Trend")
        
        if analysis_results['trend'] != 'insufficient_data':
            trend_text = {
                'improving': "Improving 📈",
                'declining': "Declining 📉",
                'stable': "Stable ↔️"
            }
            
            trend_strength = analysis_results['trend_strength']
            if trend_strength > 1.5:
                trend_desc = "strongly"
            elif trend_strength > 0.7:
                trend_desc = "moderately"
            else:
                trend_desc = "slightly"
            
            st.info(f"{selected_team} is {trend_desc} {trend_text[analysis_results['trend']]} over the last 5 games compared to earlier games.")
            
            # Create trend visualization
            if len(analysis_results['games']) >= 5:
                # Extract performance ratings in chronological order (reverse since newest are first)
                ratings = [g['performance_rating'] for g in analysis_results['games']]
                ratings.reverse()  # Now oldest game first
                
                game_numbers = list(range(1, len(ratings) + 1))
                
                trend_df = pd.DataFrame({
                    'Game Number': game_numbers,
                    'Performance Rating': ratings
                })
                
                fig = px.line(
                    trend_df, 
                    x='Game Number', 
                    y='Performance Rating',
                    title=f"{selected_team} Performance Trend",
                    markers=True,
                    range_y=[1, 10]
                )
                
                # Add a reference line for average performance
                fig.add_hline(
                    y=5, 
                    line_dash="dash", 
                    line_color="gray",
                    annotation_text="Average Performance",
                    annotation_position="bottom right"
                )
                
                st.plotly_chart(fig)
        else:
            st.warning("Insufficient data to determine performance trend. Need at least 5 games.")
        
        # Best and worst performances
        if 'best_performance' in analysis_results and 'worst_performance' in analysis_results:
            st.subheader("Best & Worst Performances")
            
            col1, col2 = st.columns(2)
            
            with col1:
                best = analysis_results['best_performance']
                st.write("### 🌟 Best Performance")
                st.write(f"**Opponent:** {best['opponent_name']} (Strength: {best['opponent_strength']:.1f}/10)")
                st.write(f"**Score:** {best['score']}")
                st.write(f"**Performance Rating:** {best['performance_rating']:.1f}/10")
                st.write(f"**Field Goal %:** {best['fg_pct']:.1f}%")
                
                if best['point_diff'] > best['expected_pt_diff']:
                    st.success(f"Performed {abs(best['performance_vs_expected']):.1f} points better than expected")
            
            with col2:
                worst = analysis_results['worst_performance']
                st.write("### 👎 Worst Performance")
                st.write(f"**Opponent:** {worst['opponent_name']} (Strength: {worst['opponent_strength']:.1f}/10)")
                st.write(f"**Score:** {worst['score']}")
                st.write(f"**Performance Rating:** {worst['performance_rating']:.1f}/10")
                st.write(f"**Field Goal %:** {worst['fg_pct']:.1f}%")
                
                if worst['point_diff'] < worst['expected_pt_diff']:
                    st.error(f"Performed {abs(worst['performance_vs_expected']):.1f} points worse than expected")
        
        # Display recommendations and insights
        st.subheader("📝 Performance Insights")
        
        # Generate insights based on analysis
        insights = []
        
        # Overall team strength insight
        if team_strength >= 7:
            insights.append(f"**Strong Team Rating**: {selected_team} rates as one of the stronger teams with a {team_strength:.1f}/10 rating.")
        elif team_strength <= 4:
            insights.append(f"**Improvement Needed**: {selected_team} currently rates below average with a {team_strength:.1f}/10 rating.")
        else:
            insights.append(f"**Average Team Rating**: {selected_team} rates as an average team with a {team_strength:.1f}/10 rating.")
        
        # Performance against different tiers
        strong_perf = tiers['strong_opponents']['avg_performance_rating'] if tiers['strong_opponents']['games_count'] > 0 else 0
        weak_perf = tiers['weak_opponents']['avg_performance_rating'] if tiers['weak_opponents']['games_count'] > 0 else 0
        
        if strong_perf > 6 and tiers['strong_opponents']['games_count'] > 1:
            insights.append(f"**Strong Against Top Competition**: {selected_team} performs well against strong opponents ({strong_perf:.1f}/10 rating).")
        
        if weak_perf < 5 and tiers['weak_opponents']['games_count'] > 1:
            insights.append(f"**Struggles Against Weaker Teams**: {selected_team} underperforms against weaker opponents ({weak_perf:.1f}/10 rating).")
        
        if analysis_results['trend'] == 'improving' and analysis_results['trend_strength'] > 1.0:
            insights.append(f"**Strong Upward Trajectory**: {selected_team} is showing significant improvement in recent games.")
        elif analysis_results['trend'] == 'declining' and analysis_results['trend_strength'] > 1.0:
            insights.append(f"**Concerning Downward Trend**: {selected_team} is showing a significant decline in performance recently.")
        
        # Display insights
        for insight in insights:
            st.write(insight)
        # Add this section to the display_team_performance_analysis function
        st.subheader("📊 Similarity Analysis")
        st.write("This analysis shows how similarly this team performed compared to how other teams typically perform against the same opponents.")

        # Create a dataframe for similarity analysis
        similarity_df = pd.DataFrame([
            {
                'Opponent': game['opponent_name'],
                'Score': game['score'],
                'Similarity Score (1-10)': round(game['similarity_score'], 1),
                'Performance vs Typical': round(game['performance_vs_typical'], 1)
            } for game in analysis_results['games']
        ])

        # Display similarity table
        st.dataframe(
            similarity_df.style
                .background_gradient(subset=['Similarity Score (1-10)'], cmap='Blues')
                .background_gradient(subset=['Performance vs Typical'], cmap='RdYlGn')
        )

        # Show most similar and most unique games
        col1, col2 = st.columns(2)

        with col1:
            most_similar = analysis_results['most_similar_to_typical']
            st.write("### ✓ Most Typical Performance")
            st.write(f"**Opponent:** {most_similar['opponent_name']}")
            st.write(f"**Score:** {most_similar['score']}")
            st.write(f"**Similarity Score:** {most_similar['similarity_score']:.1f}/10")
            st.write(f"**Compared to typical:** {'+' if most_similar['performance_vs_typical'] > 0 else ''}{most_similar['performance_vs_typical']:.1f} points")

        with col2:
            most_unique = analysis_results['most_unique_from_typical']
            st.write("### ⚡ Most Unique Performance")
            st.write(f"**Opponent:** {most_unique['opponent_name']}")
            st.write(f"**Score:** {most_unique['score']}")
            st.write(f"**Similarity Score:** {most_unique['similarity_score']:.1f}/10")
            st.write(f"**Compared to typical:** {'+' if most_unique['performance_vs_typical'] > 0 else ''}{most_unique['performance_vs_typical']:.1f} points")

        # Add insights about similarity
        st.subheader("🔍 Similarity Insights")

        avg_similarity = sum(game['similarity_score'] for game in analysis_results['games']) / len(analysis_results['games'])
        st.write(f"**Overall Similarity Rating:** {avg_similarity:.1f}/10")

        if avg_similarity > 7:
            st.write(f"✅ **Predictable Performance:** {selected_team} typically performs similarly to how other teams perform against the same opponents.")
        elif avg_similarity < 4:
            st.write(f"⚠️ **Unique Style:** {selected_team} often performs quite differently than how other teams typically perform against the same opponents.")
        else:
            st.write(f"📊 **Mixed Similarity:** {selected_team} sometimes performs like other teams against certain opponents, but has unique performances in other games.")

        # Show games where team performed significantly better than typical
        better_than_typical = [g for g in analysis_results['games'] if g['performance_vs_typical'] > 5]
        if better_than_typical:
            st.write("### 🔼 Games with Significantly Better Than Typical Performance")
            for game in better_than_typical:
                st.write(f"• vs {game['opponent_name']}: {game['performance_vs_typical']:.1f} points better than typical ({game['score']})")

        # Show games where team performed significantly worse than typical
        worse_than_typical = [g for g in analysis_results['games'] if g['performance_vs_typical'] < -5]
        if worse_than_typical:
            st.write("### 🔽 Games with Significantly Worse Than Typical Performance")
            for game in worse_than_typical:
                st.write(f"• vs {game['opponent_name']}: {game['performance_vs_typical']:.1f} points worse than typical ({game['score']})")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import os
import seaborn as sns
import numpy as np
from matplotlib.ticker import MaxNLocator

def fetch_players():
    """Fetch player names from the Shots table"""
    db_path = os.path.join(os.path.dirname(__file__), "database.db")
    
    if not os.path.exists(db_path):
        return []

    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT player_name FROM Shots WHERE player_name IS NOT NULL ORDER BY player_name;"
    players = pd.read_sql_query(query, conn)["player_name"].tolist()
    conn.close()
    return players

def fetch_teams():
    """Fetch team names from the Teams table"""
    db_path = os.path.join(os.path.dirname(__file__), "database.db")
    
    if not os.path.exists(db_path):
        return []
        
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT name FROM Teams ORDER BY name;"
    teams = pd.read_sql_query(query, conn)["name"].tolist()
    conn.close()
    return teams

def fetch_shot_types():
    """Fetch available shot types from the Shots table"""
    db_path = os.path.join(os.path.dirname(__file__), "database.db")
    
    if not os.path.exists(db_path):
        return []
        
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT shot_sub_type FROM Shots WHERE shot_sub_type IS NOT NULL ORDER BY shot_sub_type;"
    shot_types = pd.read_sql_query(query, conn)["shot_sub_type"].tolist()
    conn.close()
    return shot_types

def fetch_shooting_fouls_for_entities(entity_names, entity_type='player', selected_shot_types=None):
    """
    Fetch data about shooting fouls for specific players or teams, filtered by shot types.
    
    Parameters:
    - entity_names: List of player or team names to analyze
    - entity_type: 'player' or 'team' 
    - selected_shot_types: List of shot types to include (None for all)
    
    Returns pandas DataFrame with total shots and fouls by player/team
    """
    if not entity_names:
        return pd.DataFrame()
        
    db_path = os.path.join(os.path.dirname(__file__), "database.db")
    conn = sqlite3.connect(db_path)
    
    # Create SQL string for entity names (safely quote each name)
    entities_str = "', '".join([name.replace("'", "''") for name in entity_names])
    
    # Shot type filter condition
    shot_type_condition = ""
    if selected_shot_types:
        shot_types_str = "', '".join([st.replace("'", "''") for st in selected_shot_types])
        shot_type_condition = f"AND s.shot_sub_type IN ('{shot_types_str}')"
    
    if entity_type == 'player':
        # Query to get shots and fouls for specific players
        query = f"""
        SELECT 
            s.player_name AS entity_name,
            COUNT(DISTINCT s.shot_id) AS total_shots,
            SUM(CASE WHEN EXISTS (
                SELECT 1 FROM PlayByPlay p 
                WHERE p.game_id = s.game_id 
                AND p.action_number BETWEEN s.action_number AND s.action_number + 2
                AND p.action_type = 'foul'
                AND p.sub_type = 'personal'
                AND (
                    p.qualifiers LIKE '%shooting%1freethrow%' OR 
                    p.qualifiers LIKE '%shooting%2freethrow%' OR 
                    p.qualifiers LIKE '%shooting%3freethrow%'
                )
            ) THEN 1 ELSE 0 END) AS shooting_fouls_count
        FROM Shots s
        WHERE s.player_name IN ('{entities_str}')
        {shot_type_condition}
        GROUP BY s.player_name
        """
    else:  # team
        # Query to get shots and fouls for specific teams
        query = f"""
        SELECT 
            t.name AS entity_name,
            COUNT(DISTINCT s.shot_id) AS total_shots,
            SUM(CASE WHEN EXISTS (
                SELECT 1 FROM PlayByPlay p 
                WHERE p.game_id = s.game_id 
                AND p.action_number BETWEEN s.action_number AND s.action_number + 2
                AND p.action_type = 'foul'
                AND p.sub_type = 'personal'
                AND (
                    p.qualifiers LIKE '%shooting%1freethrow%' OR 
                    p.qualifiers LIKE '%shooting%2freethrow%' OR 
                    p.qualifiers LIKE '%shooting%3freethrow%'
                )
            ) THEN 1 ELSE 0 END) AS shooting_fouls_count
        FROM Shots s
        JOIN Teams t ON s.team_id = t.tm AND s.game_id = t.game_id
        WHERE t.name IN ('{entities_str}')
        {shot_type_condition}
        GROUP BY t.name
        """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Calculate derived metrics
    if not df.empty:
        df['fouls_per_shot'] = df['shooting_fouls_count'] / df['total_shots']
        df['foul_percentage'] = df['fouls_per_shot'] * 100
    
    return df

def create_scatter_plot(df, entity_type='player', shot_types_label="selected shot types"):
    """
    Create a scatter plot showing shooting fouls vs shot attempts
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create the scatter plot with points sized by foul percentage
    scatter = ax.scatter(
        df['total_shots'], 
        df['shooting_fouls_count'],
        s=df['foul_percentage'] * 10,  # Size points by foul percentage 
        alpha=0.6,
        c=df['foul_percentage'],  # Color by foul percentage
        cmap='viridis',
    )
    
    # Add labels for all points
    for _, row in df.iterrows():
        ax.annotate(
            row['entity_name'],
            xy=(row['total_shots'], row['shooting_fouls_count']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
        )
    
    # Add a colorbar to show the foul percentage scale
    cbar = plt.colorbar(scatter)
    cbar.set_label('Foul Percentage (%)', rotation=270, labelpad=15)
    
    # Set labels and title
    entity_label = "Players" if entity_type == 'player' else "Teams"
    ax.set_xlabel(f'Total Shot Attempts ({shot_types_label})', fontsize=12)
    ax.set_ylabel('Shooting Fouls Drawn', fontsize=12)
    ax.set_title(f'Shooting Fouls vs Shot Attempts for Selected {entity_label}', fontsize=14)
    
    # Force y-axis to use integers only (since fouls are whole numbers)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Add a reference line showing average foul rate
    if len(df) > 0:
        avg_foul_rate = df['shooting_fouls_count'].sum() / df['total_shots'].sum() if df['total_shots'].sum() > 0 else 0
        x_range = [0, df['total_shots'].max() * 1.05]
        y_values = [x * avg_foul_rate for x in x_range]
        ax.plot(x_range, y_values, 'r--', alpha=0.7, label=f'Avg Foul Rate: {avg_foul_rate:.3f}')
    
    # Add a grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    return fig

def display_shooting_fouls_analysis():
    """
    Display interactive analysis of shooting fouls
    """
    st.title("Shooting Fouls Analysis Dashboard")
    
    # Create sidebar for filters
    st.sidebar.header("Analysis Options")
    
    # Choose between team or player analysis
    entity_type = st.sidebar.radio("Select Analysis Type:", ["Players", "Teams"])
    entity_type_value = 'player' if entity_type == "Players" else 'team'
    
    # Get available player/team lists
    all_players = fetch_players()
    all_teams = fetch_teams()
    
    # Select specific entities to analyze
    if entity_type == "Players":
        if all_players:
            selected_entities = st.sidebar.multiselect(
                "Select Players to Compare:", 
                options=all_players,
                default=all_players[:min(5, len(all_players))]
            )
        else:
            st.error("No players found in the database.")
            return
    else:  # Teams
        if all_teams:
            selected_entities = st.sidebar.multiselect(
                "Select Teams to Compare:",
                options=all_teams,
                default=all_teams
            )
        else:
            st.error("No teams found in the database.")
            return
    
    # Get shot types for filtering
    all_shot_types = fetch_shot_types()
    
    # Shot type filter
    st.sidebar.subheader("Shot Type Filter")
    shot_filter_option = st.sidebar.radio("Shot Types to Include:", ["All Shot Types", "Selected Shot Types"])
    
    selected_shot_types = None
    shot_types_label = "all shot types"
    
    if shot_filter_option == "Selected Shot Types" and all_shot_types:
        selected_shot_types = st.sidebar.multiselect(
            "Select Shot Types:", 
            options=all_shot_types,
            default=all_shot_types[:min(3, len(all_shot_types))]
        )
        if selected_shot_types:
            shot_types_label = ", ".join(selected_shot_types)
    
    # Only proceed if entities are selected
    if not selected_entities:
        st.warning(f"Please select at least one {entity_type.lower()[:-1]} to analyze.")
        return
    
    # Fetch data based on selections
    df = fetch_shooting_fouls_for_entities(
        entity_names=selected_entities, 
        entity_type=entity_type_value, 
        selected_shot_types=selected_shot_types
    )
    
    if not df.empty:
        # Summary statistics
        total_shots = df['total_shots'].sum()
        total_fouls = df['shooting_fouls_count'].sum()
        overall_foul_rate = (total_fouls / total_shots) if total_shots > 0 else 0
        
        # Create metrics at the top
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"Total Shots ({shot_filter_option.lower()})", f"{total_shots:,}")
        with col2:
            st.metric("Total Shooting Fouls", f"{total_fouls:,}")
        with col3:
            st.metric("Overall Foul Rate", f"{overall_foul_rate:.3f} ({overall_foul_rate*100:.1f}%)")
        
        # Create and display the scatter plot
        st.subheader(f"Shooting Fouls vs Shot Attempts for Selected {entity_type}")
        
        fig = create_scatter_plot(df, entity_type_value, shot_types_label)
        st.pyplot(fig)
        
        # Add explanation of the visualization
        st.info("""
        **How to read this chart:**
        - Each dot represents a player or team that you selected
        - X-axis shows the total number of shot attempts (based on selected shot types)
        - Y-axis shows the total number of shooting fouls drawn
        - The size and color of each dot represent the foul drawing rate (fouls per shot)
        - The dashed red line shows the average foul rate across selected players/teams
        - Players/teams above the line are more effective at drawing fouls than average
        """)
        
        # Display the data table with sorting options
        st.subheader("Detailed Data")
        
        # Add sorting options
        sort_by = st.selectbox(
            "Sort by:",
            ["Total Shots", "Shooting Fouls", "Foul Drawing Rate"]
        )
        
        sort_column_map = {
            "Total Shots": "total_shots",
            "Shooting Fouls": "shooting_fouls_count",
            "Foul Drawing Rate": "fouls_per_shot"
        }
        
        # Format the dataframe for better display
        display_df = df.copy().sort_values(sort_column_map[sort_by], ascending=False)
        display_df['foul_percentage'] = display_df['foul_percentage'].round(2).astype(str) + '%'
        display_df = display_df.rename(columns={
            'entity_name': entity_type.rstrip('s'),  # Remove plural 's' 
            'total_shots': 'Total Shots',
            'shooting_fouls_count': 'Shooting Fouls',
            'fouls_per_shot': 'Fouls per Shot',
            'foul_percentage': 'Foul Rate'
        })
        
        st.dataframe(display_df)
        
        # Top performers analysis
        if len(df) > 1:  # Only show comparisons if we have multiple entities
            st.subheader("Comparison Insights")
            
            top_fouls = df.nlargest(1, 'shooting_fouls_count').iloc[0]
            top_foul_rate = df.nlargest(1, 'fouls_per_shot').iloc[0]
            
            entity_term = "player" if entity_type_value == 'player' else "team"
            
            st.write(f"• Among selected {entity_type.lower()}, the {entity_term} with the **most shooting fouls drawn** is "
                    f"**{top_fouls['entity_name']}** with "
                    f"**{top_fouls['shooting_fouls_count']}** fouls on {top_fouls['total_shots']} shots.")
            
            st.write(f"• The {entity_term} with the **highest foul-drawing rate** is "
                    f"**{top_foul_rate['entity_name']}** with "
                    f"**{top_foul_rate['fouls_per_shot']:.3f}** fouls per shot "
                    f"({top_foul_rate['foul_percentage']:.1f}%).")
    else:
        st.warning("No data available for the selected entities and shot types. Try different selections.")

# To integrate with your main application, add this function to the main menu options
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import os
import json
from scipy.stats import pearsonr

db_path = os.path.join(os.path.dirname(__file__), "database.db")

def fetch_player_fouls_and_minutes(shooting_fouls_only=False):
    """Fetch fouls committed on players and their total minutes played"""
    conn = sqlite3.connect(db_path)
    
    # Query to count fouls on each player - with option for shooting fouls only
    if shooting_fouls_only:
        fouls_query = """
        SELECT p.first_name || ' ' || p.last_name AS player_name, COUNT(*) AS fouls_count,
               t.name as team_name
        FROM PlayByPlay pbp
        JOIN Players p ON pbp.player_id = p.json_player_id AND pbp.game_id = p.game_id
        JOIN Teams t ON p.team_id = t.tm AND p.game_id = t.game_id
        JOIN PlayByPlay pbp_prev ON pbp.previous_action = pbp_prev.action_number 
                                  AND pbp.game_id = pbp_prev.game_id
                                  AND pbp_prev.action_type = 'foul'
        WHERE pbp.action_type = 'foulon'
        AND (pbp_prev.qualifiers LIKE '%shooting%' OR pbp_prev.qualifiers LIKE '%2freethrow%' OR pbp_prev.qualifiers LIKE '%3freethrow%')
        GROUP BY player_name, team_name
        """
    else:
        fouls_query = """
        SELECT p.first_name || ' ' || p.last_name AS player_name, COUNT(*) AS fouls_count,
               t.name as team_name
        FROM PlayByPlay pbp
        JOIN Players p ON pbp.player_id = p.json_player_id AND pbp.game_id = p.game_id
        JOIN Teams t ON p.team_id = t.tm AND p.game_id = t.game_id
        WHERE pbp.action_type = 'foulon'
        GROUP BY player_name, team_name
        """
    
    # Query to sum minutes played by each player
    minutes_query = """
    SELECT first_name || ' ' || last_name AS player_name, minutes_played, p.game_id,
           t.name as team_name
    FROM Players p
    JOIN Teams t ON p.team_id = t.tm AND p.game_id = t.game_id
    """
    
    # Query to count fouls committed BY each player
    fouls_by_player_query = """
    SELECT p.first_name || ' ' || p.last_name AS player_name, COUNT(*) AS fouls_committed,
           t.name as team_name
    FROM PlayByPlay pbp
    JOIN Players p ON pbp.player_id = p.json_player_id AND pbp.game_id = p.game_id
    JOIN Teams t ON p.team_id = t.tm AND p.game_id = t.game_id
    WHERE pbp.action_type = 'foul'
    GROUP BY player_name, team_name
    """
    
    # Execute queries
    fouls_df = pd.read_sql_query(fouls_query, conn)
    minutes_df = pd.read_sql_query(minutes_query, conn)
    fouls_by_player_df = pd.read_sql_query(fouls_by_player_query, conn)
    
    conn.close()
    
    # Process minutes data - convert from MM:SS format to decimal minutes
    def convert_minutes(time_str):
        if pd.isna(time_str) or time_str == '':
            return 0
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) + int(parts[1])/60
        return 0
    
    minutes_df['minutes_decimal'] = minutes_df['minutes_played'].apply(convert_minutes)
    
    # Sum minutes by player across all games
    total_minutes = minutes_df.groupby(['player_name', 'team_name'])['minutes_decimal'].sum().reset_index()
    
    # Merge fouls and minutes data
    result = pd.merge(fouls_df, total_minutes, on=['player_name', 'team_name'], how='outer')
    result.fillna({'fouls_count': 0, 'minutes_decimal': 0}, inplace=True)
    
    # Add fouls committed BY player
    result = pd.merge(result, fouls_by_player_df, on=['player_name', 'team_name'], how='left')
    result.fillna({'fouls_committed': 0}, inplace=True)
    
    # Calculate fouls per 40 minutes (fouls received)
    result['fouls_per_40'] = np.where(
        result['minutes_decimal'] > 0,
        (result['fouls_count'] / result['minutes_decimal']) * 40,
        0
    )
    
    # Calculate fouls committed per 40 minutes
    result['fouls_committed_per_40'] = np.where(
        result['minutes_decimal'] > 0,
        (result['fouls_committed'] / result['minutes_decimal']) * 40,
        0
    )
    
    return result

def fetch_teams():
    """Fetch all team names from the database"""
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT name FROM Teams ORDER BY name"
    teams = pd.read_sql_query(query, conn)
    conn.close()
    return teams['name'].tolist()

def calculate_foul_correlation(data):
    """Calculate correlation between fouls committed and fouls received"""
    # Filter to players with sufficient minutes
    valid_data = data[data['minutes_decimal'] >= 10]  # Minimum 10 minutes to be included
    
    if len(valid_data) < 5:
        return None, 0, 0  # Not enough data points for meaningful correlation
    
    # Calculate correlation coefficient
    correlation, p_value = pearsonr(valid_data['fouls_committed_per_40'], valid_data['fouls_per_40'])
    
    return correlation, p_value, len(valid_data)

def plot_foul_correlation(data, title_prefix=""):
    """Create a scatter plot showing relationship between fouls committed and received"""
    valid_data = data[data['minutes_decimal'] >= 10]
    
    if len(valid_data) < 5:
        return None, "Not enough data for correlation analysis (need at least 5 players with 10+ minutes)"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    scatter = ax.scatter(valid_data['fouls_committed_per_40'], valid_data['fouls_per_40'])
    
    # Add player labels
    for _, row in valid_data.iterrows():
        ax.annotate(row['player_name'], 
                   (row['fouls_committed_per_40'], row['fouls_per_40']),
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=8)
    
    # Add trend line
    z = np.polyfit(valid_data['fouls_committed_per_40'], valid_data['fouls_per_40'], 1)
    p = np.poly1d(z)
    ax.plot(valid_data['fouls_committed_per_40'], p(valid_data['fouls_committed_per_40']), 
            "r--", alpha=0.8, label=f"Trend line: y = {z[0]:.2f}x + {z[1]:.2f}")
    
    # Calculate and display correlation
    corr, p_value, n = calculate_foul_correlation(valid_data)
    ax.set_xlabel('Fouls Committed per 40 Minutes')
    ax.set_ylabel('Fouls Received per 40 Minutes')
    ax.set_title(f'{title_prefix}Correlation between Fouls Committed and Received\n(r = {corr:.3f}, p = {p_value:.3f}, n = {n})')
    ax.grid(True)
    ax.legend()
    
    return fig, None

def display_player_fouls_analysis():
    """Display visualization of fouls vs minutes played"""
    st.title("Player Fouls Analysis")
    st.subheader("Analysis of fouls committed on players vs. minutes played")
    
    # Add shooting fouls toggle button
    shooting_fouls_only = st.checkbox("Show Only Shooting Fouls", False)
    
    # Fetch data with appropriate filter
    data = fetch_player_fouls_and_minutes(shooting_fouls_only)
    
    # Create team dropdown
    teams = fetch_teams()
    selected_team = st.selectbox("Select a team to view all its players:", [""] + teams)
    
    # Create player dropdown
    all_players = sorted(data['player_name'].unique())
    
    # If a team is selected, get all players from that team
    team_players = []
    if selected_team:
        team_players = data[data['team_name'] == selected_team]['player_name'].unique().tolist()
    
    # Use the team's players as default if a team is selected, otherwise empty list
    default_selection = team_players if selected_team else []
    
    # Allow user to modify the selection
    selected_players = st.multiselect("Select players to display:", all_players, default=default_selection)
    
    # Show foul correlation option
    show_correlation = st.checkbox("Show Correlation Analysis", False)
    
    if selected_players:
        # Filter data and remove players with no minutes played
        filtered_data = data[data['player_name'].isin(selected_players)]
        valid_data = filtered_data[filtered_data['minutes_decimal'] > 0]
        
        if not valid_data.empty:
            # First chart - Minutes vs Fouls Received
            st.subheader("Minutes Played vs Fouls Received")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the scatter points
            scatter = ax.scatter(valid_data['minutes_decimal'], valid_data['fouls_count'])
            
            # Add labels for each point
            for _, row in valid_data.iterrows():
                ax.annotate(row['player_name'], 
                            (row['minutes_decimal'], row['fouls_count']),
                            xytext=(5, 5), 
                            textcoords='offset points')
            
            # Set appropriate axis ranges
            min_minutes = valid_data['minutes_decimal'].min() * 0.9  # Give 10% padding
            max_minutes = valid_data['minutes_decimal'].max() * 1.1
            min_fouls = valid_data['fouls_count'].min() * 0.9
            max_fouls = valid_data['fouls_count'].max() * 1.1
            
            # Ensure minimal values for visualization
            min_minutes = max(0, min_minutes)
            min_fouls = max(0, min_fouls)
            
            ax.set_xlim(min_minutes, max_minutes)
            ax.set_ylim(min_fouls, max_fouls)
            
            # Calculate and add mean line using only the valid data points range
            mean_fouls_per_minute = valid_data['fouls_count'].sum() / valid_data['minutes_decimal'].sum()
            
            # Plot mean line across the visible area of the chart
            x_line = np.array([min_minutes, max_minutes])
            y_line = mean_fouls_per_minute * x_line
            fouls_type = "Shooting Fouls" if shooting_fouls_only else "All Fouls"
            ax.plot(x_line, y_line, 'r--', label=f'Mean {fouls_type}/Min: {mean_fouls_per_minute:.4f}')
            ax.legend()
            
            ax.set_xlabel('Total Minutes Played')
            ax.set_ylabel(f'Number of {fouls_type} on Player')
            ax.set_title(f'Minutes Played vs {fouls_type} Committed on Players')
            ax.grid(True)
            
            # Show plot
            st.pyplot(fig)
            
            # Show data table
            st.subheader(f"Selected Player Data - {fouls_type}")
            display_df = filtered_data[['player_name', 'team_name', 'minutes_decimal', 
                                       'fouls_count', 'fouls_per_40', 
                                       'fouls_committed', 'fouls_committed_per_40']].copy()
            fouls_type_label = "Shooting Fouls" if shooting_fouls_only else "Fouls" 
            display_df.columns = ['Player', 'Team', 'Minutes Played', 
                                 f'{fouls_type_label} Received', f'{fouls_type_label} Received per 40 min', 
                                 f'{fouls_type_label} Committed', f'{fouls_type_label} Committed per 40 min']
            display_df = display_df.sort_values(by=f'{fouls_type_label} Received per 40 min', ascending=False)
            st.dataframe(display_df)
            
            # Show correlation analysis if requested
            if show_correlation:
                st.subheader("Correlation Analysis: Fouls Committed vs. Fouls Received")
                
                corr_fig, error_msg = plot_foul_correlation(filtered_data, 
                                                           f"{'Shooting ' if shooting_fouls_only else ''}")
                if corr_fig:
                    st.pyplot(corr_fig)
                    
                    # Add correlation explanation
                    corr, p_value, n = calculate_foul_correlation(filtered_data)
                    
                    if p_value < 0.05:
                        significance = "statistically significant"
                    else:
                        significance = "not statistically significant"
                        
                    if corr > 0.7:
                        strength = "strong positive"
                    elif corr > 0.3:
                        strength = "moderate positive"
                    elif corr > 0.1:
                        strength = "weak positive"
                    elif corr > -0.1:
                        strength = "no"
                    elif corr > -0.3:
                        strength = "weak negative"
                    elif corr > -0.7:
                        strength = "moderate negative" 
                    else:
                        strength = "strong negative"
                        
                    st.write(f"There is a **{strength} correlation** ({corr:.3f}) between fouls committed and fouls received " +
                            f"per 40 minutes for the selected players. This correlation is {significance} (p = {p_value:.3f}).")
                    
                    if corr > 0.3 and p_value < 0.05:
                        st.write("This suggests that players who commit more fouls also tend to receive more fouls.")
                    elif corr < -0.3 and p_value < 0.05:
                        st.write("This suggests that players who commit fewer fouls tend to receive more fouls, or vice versa.")
                    
                    st.write("Note: Correlation analysis includes only players with at least 10 minutes played.")
                else:
                    st.warning(error_msg)
        else:
            st.warning(f"No data points with valid minutes played found for the selected players{' and shooting fouls filter' if shooting_fouls_only else ''}.")
    else:
        st.info("Please select one or more players to display the analysis")

def fetch_team_recent_games(team_name, limit=5):
    db_path = os.path.join(os.path.dirname(__file__), "database.db")
    conn = sqlite3.connect(db_path)
    query = """
    SELECT t1.game_id, t2.name as opponent, 
           (t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) as team_score, 
           (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score) as opponent_score,
           t1.field_goal_percentage, t1.three_point_percentage, t1.rebounds_total, t1.assists, t1.turnovers
    FROM Teams t1
    JOIN Teams t2 ON t1.game_id = t2.game_id AND t2.name != t1.name
    WHERE t1.name = ?
    ORDER BY t1.game_id DESC
    LIMIT ?
    """
    df = pd.read_sql_query(query, conn, params=(team_name, limit))
    conn.close()
    return df

def fetch_team_strengths(team_name):
    # This can be expanded for deeper scouting
    db_path = os.path.join(os.path.dirname(__file__), "database.db")
    conn = sqlite3.connect(db_path)
    query = """
    SELECT AVG(field_goal_percentage) as avg_fg_pct,
           AVG(three_point_percentage) as avg_3p_pct,
           AVG(rebounds_total) as avg_rebounds,
           AVG(assists) as avg_assists,
           AVG(turnovers) as avg_turnovers
    FROM Teams WHERE name = ?
    """
    df = pd.read_sql_query(query, conn, params=(team_name,))
    conn.close()
    return df.iloc[0] if not df.empty else None

def display_prematch_help():
    st.title("📋 Prematch Help: Team Scouting & Preparation")
    teams = fetch_teams()
    team = st.selectbox("Select your team", teams)
    opponents = [t for t in teams if t != team]
    opponent = st.selectbox("Select opponent", opponents)

    st.subheader(f"Recent Form: {team}")
    recent_games = fetch_team_recent_games(team)
    if not recent_games.empty:
        st.dataframe(recent_games)
        wins = (recent_games['team_score'] > recent_games['opponent_score']).sum()
        losses = (recent_games['team_score'] < recent_games['opponent_score']).sum()
        st.metric("Last 5 Games", f"{wins}W - {losses}L")
    else:
        st.info("No recent games found.")

    st.subheader(f"Team Strengths and Weaknesses")
    strengths = fetch_team_strengths(team)
    if strengths is not None:
        st.write(f"**Field Goal %:** {strengths['avg_fg_pct']:.1f}")
        st.write(f"**Three Point %:** {strengths['avg_3p_pct']:.1f}")
        st.write(f"**Rebounds:** {strengths['avg_rebounds']:.1f}")
        st.write(f"**Assists:** {strengths['avg_assists']:.1f}")
        st.write(f"**Turnovers:** {strengths['avg_turnovers']:.1f}")

        # Simple suggestions based on stats
        suggestions = []
        if strengths['avg_3p_pct'] > 35:
            suggestions.append("Your team is strong from 3PT range. Consider running plays for open threes.")
        if strengths['avg_rebounds'] > 35:
            suggestions.append("Strong rebounding team – crash the boards aggressively.")
        if strengths['avg_turnovers'] > 15:
            suggestions.append("High turnovers detected. Focus on ball security.")
        if not suggestions:
            suggestions.append("Team profile is balanced. Play to your strengths.")

        st.subheader("Suggested Focus:")
        for tip in suggestions:
            st.write(f"• {tip}")

    st.subheader(f"Head-to-Head vs {opponent}")
    h2h_query = """
    SELECT t1.game_id, 
           (t1.p1_score + t1.p2_score + t1.p3_score + t1.p4_score) as team_score,
           (t2.p1_score + t2.p2_score + t2.p3_score + t2.p4_score) as opponent_score
    FROM Teams t1
    JOIN Teams t2 ON t1.game_id = t2.game_id AND t2.name = ?
    WHERE t1.name = ?
    ORDER BY t1.game_id DESC
    """
    db_path = os.path.join(os.path.dirname(__file__), "database.db")
    conn = sqlite3.connect(db_path)
    h2h_df = pd.read_sql_query(h2h_query, conn, params=(opponent, team))
    conn.close()
    if not h2h_df.empty:
        st.dataframe(h2h_df)
        h2h_wins = (h2h_df['team_score'] > h2h_df['opponent_score']).sum()
        h2h_losses = (h2h_df['team_score'] < h2h_df['opponent_score']).sum()
        st.metric("Head-to-Head", f"{h2h_wins}W - {h2h_losses}L")
    else:
        st.info("No head-to-head games found.")

    st.subheader("Prematch Checklist")
    st.markdown("""
    - Review opponent's recent games
    - Discuss defensive matchups
    - Confirm starting lineup
    - Set tactical priorities (e.g., defend 3PT, attack paint)
    - Remind players about foul discipline & ball security
    """)

def main():
    st.title("🏀 Basketball Stats Viewer")
    page = st.sidebar.selectbox("📌 Choose a page", ["Team Season Boxscore", "Shot Chart","Match report", "Four Factors", "Lebron", "Play by Play", "Match Detail", "Five Player Segments", "Team Lineup Analysis", "Shooting Foul Analysis", "Prematch Help"])

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
    elif page == "Five Player Segments":
        display_five_player_segments()
    # ... rest of your main function ...
    elif page == "Shooting Foul Analysis":
        display_shooting_fouls_analysis()
        display_player_fouls_analysis()
    elif page == "Team Lineup Analysis":
        display_team_analysis()
    elif page == "Prematch Help":
        display_prematch_help()
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

        display_four_factors_analysis()
        display_basketball_stats_win_analysis()
        display_team_comparison_analysis()
        display_advanced_metrics_analysis()
        display_team_shooting_analysis()

    elif page == "Match report":
        display_match_report()

    elif page == "In Game":
        display_in_game_page()

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
            display_team_rating_analysis()
            display_team_performance_analysis()
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
        st.subheader("🎯 Shot Chart Analysis")
    
        # Create two columns for layout
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # Player selection
            players = fetch_players()
            if not players:
                st.warning("No player data available.")
            else:
                player_name = st.selectbox("Select a Player", players)
                
                # Add filters
                st.write("### Filters")
                shot_type = st.multiselect(
                    "Shot Type",
                    ["2PT", "3PT", "All"],
                    default=["All"]
                )
                
                show_heatmap = st.checkbox("Show Shot Density Heatmap", value=False)
                
        with col2:
            if player_name:
                # Main shot chart
                st.write(f"### Shot Chart for {player_name}")
                processed_shots, zone_stats, fig = generate_improved_shot_chart(player_name=player_name)
                st.pyplot(fig)  # Display the figure

                # Display zone stats
                st.write("### Shot Zone Analysis")
                st.dataframe(zone_stats)

                # Display misclassification info
                misclassified_count = processed_shots['type_mismatch'].sum()
                if misclassified_count > 0:
                    st.warning(f"{misclassified_count} shots ({misclassified_count/len(processed_shots)*100:.1f}%) have shot type mismatches")
                                
                # Shot distribution charts
                st.write("### Shot Distribution Analysis")
                col_dist1, col_dist2 = st.columns(2)
                
                plot_fg_percentage_with_frequency(player_name)
                
                plot_interpolated_distribution(player_name)
                
                # Player stats section
                st.write("### Player Statistics")
                
                # Game-by-game stats in an expander
                with st.expander("Game by Game Statistics", expanded=False):
                    player_game_stats = fetch_player_game_stats(player_name)
                    if not player_game_stats.empty:
                        st.dataframe(player_game_stats.style.format({
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
                
                # Comparison with league averages
                with st.expander("League Comparison", expanded=False):
                    player_vs_league_40 = fetch_player_and_league_stats_per_40(player_name)
                    if not player_vs_league_40.empty:
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
                        st.warning(f"No comparison stats available for {player_name}.")

                # Show detailed shot analysis
                analyze_shot_patterns(player_name)

if __name__ == "__main__":
    main()
