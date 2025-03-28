
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
    player_impact_stats = defaultdict(lambda: {'total_plus_minus': 0, 'total_actions': 0, 'lineups': 0})
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
                
                # Group by lineup for this game
                game_lineup_stats = team_stats.groupby('lineup_str').agg({
                    'Duration': 'sum',
                    'net_points': 'sum'
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
                                player_impact_stats[player]['total_plus_minus'] += row['net_points']
                                player_impact_stats[player]['total_actions'] += row['Duration']
                                player_impact_stats[player]['lineups'] += 1
                
                all_lineup_stats.append(game_lineup_stats)
    
    if not all_lineup_stats:
        return None, [], {}, [], len(games)
    
    # Combine all games stats
    all_stats = pd.concat(all_lineup_stats, ignore_index=True)
    
    # Calculate aggregate statistics
    lineup_stats = all_stats.groupby(['lineup_str', 'team_name']).agg({
        'Duration': 'sum',
        'net_points': 'sum'
    }).reset_index()
    
    # Add games played and team info
    lineup_stats['Games Played'] = lineup_stats['lineup_str'].apply(lambda x: len(lineup_games[x]))
    lineup_stats['Teams'] = lineup_stats['lineup_str'].apply(lambda x: ' | '.join(sorted(lineup_teams[x])))
    
    # Rename net_points to Plus/Minus
    lineup_stats = lineup_stats.rename(columns={'net_points': 'Plus/Minus'})
    
    # Calculate additional metrics
    lineup_stats['Reliability'] = (lineup_stats['Duration'] / lineup_stats['Duration'].max() * 100).round(1)
    lineup_stats['Avg Plus/Minus per Game'] = (lineup_stats['Plus/Minus'] / lineup_stats['Games Played']).round(2)
    lineup_stats['Plus/Minus per 100'] = (lineup_stats['Plus/Minus'] / lineup_stats['Duration'] * 100).round(2)
    
    # Rename columns for clarity
    lineup_stats = lineup_stats.rename(columns={
        'lineup_str': 'Lineup',
        'team_name': 'Primary Team',
        'Duration': 'Total Actions'
    })
    
    # Arrange columns in desired order
    lineup_stats = lineup_stats[[
        'Lineup', 
        'Primary Team', 
        'Teams',
        'Total Actions',
        'Games Played',
        'Plus/Minus',
        'Avg Plus/Minus per Game',
        'Plus/Minus per 100',
        'Reliability'
    ]]
    
    return lineup_stats, sorted(list(all_players)), player_impact_stats, teams, len(games)

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

 # After displaying the main lineup stats, add:
    if not stats.empty:
        display_pattern_analysis(stats, player_impact)

    # Add after your main statistics display:
    if not stats.empty:
        display_player_impact_analysis(filtered_stats) 

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
        

def main():
    st.title("🏀 Basketball Stats Viewer")
    page = st.sidebar.selectbox("📌 Choose a page", ["Team Season Boxscore", "Shot Chart","Match report", "Four Factors", "Lebron", "Play by Play", "Match Detail", "Five Player Segments", "Team Lineup Analysis"])

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
    elif page == "Team Lineup Analysis":
        display_team_analysis()
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
                generate_shot_chart(player_name, show_heatmap=show_heatmap, shot_types=shot_type)
                
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
