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
    df_shots['y_coord'] = 261 - (df_shots['y_coord'] * 2.61)

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

def main():
    st.title("🏀 Basketball Stats Viewer")
    page = st.sidebar.selectbox("📌 Choose a page", ["Team Season Boxscore", "Shot Chart","Match report", "Four Factors", "Lebron"])

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

    elif page == "Match report":
        display_match_report()

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
