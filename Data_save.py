import sqlite3
import json
import requests

# Connect to SQLite database
conn = sqlite3.connect("C:\\Users\\dolez\\Desktop\\KP_Brno\\database.db")
cursor = conn.cursor()

# Ensure UTF-8 encoding
cursor.execute("PRAGMA encoding = 'UTF-8'")

# 1️⃣ Create All Tables
cursor.executescript('''
CREATE TABLE IF NOT EXISTS Games (
    game_id INTEGER PRIMARY KEY,
    clock TEXT,
    period INTEGER,
    period_length INTEGER,
    period_type TEXT,
    in_ot BOOLEAN
);

CREATE TABLE IF NOT EXISTS Teams (
    team_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER REFERENCES Games(game_id),
    tm INTEGER CHECK (tm IN (1, 2)),
    name TEXT,
    code TEXT,
    coach TEXT,
    assist_coach TEXT,
    score INTEGER,
    full_score INTEGER,
    total_minutes TEXT,
    field_goals_made INTEGER,
    field_goals_attempted INTEGER,
    field_goal_percentage REAL,
    three_pointers_made INTEGER,
    three_pointers_attempted INTEGER,
    three_point_percentage REAL,
    two_pointers_made INTEGER,
    two_pointers_attempted INTEGER,
    two_point_percentage REAL,
    free_throws_made INTEGER,
    free_throws_attempted INTEGER,
    free_throw_percentage REAL,
    rebounds_defensive INTEGER,
    rebounds_offensive INTEGER,
    rebounds_total INTEGER,
    assists INTEGER,
    turnovers INTEGER,
    steals INTEGER,
    blocks INTEGER,
    blocks_received INTEGER,
    fouls_personal INTEGER,
    fouls_on INTEGER,
    fouls_total INTEGER,
    points INTEGER,
    points_from_turnovers INTEGER,
    points_second_chance INTEGER,
    points_fast_break INTEGER,
    bench_points INTEGER,
    points_in_paint INTEGER,
    time_leading TEXT,
    biggest_scoring_run INTEGER,
    lead_changes INTEGER,
    times_scores_level INTEGER,
    tot_eff_1 REAL,
    tot_eff_2 REAL,
    tot_eff_3 REAL,
    tot_eff_4 REAL,
    tot_eff_5 REAL,
    tot_eff_6 REAL,
    tot_eff_7 REAL,
    p1_score INTEGER,
    p2_score INTEGER,
    p3_score INTEGER,
    p4_score INTEGER,
    fouls INTEGER,
    timeouts INTEGER
);

CREATE TABLE IF NOT EXISTS Players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER REFERENCES Games(game_id),
    team_id INTEGER REFERENCES Teams(team_id),
    json_player_id INTEGER,
    first_name TEXT,
    last_name TEXT,
    shirt_number TEXT,
    minutes_played TEXT,
    field_goals_made INTEGER,
    field_goals_attempted INTEGER,
    field_goal_percentage REAL,
    three_pointers_made INTEGER,
    three_pointers_attempted INTEGER,
    three_point_percentage REAL,
    two_pointers_made INTEGER,
    two_pointers_attempted INTEGER,
    two_point_percentage REAL,
    free_throws_made INTEGER,
    free_throws_attempted INTEGER,
    free_throw_percentage REAL,
    rebounds_defensive INTEGER,
    rebounds_offensive INTEGER,
    rebounds_total INTEGER,
    assists INTEGER,
    turnovers INTEGER,
    steals INTEGER,
    blocks INTEGER,
    blocks_received INTEGER,
    fouls_personal INTEGER,
    fouls_on INTEGER,
    points INTEGER,
    plus_minus INTEGER,
    efficiency_1 REAL,
    efficiency_2 REAL,
    efficiency_3 REAL,
    efficiency_4 REAL,
    efficiency_5 REAL,
    efficiency_6 REAL,
    efficiency_7 REAL,
    starter BOOLEAN
);

CREATE TABLE IF NOT EXISTS Shots (
    shot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER REFERENCES Games(game_id),
    team_id INTEGER REFERENCES Teams(team_id),
    player_name TEXT COLLATE NOCASE,
    player_shirt_number TEXT,
    json_player_id INTEGER,  -- Player ID from JSON
    period INTEGER,
    period_type TEXT,
    action_type TEXT,
    action_number INTEGER,
    previous_action TEXT,
    shot_sub_type TEXT,
    shot_result INTEGER CHECK (shot_result IN (0, 1)),  -- 0 = Missed, 1 = Made
    x_coord FLOAT,
    y_coord FLOAT
);

CREATE TABLE IF NOT EXISTS Scoring (
     scoring_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER REFERENCES Games(game_id),
    team_id INTEGER REFERENCES Teams(team_id),
    player_name TEXT COLLATE NOCASE,
    player_shirt_number TEXT,
    json_player_id INTEGER,  -- Player ID from JSON
    period INTEGER,
    period_type TEXT,
    game_time TEXT  -- Time of the score in the period
);

CREATE TABLE IF NOT EXISTS PlayByPlay (
   pbp_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER REFERENCES Games(game_id),
    team_id INTEGER REFERENCES Teams(team_id),
    player_id INTEGER REFERENCES Players(player_id),
    period INTEGER,
    period_type TEXT,
    game_time TEXT,
    clock TEXT,
    current_score_team1 INTEGER,
    current_score_team2 INTEGER,
    lead INTEGER,
    action_type TEXT,
    sub_type TEXT,
    action_number INTEGER,
    previous_action INTEGER,
    success INTEGER CHECK (success IN (0, 1)),
    scoring INTEGER CHECK (scoring IN (0, 1)),
    qualifiers TEXT
);

CREATE TABLE IF NOT EXISTS Officials (
    official_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER REFERENCES Games(game_id),
    role TEXT CHECK (role IN ('commissioner', 'referee1', 'referee2', 'referee3')),
    person_id INTEGER UNIQUE,
    external_id TEXT,
    first_name TEXT,
    last_name TEXT,
    scoreboard_name TEXT,
    nationality TEXT
);
''')
conn.commit()
print("✅ Database schema initialized.")

# Function to check if a game ID already exists
def game_exists(game_id):
    cursor.execute("SELECT COUNT(*) FROM Games WHERE game_id = ?", (game_id,))
    return cursor.fetchone()[0] > 0

def fetch_and_store_game_data(game_id):
    """Fetch and store ALL game data in one flow."""
    
    if game_exists(game_id):
        print(f"⚠️ Game {game_id} already exists in the database. Skipping upload.")
        return
    
    json_url = f"https://fibalivestats.dcd.shared.geniussports.com/data/{game_id}/data.json"
    response = requests.get(json_url)
    
    if response.status_code == 200:
        data = response.json()
        print(f"📌 Inserting Game ID: {game_id}")

        # Call functions to store all parts of the game data
        store_game_metadata(game_id, data)  
        store_team_data(game_id, data)  
        store_player_data(game_id, data)  
        store_shot_data(game_id, data)  
        store_scoring_data(game_id, data)  
        store_pbp_data(game_id, data)  
        store_officials_data(game_id, data)  

        print(f"✅ Game {game_id} data stored successfully!")
    else:
        print(f"❌ Failed to fetch data for Game {game_id}. Status code: {response.status_code}")


# 2️⃣ Insert Game Metadata
def store_game_metadata(game_id, data):

    # Extract game metadata
    clock = data.get("clock", "00:00")
    period = data.get("period", 4)
    period_length = data.get("periodLength", 10)
    period_type = data.get("periodType", "REGULAR")
    in_ot = data.get("inOT", 0)

    # Insert into database
    cursor.execute("""
    INSERT OR IGNORE INTO Games (game_id, clock, period, period_length, period_type, in_ot)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (game_id, clock, period, period_length, period_type, in_ot))

    conn.commit()

# 3️⃣ Insert Teams and Players
def store_team_data(game_id, data):
    

    teams = data.get("tm", {})

    for tm_id, team_info in teams.items():
        cursor.execute("""
        INSERT INTO Teams (
            game_id, tm, name, code, coach, assist_coach, score, full_score, total_minutes,
            field_goals_made, field_goals_attempted, field_goal_percentage,
            three_pointers_made, three_pointers_attempted, three_point_percentage,
            two_pointers_made, two_pointers_attempted, two_point_percentage,
            free_throws_made, free_throws_attempted, free_throw_percentage,
            rebounds_defensive, rebounds_offensive, rebounds_total,
            assists, turnovers, steals, blocks, blocks_received,
            fouls_personal, fouls_on, fouls_total, points, points_from_turnovers,
            points_second_chance, points_fast_break, bench_points, points_in_paint,
            time_leading, biggest_scoring_run, lead_changes, times_scores_level,tot_eff_1,tot_eff_2,tot_eff_3,tot_eff_4,tot_eff_5,tot_eff_6,tot_eff_7,p1_score,p2_score,p3_score,p4_score,fouls,timeouts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            game_id, int(tm_id),
            team_info.get("name", "Unknown"),
            team_info.get("code", "N/A"),
            team_info.get("coach", "N/A"),
            team_info.get("assistcoach1", "N/A"),
            team_info.get("score", 0),
            team_info.get("full_score", 0),
            team_info.get("tot_sMinutes", "00:00"),
            team_info.get("tot_sFieldGoalsMade", 0),
            team_info.get("tot_sFieldGoalsAttempted", 0),
            team_info.get("tot_sFieldGoalsPercentage", 0.0),
            team_info.get("tot_sThreePointersMade", 0),
            team_info.get("tot_sThreePointersAttempted", 0),
            team_info.get("tot_sThreePointersPercentage", 0.0),
            team_info.get("tot_sTwoPointersMade", 0),
            team_info.get("tot_sTwoPointersAttempted", 0),
            team_info.get("tot_sTwoPointersPercentage", 0.0),
            team_info.get("tot_sFreeThrowsMade", 0),
            team_info.get("tot_sFreeThrowsAttempted", 0),
            team_info.get("tot_sFreeThrowsPercentage", 0.0),
            team_info.get("tot_sReboundsDefensive", 0),
            team_info.get("tot_sReboundsOffensive", 0),
            team_info.get("tot_sReboundsTotal", 0),
            team_info.get("tot_sAssists", 0),
            team_info.get("tot_sTurnovers", 0),
            team_info.get("tot_sSteals", 0),
            team_info.get("tot_sBlocks", 0),
            team_info.get("tot_sBlocksReceived", 0),
            team_info.get("tot_sFoulsPersonal", 0),
            team_info.get("tot_sFoulsOn", 0),
            team_info.get("tot_sFoulsTotal", 0),
            team_info.get("tot_sPoints", 0),
            team_info.get("tot_sPointsFromTurnovers", 0),
            team_info.get("tot_sPointsSecondChance", 0),
            team_info.get("tot_sPointsFastBreak", 0),
            team_info.get("tot_sBenchPoints", 0),
            team_info.get("tot_sPointsInThePaint", 0),
            team_info.get("tot_sTimeLeading", "00:00"),
            team_info.get("tot_sBiggestScoringRun", 0),
            team_info.get("tot_sLeadChanges", 0),
            team_info.get("tot_sTimesScoresLevel", 0),
            team_info.get("tot_eff_1", 0.0),
            team_info.get("tot_eff_2", 0.0),
            team_info.get("tot_eff_3", 0.0),
            team_info.get("tot_eff_4", 0.0),
            team_info.get("tot_eff_5", 0.0),
            team_info.get("tot_eff_6", 0.0),
            team_info.get("tot_eff_7", 0.0),
            team_info.get("p1_score", 0),
            team_info.get("p2_score", 0),
            team_info.get("p3_score", 0),
            team_info.get("p4_score", 0),
            team_info.get("fouls", 0),
            team_info.get("timeouts", 0),
        ))

    conn.commit()
    
       
# 5 Insert shots
def store_player_data(game_id,data):
    

    teams = data.get("tm", {})

    for tm_id, team_info in teams.items():
        team_id = int(tm_id)  # Home = 1, Away = 2
        players = team_info.get("pl", {})

        for json_player_id, player in players.items():
            cursor.execute("""
            INSERT INTO Players (
                game_id, team_id, json_player_id, first_name, last_name, shirt_number, minutes_played,
                field_goals_made, field_goals_attempted, field_goal_percentage,
                three_pointers_made, three_pointers_attempted, three_point_percentage,
                two_pointers_made, two_pointers_attempted, two_point_percentage,
                free_throws_made, free_throws_attempted, free_throw_percentage,
                rebounds_defensive, rebounds_offensive, rebounds_total,
                assists, turnovers, steals, blocks, blocks_received,
                fouls_personal, fouls_on, points, plus_minus,
                efficiency_1, efficiency_2, efficiency_3, efficiency_4, efficiency_5,
                efficiency_6, efficiency_7, starter
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?,?)
            """, (
                game_id, team_id, int(json_player_id),
                player.get("firstName", "Unknown"),
                player.get("familyName", "Unknown"),
                player.get("shirtNumber", ""),
                player.get("sMinutes", "00:00"),
                player.get("sFieldGoalsMade", 0),
                player.get("sFieldGoalsAttempted", 0),
                player.get("sFieldGoalsPercentage", 0.0),
                player.get("sThreePointersMade", 0),
                player.get("sThreePointersAttempted", 0),
                player.get("sThreePointersPercentage", 0.0),
                player.get("sTwoPointersMade", 0),
                player.get("sTwoPointersAttempted", 0),
                player.get("sTwoPointersPercentage", 0.0),
                player.get("sFreeThrowsMade", 0),
                player.get("sFreeThrowsAttempted", 0),
                player.get("sFreeThrowsPercentage", 0.0),
                player.get("sReboundsDefensive", 0),
                player.get("sReboundsOffensive", 0),
                player.get("sReboundsTotal", 0),
                player.get("sAssists", 0),
                player.get("sTurnovers", 0),
                player.get("sSteals", 0),
                player.get("sBlocks", 0),
                player.get("sBlocksReceived", 0),
                player.get("sFoulsPersonal", 0),
                player.get("sFoulsOn", 0),
                player.get("sPoints", 0),
                player.get("plusMinusPoints", 0),
                player.get("eff_1", 0.0),
                player.get("eff_2", 0.0),
                player.get("eff_3", 0.0),
                player.get("eff_4", 0.0),
                player.get("eff_5", 0.0),
                player.get("eff_6", 0.0),
                player.get("eff_7", 0.0),
                bool(player.get("starter", 0))
            ))

    conn.commit()

# 6 Insert Shots
def store_shot_data(game_id, data):

    teams = data.get("tm", {})

    for tm_id, team_info in teams.items():
        team_id = int(tm_id)  # Home = 1, Away = 2
        shots = team_info.get("shot", [])

        for shot in shots:
            cursor.execute("""
            INSERT INTO Shots (
                game_id, team_id, player_name, player_shirt_number, json_player_id,
                period, period_type, action_type, action_number,
                previous_action, shot_sub_type, shot_result, x_coord, y_coord
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id, team_id,
                shot.get("player", "Unknown"),
                shot.get("shirtNumber", ""),
                shot.get("p", None),
                shot.get("per", 0),
                shot.get("perType", "REGULAR"),
                shot.get("actionType", "Unknown"),
                shot.get("actionNumber", 0),
                shot.get("previousAction", ""),
                shot.get("subType", "Unknown"),
                shot.get("r", 0),
                shot.get("x", 0.0),
                shot.get("y", 0.0)
            ))

    conn.commit()

# 7 Insert Scoring
def store_scoring_data(game_id, data):

    teams = data.get("tm", {})

    for tm_id, team_info in teams.items():
        team_id = int(tm_id)  # Home = 1, Away = 2
        scoring_events = team_info.get("scoring", [])

        for score in scoring_events:
            cursor.execute("""
            INSERT INTO Scoring (
                game_id, team_id, player_name, player_shirt_number, json_player_id,
                period, period_type, game_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id, team_id,
                score.get("player", "Unknown"),
                score.get("shirtNumber", ""),
                score.get("pno", None),
                score.get("per", 0),
                score.get("perType", "REGULAR"),
                score.get("gt", "00:00")
            ))

    conn.commit()
# 8 Insert Play-by-Play
def store_pbp_data(game_id, data):

    pbp_events = data.get("pbp", [])

    for event in pbp_events:
        cursor.execute("""
        INSERT INTO PlayByPlay (
            game_id, team_id, player_id, period, period_type, game_time,
            clock, current_score_team1, current_score_team2, lead, action_type,
            sub_type, action_number, previous_action, success, scoring, qualifiers
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id, event.get("tno", 0), 
            event.get("pno", None),  # Player ID from JSON
            event.get("period", 0),
            event.get("periodType", "REGULAR"),
            event.get("gt", "00:00"),
            event.get("clock", "00:00:00"),
            event.get("s1", 0),
            event.get("s2", 0),
            event.get("lead", 0),
            event.get("actionType", "Unknown"),
            event.get("subType", ""),
            event.get("actionNumber", 0),
            event.get("previousAction", None),
            event.get("success", 0),
            event.get("scoring", 0),
            ", ".join(event.get("qualifier", []))  # Store as comma-separated text
        ))

    conn.commit()
 # 9 Off
def store_officials_data(game_id, data):

    officials = data.get("officials", {})

    for role, details in officials.items():
        # Skip empty referee slots
        if not details.get("firstName") and not details.get("familyName"):
            continue

        cursor.execute("""
        INSERT OR IGNORE INTO Officials (
            game_id, role, person_id, external_id, first_name, last_name,
            scoreboard_name, nationality
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            game_id, role,
            details.get("personId"),
            details.get("externalId"),
            details.get("firstName", ""),
            details.get("familyName", ""),
            details.get("scoreboardName", ""),
            details.get("nationality", "Unknown")
        ))

    conn.commit()

def run_game_input_loop():
    """Interactive loop for entering multiple game IDs."""
    while True:
        game_ids_input = input("Enter game IDs (comma-separated) or type 'exit' to quit: ").strip()
        if game_ids_input.lower() == 'exit':
            print("🏀 Exiting program.")
            break
        
        game_ids = [game_id.strip() for game_id in game_ids_input.split(",")]
        
        for game_id in game_ids:
            if game_id.isdigit():
                fetch_and_store_game_data(int(game_id))
            else:
                print(f"⚠️ Invalid game ID: {game_id}. Skipping...")

# Start the input loop
run_game_input_loop()

conn.close()

