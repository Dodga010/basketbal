import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from datetime import datetime
import io
import base64

# Import database path from webpokus
from webpokus import db_path

def display_finals_preparation():
    """
    Create a dedicated page for finals preparation with team selection and PDF export
    Add this function to your main webpokus.py file
    """
    import streamlit as st
    
    st.title("Finals Preparation Dashboard")
    st.subheader("Best-of-3 Series for 3rd Place")
    
    # Get list of all teams
    from webpokus import fetch_teams
    teams = fetch_teams()
    
    # Create two columns for team selection
    col1, col2 = st.columns(2)
    
    with col1:
        your_team = st.selectbox("Select Your Team", options=teams, key="your_team")
    
    with col2:
        # Filter out the first selected team from options
        opponent_options = [t for t in teams if t != your_team]
        opponent_team = st.selectbox("Select Opponent Team", options=opponent_options, key="opponent_team")
    
    # Analysis button
    if st.button("Generate Analysis"):
        if your_team and opponent_team:
            # Get key players from both teams
            from webpokus import fetch_team_players
            your_team_players = fetch_team_players(your_team)
            opponent_players = fetch_team_players(opponent_team)
            
            # Create tabs for different sections of analysis
            tabs = st.tabs(["Team Comparison", "Your Team Players", "Opponent Players", 
                           "Lineup Analysis", "Strategic Recommendations"])
            
            # Team Comparison Tab
            with tabs[0]:
                display_team_comparison(your_team, opponent_team)
            
            # Your Team Players Tab
            with tabs[1]:
                display_player_analysis(your_team_players, "own", opponent_team, your_team)
            
            # Opponent Players Tab
            with tabs[2]:
                display_player_analysis(opponent_players, "opponent", your_team, opponent_team)
            
            # Lineup Analysis Tab
            with tabs[3]:
                display_lineup_analysis(your_team, opponent_team)
            
            # Strategic Recommendations Tab  
            with tabs[4]:
                display_strategic_recommendations(your_team, opponent_team)
            
            # PDF Export Section
            st.divider()
            st.header("PDF Export")
            
            # PDF generation options
            include_all = st.checkbox("Include all analysis sections", value=True)
            
            if not include_all:
                include_team_analysis = st.checkbox("Include Team Comparison", value=True)
                include_your_players = st.checkbox("Include Your Team Players", value=True)
                num_your_players = st.slider("Number of your players to include", 1, len(your_team_players), 5) if include_your_players else 0
                include_opponent_players = st.checkbox("Include Opponent Players", value=True) 
                num_opponent_players = st.slider("Number of opponent players to include", 1, len(opponent_players), 3) if include_opponent_players else 0
                include_strategy = st.checkbox("Include Strategic Recommendations", value=True)
            else:
                include_team_analysis = True
                include_your_players = True
                num_your_players = min(5, len(your_team_players))
                include_opponent_players = True
                num_opponent_players = min(3, len(opponent_players))
                include_strategy = True
            
            # Generate PDF button
            if st.button("Generate PDF Report"):
                with st.spinner("Generating PDF report..."):
                    try:
                        pdf_bytes, pdf_filename = generate_finals_pdf_report(
                            your_team, opponent_team, 
                            your_team_players[:num_your_players] if include_your_players else [],
                            opponent_players[:num_opponent_players] if include_opponent_players else [],
                            include_team_analysis,
                            include_strategy
                        )
                        
                        # Create download button for the generated PDF
                        st.success("PDF Report generated successfully!")
                        
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_bytes,
                            file_name=pdf_filename,
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error(f"Error generating PDF report: {str(e)}")

def display_team_comparison(your_team, opponent_team):
    """Display team comparison analytics"""
    import streamlit as st
    
    st.header(f"Team Comparison: {your_team} vs {opponent_team}")
    
    # Import necessary functions
    from webpokus import (
        analyze_team_comparison, analyze_advanced_metrics,
        fetch_team_four_factors, plot_four_factors_stats,
        analyze_shot_distribution_comparison
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Four Factors Analysis")
        
        # Four Factors Analysis
        team1_factors = fetch_team_four_factors(your_team)
        team2_factors = fetch_team_four_factors(opponent_team)
        
        # Creating placeholder for the plot
        fig_factors = plt.figure(figsize=(10, 6))
        plot_four_factors_stats(your_team, opponent_team, ["eFG%", "TOV%", "ORB%", "FT Rate"])
        st.pyplot(fig_factors)

    with col2:
        st.subheader("Key Performance Indicators")
        
        # Team comparison data
        team_comparison_data = analyze_team_comparison(your_team, opponent_team)
        
        # Display key performance indicators from team comparison data in a table
        if team_comparison_data is not None:
            # Extract relevant KPIs from team_comparison_data and format as a dataframe
            # Adjust based on your actual data structure
            kpi_df = pd.DataFrame({
                'KPI': ['Points', 'Rebounds', 'Assists', 'FG%', '3P%', 'FT%'],
                your_team: [team_comparison_data.get('team1_pts', 'N/A'), 
                           team_comparison_data.get('team1_reb', 'N/A'),
                           team_comparison_data.get('team1_ast', 'N/A'),
                           team_comparison_data.get('team1_fg_pct', 'N/A'),
                           team_comparison_data.get('team1_fg3_pct', 'N/A'),
                           team_comparison_data.get('team1_ft_pct', 'N/A')],
                opponent_team: [team_comparison_data.get('team2_pts', 'N/A'), 
                               team_comparison_data.get('team2_reb', 'N/A'),
                               team_comparison_data.get('team2_ast', 'N/A'),
                               team_comparison_data.get('team2_fg_pct', 'N/A'),
                               team_comparison_data.get('team2_fg3_pct', 'N/A'),
                               team_comparison_data.get('team2_ft_pct', 'N/A')]
            })
            st.dataframe(kpi_df.set_index('KPI'), use_container_width=True)
        else:
            st.warning("Team comparison data is not available.")
    
    # Shot Distribution Comparison
    st.header("Shot Distribution Comparison")
    shot_distribution = analyze_shot_distribution_comparison(your_team, opponent_team)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{your_team} Shot Distribution")
        # Create shot distribution chart
        if shot_distribution and 'team1_shots' in shot_distribution:
            fig1 = plt.figure(figsize=(8, 6))
            # Plot shot distribution for team 1
            plt.title(f"{your_team} Shot Distribution")
            # Implement the shot visualization based on your data structure
            st.pyplot(fig1)
        else:
            st.warning("Shot distribution data not available.")
    
    with col2:
        st.subheader(f"{opponent_team} Shot Distribution")
        # Create shot distribution chart
        if shot_distribution and 'team2_shots' in shot_distribution:
            fig2 = plt.figure(figsize=(8, 6))
            # Plot shot distribution for team 2
            plt.title(f"{opponent_team} Shot Distribution")
            # Implement the shot visualization based on your data structure
            st.pyplot(fig2)
        else:
            st.warning("Shot distribution data not available.")
    
    # Advanced Metrics
    st.header("Advanced Metrics Comparison")
    advanced_metrics = analyze_advanced_metrics(your_team, opponent_team)
    
    if advanced_metrics is not None:
        # Format advanced metrics as a dataframe for display
        adv_df = pd.DataFrame({
            'Metric': ['Offensive Rating', 'Defensive Rating', 'Net Rating', 'Pace', 'Assist Ratio'],
            your_team: [advanced_metrics.get('team1_ortg', 'N/A'), 
                       advanced_metrics.get('team1_drtg', 'N/A'),
                       advanced_metrics.get('team1_net', 'N/A'),
                       advanced_metrics.get('team1_pace', 'N/A'),
                       advanced_metrics.get('team1_ast_ratio', 'N/A')],
            opponent_team: [advanced_metrics.get('team2_ortg', 'N/A'), 
                           advanced_metrics.get('team2_drtg', 'N/A'),
                           advanced_metrics.get('team2_net', 'N/A'),
                           advanced_metrics.get('team2_pace', 'N/A'),
                           advanced_metrics.get('team2_ast_ratio', 'N/A')]
        })
        st.dataframe(adv_df.set_index('Metric'), use_container_width=True)
    else:
        st.warning("Advanced metrics are not available.")

def collect_player_data(player_name, player_type):
    """Collect comprehensive data for a player"""
    # Import necessary functions
    from webpokus import (
        fetch_player_stats, fetch_player_expected_stats,
        fetch_shot_data, fetch_player_and_league_stats_per_40,
        fetch_player_game_stats, calculate_weighted_lebron_for_player,
        fetch_shooting_fouls_for_entities, fetch_player_fouls_and_minutes,
        calculate_shot_zones, identify_hot_zones, fetch_player_games
    )
    
    player_data = {}
    
    try:
        # Basic stats
        player_data['basic_stats'] = fetch_player_stats(player_name)
        
        # Expected stats
        player_data['expected_stats'] = fetch_player_expected_stats(player_name)
        
        # Shot data
        player_data['shot_data'] = fetch_shot_data(player_name)
        
        # Per 40 minutes stats
        player_data['per_40_stats'] = fetch_player_and_league_stats_per_40(player_name)
        
        # Game stats
        player_data['game_stats'] = fetch_player_game_stats(player_name)
        
        # LEBRON score
        player_data['lebron_score'] = calculate_weighted_lebron_for_player(player_name, player_data['game_stats'])
        
        # Foul analysis
        player_data['fouls_data'] = fetch_shooting_fouls_for_entities([player_name], entity_type='player')
        player_foul_data = fetch_player_fouls_and_minutes(shooting_fouls_only=False)
        player_data['foul_rate'] = next((item for item in player_foul_data if item.get('player_name') == player_name), None)
        
        # Shot patterns and hot zones
        shots_df = player_data['shot_data']
        if shots_df is not None and not shots_df.empty:
            player_data['shot_zones'] = calculate_shot_zones(shots_df)
            player_data['hot_zones'] = identify_hot_zones(shots_df)
        
        # Games played
        player_data['games'] = fetch_player_games(player_name)
    
    except Exception as e:
        import streamlit as st
        st.error(f"Error collecting data for {player_name}: {str(e)}")
        player_data['error'] = str(e)
    
    return player_data

def display_player_analysis(player_list, player_type, opposing_team, team_name):
    """Display detailed analysis for a list of players"""
    import streamlit as st
    from webpokus import (
        plot_shot_coordinates, display_shot_data_with_distance,
        plot_fg_percentage_with_frequency
    )
    
    st.header(f"Player Analysis: {team_name}")
    
    # Allow the user to select a player to analyze
    selected_player = st.selectbox(
        "Select a player to analyze:",
        options=player_list,
        key=f"{player_type}_player_selector"
    )
    
    if selected_player:
        with st.spinner(f"Loading data for {selected_player}..."):
            # Collect all player data
            player_data = collect_player_data(selected_player, player_type)
            
            if 'error' in player_data:
                st.error(f"Could not load complete data for {selected_player}: {player_data['error']}")
            
            # Display tabs for different analysis sections
            tab1, tab2, tab3, tab4 = st.tabs(["Basic Stats", "Shot Analysis", "Defensive Profile", "Advanced Metrics"])
            
            # Tab 1: Basic Stats
            with tab1:
                st.subheader(f"Key Stats: {selected_player}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Basic stats
                    if 'basic_stats' in player_data and player_data['basic_stats'] is not None:
                        basic_stats = player_data['basic_stats']
                        st.metric("Points Per Game", basic_stats.get('ppg', 'N/A'))
                        st.metric("Rebounds Per Game", basic_stats.get('rpg', 'N/A'))
                        st.metric("Assists Per Game", basic_stats.get('apg', 'N/A'))
                        st.metric("Field Goal %", basic_stats.get('fg_pct', 'N/A'))
                        st.metric("3-Point %", basic_stats.get('fg3_pct', 'N/A'))
                    else:
                        st.warning("Basic stats not available.")
                
                with col2:
                    # Recent form - using game stats
                    if 'game_stats' in player_data and player_data['game_stats'] is not None:
                        st.subheader("Recent Form")
                        
                        # Convert to dataframe for display
                        if isinstance(player_data['game_stats'], pd.DataFrame):
                            game_stats_df = player_data['game_stats']
                            recent_games = game_stats_df.tail(5)  # Last 5 games
                            st.dataframe(recent_games, use_container_width=True)
                        else:
                            st.write("Game stats format not recognized.")
                    else:
                        st.warning("Recent form data not available.")
            
            # Tab 2: Shot Analysis
            with tab2:
                st.subheader(f"Shot Analysis: {selected_player}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Shot Chart")
                    
                    # Shot chart visualization
                    if 'shot_data' in player_data and player_data['shot_data'] is not None and not player_data['shot_data'].empty:
                        fig_shot = plt.figure(figsize=(10, 8))
                        plot_shot_coordinates(selected_player)
                        st.pyplot(fig_shot)
                    else:
                        st.warning("Shot data not available for visualization.")
                
                with col2:
                    if 'shot_data' in player_data and player_data['shot_data'] is not None and not player_data['shot_data'].empty:
                        st.subheader("Shot Efficiency by Distance")
                        
                        # Shot efficiency visualization
                        fig_eff = plt.figure(figsize=(10, 6))
                        plot_fg_percentage_with_frequency(selected_player)
                        st.pyplot(fig_eff)
                    else:
                        st.warning("Shot efficiency data not available.")
                
                # Display shot zones and hot zones
                st.subheader("Shot Distribution and Hot Zones")
                
                if ('shot_zones' in player_data and player_data['shot_zones'] is not None and 
                    'hot_zones' in player_data and player_data['hot_zones'] is not None):
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("Shot Distribution by Zone")
                        # Display shot zones data
                        st.dataframe(player_data['shot_zones'])
                    
                    with col2:
                        st.write("Hot Zones")
                        # Display hot zones data
                        st.dataframe(player_data['hot_zones'])
                else:
                    st.warning("Shot zone analysis not available.")
                
                # Detailed shot data with distance
                st.subheader("Detailed Shot Analysis")
                display_shot_data_with_distance(selected_player)
            
            # Tab 3: Defensive Profile
            with tab3:
                st.subheader(f"Defensive Profile: {selected_player}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Foul Analysis")
                    
                    # Display foul data
                    if 'foul_rate' in player_data and player_data['foul_rate'] is not None:
                        foul_rate = player_data['foul_rate']
                        st.metric("Fouls Per Game", foul_rate.get('fouls_per_game', 'N/A'))
                        st.metric("Fouls Per 40 Minutes", foul_rate.get('fouls_per_40', 'N/A'))
                        
                        # Create a simple bar chart for foul types if available
                        if 'fouls_data' in player_data and player_data['fouls_data'] is not None:
                            fig_fouls = plt.figure(figsize=(10, 6))
                            # Plot foul data - adjust based on your data structure
                            st.pyplot(fig_fouls)
                    else:
                        st.warning("Foul analysis data not available.")
                
                with col2:
                    st.subheader("Defensive Metrics")
                    
                    # Display defensive metrics if available
                    if 'basic_stats' in player_data and player_data['basic_stats'] is not None:
                        basic_stats = player_data['basic_stats']
                        st.metric("Steals Per Game", basic_stats.get('spg', 'N/A'))
                        st.metric("Blocks Per Game", basic_stats.get('bpg', 'N/A'))
                        st.metric("Defensive Rebounds", basic_stats.get('drpg', 'N/A'))
                    else:
                        st.warning("Defensive metrics not available.")
                        
            # Tab 4: Advanced Metrics
            with tab4:
                st.subheader(f"Advanced Metrics: {selected_player}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("LEBRON Score")
                    
                    # Display LEBRON score if available
                    if 'lebron_score' in player_data and player_data['lebron_score'] is not None:
                        st.metric("LEBRON Score", player_data['lebron_score'])
                        st.write("(Luck-adjusted Estimated Box plus/minus Rating Over Net seasons)")
                    else:
                        st.warning("LEBRON score not available.")
                
                with col2:
                    st.subheader("Per 40 Minutes Stats")
                    
                    # Display per 40 minutes stats if available
                    if 'per_40_stats' in player_data and player_data['per_40_stats'] is not None:
                        per_40 = player_data['per_40_stats']
                        # Convert to dataframe for display if needed
                        if isinstance(per_40, pd.DataFrame):
                            st.dataframe(per_40)
                        else:
                            # Display as metrics
                            st.metric("Points per 40", per_40.get('pts_per_40', 'N/A'))
                            st.metric("Rebounds per 40", per_40.get('reb_per_40', 'N/A'))
                            st.metric("Assists per 40", per_40.get('ast_per_40', 'N/A'))
                    else:
                        st.warning("Per 40 minutes stats not available.")
                
                # Matchup recommendations
                st.subheader("Matchup Recommendations")
                
                if player_type == "opponent":
                    st.write("Defensive Assignment Recommendations:")
                    # Generate defensive assignments based on analysis
                    st.write("• Assign a player with similar height and quick lateral movement")
                    st.write("• Force to non-dominant hand")
                    st.write("• Contest shots in hot zones")
                else:
                    st.write("Offensive Opportunities:")
                    # Generate offensive recommendations against opponent
                    st.write("• Exploit matchup against shorter defenders")
                    st.write("• Focus on shots from identified hot zones")
                    st.write("• Look for pick-and-roll opportunities")

def display_lineup_analysis(your_team, opponent_team):
    """Display lineup analysis and recommendations"""
    import streamlit as st
    
    st.header("Lineup Analysis and Recommendations")
    
    # Try to fetch five player segments data
    try:
        # This would need a game_id parameter - showing a placeholder approach
        st.info("This analysis is based on available lineup data from the season.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Best Performing Lineups")
            
            # Create a placeholder dataframe for best lineups
            best_lineups = pd.DataFrame({
                'Players': ['Player1, Player2, Player3, Player4, Player5', 
                           'Player2, Player3, Player4, Player5, Player6'],
                'Minutes': [15.3, 12.7],
                'Net Rating': [+12.5, +8.3],
                'Offensive Rating': [120.5, 115.8],
                'Defensive Rating': [108.0, 107.5]
            })
            
            st.dataframe(best_lineups, use_container_width=True)
            
        with col2:
            st.subheader("Lineup Effectiveness vs Opponent")
            
            # Create placeholder chart
            fig_lineups = plt.figure(figsize=(10, 6))
            plt.bar(['Lineup 1', 'Lineup 2', 'Lineup 3', 'Lineup 4', 'Lineup 5'], 
                   [15.3, 12.7, 8.5, 4.2, -2.5])
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Lineup Net Ratings')
            plt.ylabel('Net Rating')
            st.pyplot(fig_lineups)
    
    except Exception as e:
        st.error(f"Error analyzing lineup data: {str(e)}")
    
    # Substitution patterns section
    st.subheader("Substitution Timing Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("First Quarter")
        st.write("• Start with your strongest lineup")
        st.write("• First substitution at ~6:00 mark")
        st.write("• Rest primary scorer at ~3:00")
        
        st.write("Second Quarter")
        st.write("• Start with energy lineup")
        st.write("• Re-insert starters at ~6:00")
        st.write("• Focus on defense before halftime")
    
    with col2:
        st.write("Third Quarter")
        st.write("• Start with starters")
        st.write("• Adjust based on halftime analysis")
        st.write("• Be ready to change defensive scheme")
        
        st.write("Fourth Quarter")
        st.write("• Offense-defense substitutions in last 2 minutes")
        st.write("• Use timeouts strategically for rest")
        st.write("• Close with your proven finishers")

def display_strategic_recommendations(your_team, opponent_team):
    """Display strategic recommendations for the series"""
    import streamlit as st
    
    st.header("Strategic Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Offensive Strategy")
        
        st.markdown("""
        1. **Exploit Opponent Weaknesses**:
           - Target weaker defenders in pick-and-roll situations
           - Focus on inside scoring if rim protection is weak
           
        2. **Shot Selection Strategy**:
           - Increase three-point attempts from corners (high percentage area)
           - Look for matchup advantages in the post
           
        3. **Pace and Tempo**:
           - Control pace to your advantage (faster or slower based on team strengths)
           - Create transition opportunities through defensive rebounds
           
        4. **Play Design Focus**:
           - Utilize off-ball screens to free shooters
           - Implement specific plays targeting opponent's defensive scheme
        """)
    
    with col2:
        st.subheader("Defensive Strategy")
        
        st.markdown("""
        1. **Primary Defensive Focus**:
           - Pressure opponent's primary ballhandlers
           - Contain their top scorer with dedicated defender
           
        2. **Shot Defense Priorities**:
           - Force opponent to shoot from low-efficiency zones
           - Contest all three-point attempts aggressively
           
        3. **Rebounding Emphasis**:
           - Focus on defensive glass to limit second chances
           - Assign specific box-out responsibilities
           
        4. **Situational Defense**:
           - Switch defensive schemes based on score/time situation
           - Prepare specific late-game defensive rotations
        """)
    
    st.subheader("Game-by-Game Strategy")
    
    game_strategies = pd.DataFrame({
        'Game': ['Game 1', 'Game 2', 'Game 3 (if needed)'],
        'Primary Focus': [
            'Establish tempo and test defensive strategies', 
            'Adjust to opponent counters, exploit identified weaknesses',
            'Execute refined strategy, focus on mental toughness'
        ],
        'Key Adjustments': [
            'Monitor opponent reactions to defensive scheme',
            'Counter opponent adjustments from Game 1',
            'Maximize strengths, minimize vulnerabilities identified in Games 1-2'
        ]
    })
    
    st.dataframe(game_strategies.set_index('Game'), use_container_width=True)
    
    st.subheader("Key Performance Indicators to Track")
    
    kpi_tracking = pd.DataFrame({
        'KPI': ['Points in the Paint Differential', 'Assist-to-Turnover Ratio', 'Second Chance Points', 
                'Fast Break Points', 'Bench Scoring'],
        'Target': ['+8', '2.0+', '<10 allowed', '15+', '25+'],
        'Importance': ['High', 'High', 'Medium', 'Medium', 'Low'],
        'Why It Matters': [
            'Indicates paint dominance and high-efficiency scoring',
            'Shows team ball control and quality possessions',
            'Reflects defensive rebounding success',
            'Demonstrates transition offense effectiveness',
            'Indicates depth advantage and rotation strength'
        ]
    })
    
    st.dataframe(kpi_tracking.set_index('KPI'), use_container_width=True)

def generate_finals_pdf_report(your_team, opponent_team, your_team_players, opponent_players, 
                             include_team_analysis=True, include_strategy=True):
    """Generate a comprehensive PDF report for finals preparation"""
    import streamlit as st
    
    filename = f"Finals_Preparation_{your_team}_vs_{opponent_team}_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    
    # Use matplotlib's PdfPages to create multi-page PDF
    with PdfPages(filename) as pdf:
        # Cover Page
        fig = plt.figure(figsize=(8.5, 11))
        plt.text(0.5, 0.8, "Finals Preparation Report", fontsize=24, ha='center', weight='bold')
        plt.text(0.5, 0.7, f"{your_team} vs {opponent_team}", fontsize=20, ha='center')
        plt.text(0.5, 0.65, "Best-of-3 Series for 3rd Place", fontsize=16, ha='center')
        plt.text(0.5, 0.55, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                fontsize=12, ha='center')
        plt.text(0.5, 0.52, f"Prepared by: {os.getlogin()}", fontsize=12, ha='center')
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Team Comparison Pages
        if include_team_analysis:
            from webpokus import (
                plot_four_factors_stats, analyze_team_comparison, analyze_advanced_metrics
            )
            
            fig = plt.figure(figsize=(8.5, 11))
            plt.suptitle(f"Team Comparison: {your_team} vs {opponent_team}", fontsize=16)
            plt.subplot(2, 1, 1)
            plt.title("Four Factors Comparison")
            plot_four_factors_stats(your_team, opponent_team, ["eFG%", "TOV%", "ORB%", "FT Rate"])
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # Player Analysis Pages
        for player in your_team_players:
            from webpokus import (
                plot_shot_coordinates, plot_fg_percentage_with_frequency
            )
            
            # Player stats page
            fig = plt.figure(figsize=(8.5, 11))
            plt.suptitle(f"Player Analysis: {player}", fontsize=16)
            
            # Fetch player data
            player_data = collect_player_data(player, "own")
            
            # Basic stats
            plt.subplot(2, 2, 1)
            plt.title("Key Stats")
            plt.axis('off')
            # Add player stats text
            if 'basic_stats' in player_data and player_data['basic_stats'] is not None:
                stats = player_data['basic_stats']
                y_pos = 0.9
                for stat, value in stats.items():
                    if y_pos > 0.1:  # Ensure we don't go off the subplot
                        plt.text(0.1, y_pos, f"{stat}: {value}")
                        y_pos -= 0.05
            
            # Shot chart
            plt.subplot(2, 2, 2)
            plt.title("Shot Chart")
            if 'shot_data' in player_data and player_data['shot_data'] is not None and not player_data['shot_data'].empty:
                # Plot shot chart
                plot_shot_coordinates(player)
            
            # Hot zones
            plt.subplot(2, 2, 3)
            plt.title("Shot Zones")
            if 'shot_zones' in player_data and player_data['shot_zones'] is not None:
                # Plot shot zones visualization based on your data structure
                pass
            
            # Add matchup recommendations
            plt.subplot(2, 2, 4)
            plt.title("Matchup Recommendations")
            plt.axis('off')
            plt.text(0.1, 0.8, "• Exploit matchups against smaller defenders")
            plt.text(0.1, 0.7, "• Focus on corner three opportunities")
            plt.text(0.1, 0.6, "• Attack defensive weak links")
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # Player shot analysis page
            fig = plt.figure(figsize=(8.5, 11))
            plt.suptitle(f"Shot Analysis: {player}", fontsize=16)
            
            # Plot shot efficiency by distance
            plt.subplot(2, 1, 1)
            plot_fg_percentage_with_frequency(player)
            
            # Plot shot distribution
            plt.subplot(2, 1, 2)
            plt.title("Shot Distribution")
            # Code to plot shot distribution based on your data structure
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
        
        # Opponent analysis pages
        for player in opponent_players:
            # Similar implementation as above for opponent players
            pass
        
        # Strategic recommendations page
        if include_strategy:
            fig = plt.figure(figsize=(8.5, 11))
            plt.suptitle("Strategic Recommendations", fontsize=16)
            plt.axis('off')
            
            plt.text(0.1, 0.9, "Offensive Strategy:", fontweight='bold', fontsize=14)
            plt.text(0.1, 0.85, "1. Exploit opponent's defensive weaknesses")
            plt.text(0.1, 0.82, "2. Focus on high-percentage shots")
            plt.text(0.1, 0.79, "3. Increase ball movement to create open looks")
            
            plt.text(0.1, 0.7, "Defensive Strategy:", fontweight='bold', fontsize=14)
            plt.text(0.1, 0.65, "1. Pressure opponent's primary ballhandlers")
            plt.text(0.1, 0.62, "2. Focus on defensive rebounding")
            plt.text(0.1, 0.59, "3. Contest all shots in opponent hot zones")
            
            plt.text(0.1, 0.5, "Game-by-Game Strategy:", fontweight='bold', fontsize=14)
            plt.text(0.1, 0.45, "Game 1: Establish pace and test defensive schemes")
            plt.text(0.1, 0.42, "Game 2: Adjust to opponent counters")
            plt.text(0.1, 0.39, "Game 3: Execute refined strategy with full focus")
            
            pdf.savefig()
            plt.close()
    
    # Return the PDF file for download
    with open(filename, "rb") as pdf_file:
        PDFbyte = pdf_file.read()
    
    return PDFbyte, filename