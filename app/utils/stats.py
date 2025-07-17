def get_team_stats(row, team, is_home):
    if is_home:
        return row['FTHG'], row['FTAG'], 3 if row['FTR'] == 'H' else 1 if row['FTR'] == 'D' else 0
    else:
        return row['FTAG'], row['FTHG'], 3 if row['FTR'] == 'A' else 1 if row['FTR'] == 'D' else 0

def get_rolling_averages(matches, team, date, window=5):
    team_matches = matches[
        ((matches['HomeTeam'] == team) | (matches['AwayTeam'] == team)) &
        (matches['Date'] < date)
    ].sort_values(by='Date', ascending=False).head(window)

    if team_matches.empty:
        return 0, 0, 0

    gs_list, gc_list, pts_list = [], [], []

    for _, row in team_matches.iterrows():
        is_home = row['HomeTeam'] == team
        gs, gc, pts = get_team_stats(row, team, is_home)
        gs_list.append(gs)
        gc_list.append(gc)
        pts_list.append(pts)

    return sum(gs_list)/len(gs_list), sum(gc_list)/len(gc_list), sum(pts_list)/len(pts_list)
