#!/usr/bin/env python3

def _generate_player_explanation(player_data):
    """Generate natural language explanation from player analytics."""
    rows = player_data.get("rows", [])

    if not rows:
        return "Insufficient data to generate player explanation."

    # Analyze stable vs emerging skills
    stable_skills = []
    emerging_skills = []

    for row in rows:
        skill_name = row.get("label", f"Dimension {row['dim_idx']}")
        mean = row["forecast_mean"]
        is_stable = row.get("is_stable", False)
        seasons = row.get("stable_seasons", 0)

        skill_info = {
            "name": skill_name,
            "strength": mean,
            "seasons": seasons
        }

        if is_stable:
            stable_skills.append(skill_info)
        else:
            emerging_skills.append(skill_info)

    # Sort by strength
    stable_skills.sort(key=lambda x: x["strength"], reverse=True)
    emerging_skills.sort(key=lambda x: abs(x["strength"]), reverse=True)

    # Build explanation
    explanation_parts = []

    # Overall assessment
    total_skills = len(stable_skills) + len(emerging_skills)
    stable_count = len(stable_skills)
    emerging_count = len(emerging_skills)

    explanation_parts.append(f"This player demonstrates {stable_count} stable skills (consistent across seasons) and {emerging_count} emerging skills (developing or inconsistent).")

    # Top strengths
    if stable_skills:
        top_stable = [s for s in stable_skills if s["strength"] > 0.1][:3]
        if top_stable:
            strength_names = [s["name"] for s in top_stable]
            explanation_parts.append(f"Their most consistent strengths are in {', '.join(strength_names[:2])}{' and ' + strength_names[2] if len(strength_names) > 2 else ''}.")

    # Emerging skills analysis
    if emerging_skills:
        strong_emerging = [s for s in emerging_skills if s["strength"] > 0.2][:2]
        weak_emerging = [s for s in emerging_skills if s["strength"] < -0.2][:2]

        if strong_emerging:
            emerging_names = [s["name"] for s in strong_emerging]
            explanation_parts.append(f"Emerging skills show significant potential in {', '.join(emerging_names)}.")

        if weak_emerging:
            weak_names = [s["name"] for s in weak_emerging]
            explanation_parts.append(f"Emerging challenges appear in {', '.join(weak_names)}.")

    # Weaknesses
    weak_stable = [s for s in stable_skills if s["strength"] < -0.1][:2]
    if weak_stable:
        weak_names = [s["name"] for s in weak_stable]
        explanation_parts.append(f"Consistent weaknesses appear in {', '.join(weak_names)}.")

    # Development trajectory
    high_emerging = len([s for s in emerging_skills if s["strength"] > 0.2])
    low_emerging = len([s for s in emerging_skills if s["strength"] < -0.2])

    if high_emerging > low_emerging:
        explanation_parts.append("The player's development trajectory suggests improving capabilities.")
    elif low_emerging > high_emerging:
        explanation_parts.append("Development trends indicate ongoing challenges in certain areas.")
    else:
        explanation_parts.append("The player's skills show balanced development patterns.")

    return " ".join(explanation_parts)

# Test with McDavid data
test_data = {
    'rows': [
        {'dim_idx': 0, 'forecast_mean': 1.585, 'label': 'Play driver', 'is_stable': False, 'stable_seasons': 2},
        {'dim_idx': 1, 'forecast_mean': -0.559, 'label': 'Elite shutdown (HD)', 'is_stable': False, 'stable_seasons': 2},
        {'dim_idx': 2, 'forecast_mean': -1.014, 'label': 'Transition killer', 'is_stable': True, 'stable_seasons': 5},
        {'dim_idx': 3, 'forecast_mean': -0.024, 'label': 'PP quarterback', 'is_stable': True, 'stable_seasons': 3},
        {'dim_idx': 4, 'forecast_mean': 0.0, 'label': 'PK stopper', 'is_stable': True, 'stable_seasons': 3},
        {'dim_idx': 5, 'forecast_mean': 0.828, 'label': 'Transition killer', 'is_stable': False, 'stable_seasons': 1},
        {'dim_idx': 6, 'forecast_mean': -0.007, 'label': 'Transition killer', 'is_stable': True, 'stable_seasons': 5},
        {'dim_idx': 7, 'forecast_mean': -0.166, 'label': 'Two-way profile', 'is_stable': True, 'stable_seasons': 3},
        {'dim_idx': 8, 'forecast_mean': -0.003, 'label': 'Play driver', 'is_stable': True, 'stable_seasons': 3},
        {'dim_idx': 9, 'forecast_mean': 0.0, 'label': 'PK stopper', 'is_stable': True, 'stable_seasons': 4},
        {'dim_idx': 10, 'forecast_mean': 0.0, 'label': 'PP quarterback', 'is_stable': True, 'stable_seasons': 3},
        {'dim_idx': 11, 'forecast_mean': -0.048, 'label': 'Elite shutdown (HD)', 'is_stable': False, 'stable_seasons': 2}
    ]
}

result = _generate_player_explanation(test_data)
print("EXPLANATION GENERATED:")
print(result)