#!/usr/bin/env python3
"""
Example: Supervised approach to hockey skill prediction.

Instead of unsupervised SAE, train directly on hockey skill labels.
This would give us the granular attributions the user wants.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from typing import Dict

# Mock training data structure
def create_mock_skill_training_data():
    """Create example training data with hockey skill labels."""

    # Features: RAPM metrics
    feature_names = [
        'corsi_off_rapm_5v5', 'corsi_def_rapm_5v5',
        'xg_off_rapm_5v5', 'xg_def_rapm_5v5',
        'hd_xg_off_rapm_5v5_ge020', 'hd_xg_def_rapm_5v5_ge020',
        'penalties_taken_rapm_5v5',
        'turnover_to_xg_swing_rapm_5v5_w10',
        'xg_pp_off_rapm', 'xg_pk_def_rapm'
    ]

    # Target skills (0-100 scale)
    skill_names = [
        'playmaker', 'sniper', 'net_front_presence',
        'defensive_specialist', 'transition_game', 'physicality'
    ]

    # Generate mock data
    n_samples = 1000
    n_features = len(feature_names)
    n_skills = len(skill_names)

    # Random features (normally from RAPM)
    X = np.random.randn(n_samples, n_features)

    # Skill labels (correlated with features)
    y = np.zeros((n_samples, n_skills))

    # Create realistic correlations
    for i in range(n_samples):
        # Playmaker correlates with offensive RAPM
        y[i, 0] = np.clip(50 + X[i, 0] * 10 + X[i, 2] * 15 + np.random.randn() * 5, 0, 100)

        # Sniper correlates with HD xG offense
        y[i, 1] = np.clip(50 + X[i, 4] * 20 + np.random.randn() * 5, 0, 100)

        # Net front presence correlates with rebound/goal area metrics
        y[i, 2] = np.clip(50 + X[i, 4] * 12 + X[i, 5] * -8 + np.random.randn() * 5, 0, 100)

        # Other skills...
        for j in range(3, n_skills):
            y[i, j] = np.clip(50 + np.sum(X[i, :]) * 5 + np.random.randn() * 10, 0, 100)

    return X, y, feature_names, skill_names

def train_skill_predictors():
    """Train individual models for each hockey skill."""

    X, y, feature_names, skill_names = create_mock_skill_training_data()

    models = {}
    scores = {}

    print("TRAINING SUPERVISED SKILL PREDICTORS")
    print("=" * 50)

    for i, skill_name in enumerate(skill_names):
        print(f"\nTraining {skill_name} predictor...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y[:, i], test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        models[skill_name] = model
        scores[skill_name] = r2

        print(".3f")

        # Feature importance
        importances = model.feature_importances_
        top_features = sorted(zip(feature_names, importances),
                            key=lambda x: x[1], reverse=True)[:3]
        print("  Top features:")
        for feat, imp in top_features:
            print(".3f")

    return models, scores

def predict_player_skills(models: dict, player_features: np.ndarray) -> Dict[str, float]:
    """Predict skill levels for a player."""

    skills = {}
    for skill_name, model in models.items():
        prediction = model.predict(player_features.reshape(1, -1))[0]
        skills[skill_name] = float(prediction)

    return skills

def format_skill_profile(skills: Dict[str, float]) -> str:
    """Format skills into percentages that sum to 100%."""

    # Normalize to sum to 100%
    total = sum(skills.values())
    if total > 0:
        normalized = {skill: (value / total) * 100 for skill, value in skills.items()}
    else:
        normalized = {skill: 100.0 / len(skills) for skill in skills.keys()}

    # Sort and format
    sorted_skills = sorted(normalized.items(), key=lambda x: x[1], reverse=True)

    profile_parts = []
    for skill, pct in sorted_skills:
        if pct >= 5:  # Only show meaningful contributions
            profile_parts.append(f"{pct:.0f}% {skill.replace('_', ' ')}")

    return ", ".join(profile_parts)

if __name__ == "__main__":
    # Train models
    models, scores = train_skill_predictors()

    print("\nMODEL PERFORMANCE SUMMARY")
    print("=" * 50)
    avg_r2 = sum(scores.values()) / len(scores)
    print(".3f")

    # Example prediction
    print("\nEXAMPLE PREDICTION")
    print("=" * 50)

    # Mock player features (high offensive, low defensive)
    player_features = np.array([
        0.8, -0.3,  # High corsi off, low def
        0.9, -0.4,  # High xG off, low def
        1.2, -0.6,  # Very high HD xG off, low def
        -0.1,       # Neutral penalties
        0.3,        # Good turnover swing
        0.7, -0.2   # Good PP off, poor PK def
    ])

    skills = predict_player_skills(models, player_features)
    profile = format_skill_profile(skills)

    print(f"Player skill profile: {profile}")
    print("\nRaw skill scores:")
    for skill, score in sorted(skills.items(), key=lambda x: x[1], reverse=True):
        print(".1f")

    print("\nThis approach would give us the granular skill attributions you want!")
    print("But requires expert-labeled training data and supervised learning.")
    print("Current unsupervised SAE approach is more general but less interpretable.")