#!/usr/bin/env python3
"""Validate and test LLM explanation accuracy."""

import requests
import json
from typing import Dict, List, Any
import statistics

def validate_explanation_accuracy():
    """Test explanation accuracy against known player profiles."""

    # Test cases with known player archetypes
    test_cases = [
        {
            "player_id": 8478402,  # Connor McDavid
            "expected_profile": "elite_offensive_forward",
            "expected_strengths": ["transition_offense", "playmaking"],
            "expected_weaknesses": ["defense"]
        },
        {
            "player_id": 8475690,  # Christopher Tanev
            "expected_profile": "defensive_specialist",
            "expected_strengths": ["transition_defense"],
            "expected_weaknesses": ["offense", "special_teams"]
        },
        {
            "player_id": 8480803,  # Evan Bouchard
            "expected_profile": "offensive_defenseman",
            "expected_strengths": ["transition_offense"],
            "expected_weaknesses": ["defense", "two_way"]
        }
    ]

    print("🔍 LLM EXPLANATION VALIDATION")
    print("=" * 60)

    for test_case in test_cases:
        player_id = test_case["player_id"]
        expected_profile = test_case["expected_profile"]

        print(f"\n🎯 Testing {expected_profile.upper()} (ID: {player_id})")
        print("-" * 40)

        # Get explanation
        try:
            response = requests.get(f"http://localhost:8000/api/explanations/player/{player_id}?model=sae_apm_v1_k12_a1&season=20242025&window=10&horizon=3")
            response.raise_for_status()
            data = response.json()
            explanation = data["explanation"]
        except Exception as e:
            print(f"❌ API Error: {e}")
            continue

        # Get raw forecast data for comparison
        try:
            forecast_response = requests.get(f"http://localhost:8000/api/player/{player_id}/dlm-forecast?model=sae_apm_v1_k12_a1&season=20242025&window=10&horizon=3")
            forecast_data = forecast_response.json()
        except Exception as e:
            print(f"❌ Forecast API Error: {e}")
            continue

        # Analyze explanation content
        exp_lower = explanation.lower()

        # Check for key indicators
        validation_results = {
            "mentions_stable_skills": "stable skills" in exp_lower or "consistent" in exp_lower,
            "mentions_emerging_skills": "emerging skills" in exp_lower or "developing" in exp_lower,
            "discusses_strengths": any(word in exp_lower for word in ["strengths", "strong", "elite", "excellent"]),
            "discusses_weaknesses": any(word in exp_lower for word in ["weaknesses", "struggles", "challenges", "difficulties"]),
            "trajectory_assessment": any(word in exp_lower for word in ["trajectory", "developing", "improving", "challenges"])
        }

        # Check forecast data alignment
        rows = forecast_data.get("rows", [])
        stable_count = len([r for r in rows if r.get("is_stable", False)])
        emerging_count = len([r for r in rows if not r.get("is_stable", False)])

        # Validate explanation matches data
        explanation_mentions_stable = f"{stable_count} stable" in explanation
        explanation_mentions_emerging = f"{emerging_count} emerging" in explanation

        print("Explanation Content:")
        print(f"  📝 {explanation}")
        print()
        print("Validation Checks:")
        for check, result in validation_results.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check.replace('_', ' ').title()}")
        print()
        print("Data Alignment:")
        print(f"  📊 Stable skills: {stable_count} (mentioned: {explanation_mentions_stable})")
        print(f"  📈 Emerging skills: {emerging_count} (mentioned: {explanation_mentions_emerging})")

        # Overall assessment
        validation_score = sum(validation_results.values()) + explanation_mentions_stable + explanation_mentions_emerging
        max_score = len(validation_results) + 2

        print(f"  🎯 Validation Score: {validation_score}/{max_score}")

        if validation_score >= max_score * 0.8:
            print("  ✅ HIGH ACCURACY - Explanation well-aligned with data")
        elif validation_score >= max_score * 0.6:
            print("  ⚠️ MODERATE ACCURACY - Some alignment issues")
        else:
            print("  ❌ LOW ACCURACY - Significant misalignment")

def analyze_explanation_patterns():
    """Analyze patterns in explanations to identify potential biases."""

    print("\n🔍 EXPLANATION PATTERN ANALYSIS")
    print("=" * 60)

    # Get explanations for multiple players
    player_ids = [8478402, 8477934, 8475690, 8475786, 8478550, 8480803]  # Diverse sample

    explanations = []
    for player_id in player_ids:
        try:
            response = requests.get(f"http://localhost:8000/api/explanations/player/{player_id}?model=sae_apm_v1_k12_a1&season=20242025&window=10&horizon=3")
            if response.status_code == 200:
                data = response.json()
                explanations.append({
                    "player_id": player_id,
                    "explanation": data["explanation"],
                    "stable_count": data["stable_skills"],
                    "emerging_count": data["emerging_skills"]
                })
        except:
            continue

    print(f"Analyzed {len(explanations)} explanations")
    print()

    # Analyze patterns
    trajectory_mentions = []
    for exp in explanations:
        text = exp["explanation"].lower()
        if "improving capabilities" in text:
            trajectory_mentions.append("improving")
        elif "ongoing challenges" in text:
            trajectory_mentions.append("challenges")
        elif "balanced development" in text:
            trajectory_mentions.append("balanced")
        else:
            trajectory_mentions.append("other")

    print("Trajectory Assessment Distribution:")
    for trajectory in ["improving", "challenges", "balanced", "other"]:
        count = trajectory_mentions.count(trajectory)
        pct = count / len(trajectory_mentions) * 100 if trajectory_mentions else 0
        print(f"  {trajectory.title()}: {count} ({pct:.1f}%)")

    # Check for potential biases
    print("\nPotential Biases Detected:")
    improving_count = trajectory_mentions.count("improving")
    challenges_count = trajectory_mentions.count("challenges")

    if abs(improving_count - challenges_count) > len(explanations) * 0.4:
        print("  ⚠️ UNBALANCED TRAJECTORY ASSESSMENT - May be biased toward positive/negative")
    else:
        print("  ✅ BALANCED TRAJECTORY ASSESSMENT")

    # Check explanation lengths
    lengths = [len(exp["explanation"]) for exp in explanations]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    print(f"  📏 Average explanation length: {avg_length:.0f} characters")

    # Check for repetitive patterns
    common_phrases = ["stable skills", "emerging skills", "consistent strengths", "development trajectory"]
    print("\nCommon Phrase Usage:")
    for phrase in common_phrases:
        count = sum(1 for exp in explanations if phrase in exp["explanation"].lower())
        pct = count / len(explanations) * 100 if explanations else 0
        print(f"  '{phrase}': {count}/{len(explanations)} ({pct:.1f}%)")
        print(f"  '{phrase}': {count}/{len(explanations)} ({pct:.1f}%)")

def suggest_improvements():
    """Suggest ways to improve explanation accuracy."""

    print("\n🚀 EXPLANATION IMPROVEMENT SUGGESTIONS")
    print("=" * 60)

    improvements = [
        {
            "category": "Data Validation",
            "suggestions": [
                "Add confidence intervals to forecast assessments",
                "Include statistical significance tests for skill changes",
                "Validate against known player performance metrics",
                "Cross-reference with traditional hockey analytics"
            ]
        },
        {
            "category": "Explanation Logic",
            "suggestions": [
                "Implement context-aware thresholds (adjust for position/skill)",
                "Add trend analysis over multiple seasons",
                "Include uncertainty quantification in explanations",
                "Add comparative context (vs league average, position peers)"
            ]
        },
        {
            "category": "Quality Assurance",
            "suggestions": [
                "Create comprehensive test suite with known player profiles",
                "Implement human review workflow for edge cases",
                "Add explanation confidence scores",
                "Regular validation against domain expert feedback"
            ]
        },
        {
            "category": "Algorithm Enhancement",
            "suggestions": [
                "Replace rule-based with true LLM (GPT-4, Claude) for nuanced explanations",
                "Implement few-shot learning with expert-written examples",
                "Add multi-modal explanations (charts + text)",
                "Include coaching recommendations based on skill gaps"
            ]
        }
    ]

    for improvement in improvements:
        print(f"\n{improvement['category']}:")
        for i, suggestion in enumerate(improvement['suggestions'], 1):
            print(f"  {i}. {suggestion}")

if __name__ == "__main__":
    validate_explanation_accuracy()
    analyze_explanation_patterns()
    suggest_improvements()