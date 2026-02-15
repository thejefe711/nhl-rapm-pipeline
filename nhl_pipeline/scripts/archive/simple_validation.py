#!/usr/bin/env python3
"""Simple validation of LLM explanations."""

import requests
import json

def test_explanations():
    """Test a few key explanations for accuracy."""

    test_players = [
        (8478402, "Connor McDavid", "Should show elite offense, defensive struggles"),
        (8480803, "Evan Bouchard", "Should show elite transition offense, defensive concerns"),
        (8475690, "Christopher Tanev", "Should show defensive strengths, offensive struggles")
    ]

    print("LLM EXPLANATION VALIDATION")
    print("=" * 50)

    for player_id, name, expected in test_players:
        print(f"\nTesting {name} (ID: {player_id})")
        print(f"Expected: {expected}")
        print("-" * 40)

        try:
            response = requests.get(f"http://localhost:8000/api/explanations/player/{player_id}?model=sae_apm_v1_k12_a1&season=20242025&window=10&horizon=3")
            response.raise_for_status()
            data = response.json()

            print(f"Explanation: {data['explanation']}")
            print(f"Stable skills: {data['stable_skills']}, Emerging: {data['emerging_skills']}")

            # Basic validation
            explanation = data['explanation'].lower()
            has_stable = 'stable' in explanation
            has_emerging = 'emerging' in explanation
            has_strengths = any(word in explanation for word in ['strength', 'strong', 'elite'])
            has_weaknesses = any(word in explanation for word in ['weakness', 'struggle', 'challenge'])

            print(f"Contains stable mention: {has_stable}")
            print(f"Contains emerging mention: {has_emerging}")
            print(f"Discusses strengths: {has_strengths}")
            print(f"Discusses weaknesses: {has_weaknesses}")

        except Exception as e:
            print(f"ERROR: {e}")

def analyze_patterns():
    """Analyze common patterns in explanations."""

    print("\nEXPLANATION PATTERN ANALYSIS")
    print("=" * 50)

    player_ids = [8478402, 8477934, 8475690, 8475786, 8478550, 8480803]

    explanations = []
    for player_id in player_ids:
        try:
            response = requests.get(f"http://localhost:8000/api/explanations/player/{player_id}?model=sae_apm_v1_k12_a1&season=20242025&window=10&horizon=3")
            if response.status_code == 200:
                data = response.json()
                explanations.append(data['explanation'])
        except:
            continue

    print(f"Collected {len(explanations)} explanations")

    # Count trajectory types
    improving = sum(1 for exp in explanations if 'improving capabilities' in exp.lower())
    challenges = sum(1 for exp in explanations if 'ongoing challenges' in exp.lower())
    balanced = sum(1 for exp in explanations if 'balanced development' in exp.lower())

    print(f"Trajectory assessments:")
    print(f"  Improving: {improving}")
    print(f"  Challenges: {challenges}")
    print(f"  Balanced: {balanced}")

    if abs(improving - challenges) > len(explanations) * 0.3:
        print("WARNING: Unbalanced trajectory distribution - may indicate bias")
    else:
        print("OK: Balanced trajectory distribution")

def improvement_suggestions():
    """Provide suggestions for improving explanations."""

    print("\nIMPROVEMENT SUGGESTIONS")
    print("=" * 50)

    suggestions = [
        "1. Add confidence scores to explanations based on forecast variance",
        "2. Include trend direction (improving/declining/stable) with evidence",
        "3. Add position-specific context (forward vs defenseman expectations)",
        "4. Include uncertainty quantification ('likely', 'possibly', 'uncertain')",
        "5. Cross-validate against known player reputations and stats",
        "6. Add comparative context (vs position average, team peers)",
        "7. Implement human expert review for edge cases",
        "8. Add explanation versioning and A/B testing",
        "9. Include actionable coaching recommendations",
        "10. Replace rule-based with true LLM for more nuanced explanations"
    ]

    for suggestion in suggestions:
        print(f"  {suggestion}")

if __name__ == "__main__":
    test_explanations()
    analyze_patterns()
    improvement_suggestions()