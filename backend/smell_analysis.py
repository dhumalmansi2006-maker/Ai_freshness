def analyze_smell(smell_description):
    """
    Analyze the user's smell description
    and return a smell-based freshness result.
    """

    smell = smell_description.lower().strip()

    # Spoiled / bad smell keywords
    spoiled_keywords = [
        "rotten",
        "foul",
        "bad",
        "putrid",
        "decaying",
        "decayed",
        "very bad",
        "terrible",
        "awful",
        "sewage",
        "fermented",
        "strong bad smell",
        "rotten smell"
    ]

    # Semi-fresh / slightly changed smell keywords
    semi_fresh_keywords = [
        "slightly sour",
        "sour",
        "strong",
        "unusual",
        "different",
        "stale",
        "musty",
        "slight smell",
        "slightly bad",
        "fermented smell"
    ]

    # Fresh / normal smell keywords
    fresh_keywords = [
        "fresh",
        "normal",
        "good",
        "pleasant",
        "sweet",
        "natural",
        "no smell",
        "normal smell",
        "fresh smell"
    ]

    # Check for spoiled smell
    for keyword in spoiled_keywords:
        if keyword in smell:
            return "Spoiled"

    # Check for semi-fresh smell
    for keyword in semi_fresh_keywords:
        if keyword in smell:
            return "Semi-Fresh"

    # Check for fresh smell
    for keyword in fresh_keywords:
        if keyword in smell:
            return "Fresh"

    # If the description doesn't match known keywords
    return "Unknown"


# Test the smell analyzer
if __name__ == "__main__":

    print("=== Smell Analysis ===")

    smell = input("Enter smell description: ")

    result = analyze_smell(smell)

    print("\nSmell Result:", result)