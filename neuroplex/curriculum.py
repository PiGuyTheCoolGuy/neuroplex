"""Small, explicit habitat stages. Progress requires recent survival performance."""

STAGES = (
    {"name": "Foraging", "food_fraction": 1.0, "vision_fraction": 1.0, "water": False, "predators": 0},
    {"name": "Scarce food", "food_fraction": 0.65, "vision_fraction": 0.85, "water": False, "predators": 0},
    {"name": "Food and water", "food_fraction": 0.65, "vision_fraction": 0.85, "water": True, "predators": 0},
    {"name": "First predator", "food_fraction": 0.65, "vision_fraction": 0.85, "water": True, "predators": 1},
    {"name": "Wild habitat", "food_fraction": 0.40, "vision_fraction": 0.65, "water": True, "predators": 2},
)


def ready_to_advance(world, samples, started_at, minimum_seconds):
    if world.stage >= len(STAGES) - 1 or world.time - started_at < minimum_seconds:
        return False
    recent = [p for p in samples if p["stage"] == world.stage and p["age"] >= world.time - 60]
    if len(recent) < 12 or recent[-1]["age"] - recent[0]["age"] < 50:
        return False
    if min(p["energy"] for p in recent) < 45 or recent[-1]["food_per_minute"] < 6:
        return False
    if world.water_active and (min(p["hydration"] for p in recent) < 35
                               or recent[-1]["drinks"] <= recent[0]["drinks"]):
        return False
    return min(p["health"] for p in recent) >= 75
