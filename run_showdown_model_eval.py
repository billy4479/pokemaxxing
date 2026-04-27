#!/usr/bin/env python3
import json

from model_eval import ChoiceRequest, ModelConfig, ModelEvalClient


def main() -> int:
    SEED = "1,2,3,4"
    OPPONENTS = 5

    config = ModelConfig(
        types=("Water", "Dragon"),
        base_stats={"hp": 90, "atk": 70, "def": 85, "spa": 120, "spd": 95, "spe": 100},
        ability="Torrent",
        moves=("Surf", "Draco Meteor", "Protect", "Ice Beam"),
    )

    def chooser(req: ChoiceRequest) -> str:
        print(json.dumps(req, indent=2) + "\n" * 3)

        if req["request"].get("teamPreview"):
            return "team 1"
        return "move 1"

    with ModelEvalClient() as client:
        results = client.run_eval_batch(
            config=config,
            opponent_count=OPPONENTS,
            sample_seed=SEED,
            chooser=chooser,
        )

    p1_wins = sum(1 for result in results if result.winner == "p1")
    print(f"Ran {len(results)} battles with the sample config")
    print(f"p1 wins: {p1_wins}/{len(results)}")
    for result in results:
        print(
            f"- {result.battle_id}: opponent={result.opponent_species} "
            f"winner={result.winner} turns={result.turns} seed={result.seed}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
