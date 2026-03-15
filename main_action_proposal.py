import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(os.getcwd())


import argparse
import importlib
import sys
from world import avail_games
from world.model import WorldModel

# import torch
import json
from world.utils import query_actions

sys.path.append(os.path.join(os.path.dirname(__file__), ""))
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_data_folder", type=str, default="data")
    parser.add_argument("--test_data", type=str, default="test.jsonl")
    parser.add_argument("--example_data", type=str, default="./data/examples.json")
    parser.add_argument(
        "--rule_folder", type=str, default="./world/rules/human_written_rules"
    )
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--output_prefix", type=str, default="")
    parser.add_argument("--output_suffix", type=str, default="")

    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument(
        "--data_distribution_file",
        type=str,
        default="./data/dynamic_static_states_per_action.json",
    )
    parser.add_argument(
        "--state_change_file", type=str, default="./data/dynamic_states.json"
    )  # fine-grained state change info

    parser.add_argument("--game_file_names", default="./experiments/games.json")
    parser.add_argument("--shard_idx", type=int, default=0)  # zero-base
    parser.add_argument("--total_shards", type=int, default=1)

    parser.add_argument("--partial", default=True)
    parser.add_argument(
        "--data_type",
        type=str,
        default="full",
        choices=["action", "tick", "score", "full"],
    )
    parser.add_argument("--no_rule", default=False)

    # Make a boolean parameter, "notlive", which defaults to False
    parser.add_argument("--notlive", dest="notlive", action="store_true", default=False)

    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_try", type=int, default=30)  #

    args = parser.parse_args()
    return args


def main():
    game_code_folder = "./games"

    game_list = avail_games["games"]
    args = parse_args()

    for topk in [args.topk]:
        game_result_dict = {}
        for game_name in game_list:
            # if game_name not in missing_game_list[args.model][topk]:
            #     continue
            print(game_name)

            avail_embds = {}

            max_try = args.max_try
            # max_try = 1
            game_result_dict[game_name] = {}
            for idx in range(max_try):
                if os.path.dirname(game_code_folder) not in sys.path:
                    sys.path.append(game_code_folder)
                TextGame = importlib.import_module(game_name).TextGame
                game_random_seed = importlib.import_module(game_name).randomSeed
                real_random_seed = game_random_seed + 10 * idx

                game = TextGame(randomSeed=real_random_seed)

                args.game_name = game_name

                demo_actions = game.get_demo_actions()

                wm = WorldModel(args=args, game=game)

                game_result_dict[game_name][real_random_seed] = {
                    "action_seq": [],
                    "score": 0,
                }

                for step, action in enumerate(demo_actions):
                    # print("This is the step action")
                    # print(action)
                    if action in ["look", "inventory"]:
                        continue

                    print("=" * 100)
                    results = wm.get_action_proposals(k=topk)
                    real_avail_actions = list(wm.game.generatePossibleActions().keys())
                    # print(real_avail_actions)
                    results_avail_actions = results["avail_actions"]
                    if type(results_avail_actions) == str:
                        results_avail_actions = [results_avail_actions]
                    recommend_avail_actions = results_avail_actions[
                        : min(topk, len(results["avail_actions"]))
                    ]
                    # print(recommend_avail_actions)
                    print("=" * 30)
                    print("This step action")
                    print(action)
                    print("=" * 30)
                    print("This action proposals")
                    print(results)
                    queried_actions, avail_embds = query_actions(
                        recommend_avail_actions,
                        real_avail_actions,
                        pre_avail_embedding=avail_embds,
                    )

                    print("This is queried actions")
                    print(queried_actions)
                    wm.step(action=action, with_predict=False)
                    print("=" * 100)

                    game_result_dict[game_name][real_random_seed]["action_seq"].append(
                        True if action in queried_actions else False
                    )
                score = np.sum(
                    np.array(
                        game_result_dict[game_name][real_random_seed]["action_seq"]
                    )
                ) / len(game_result_dict[game_name][real_random_seed]["action_seq"])
                game_result_dict[game_name][real_random_seed]["score"] = score
                print(score)
                # break
            results_folder = "./results/results_data"
            with open(
                f"{results_folder}/results_action_proposal_{topk}_{args.model}.json",
                "w",
            ) as f:
                json.dump(game_result_dict, f, indent=4)
    print("end of the program")


if __name__ == "__main__":
    main()
