import os

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(os.getcwd())

with open("./api_keys/openai_api_key.txt", "r") as file:
    api_key = file.read().strip()
file.close()
# openai.api_key = api_key
os.environ["OPENAI_API_KEY"] = api_key

import argparse
import importlib
import sys
from world import avail_games
from world.model import WorldModel
from world.utils import query_actions

# import torch
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ""))


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

    parser.add_argument("--model", type=str, default="gpt-4o-mini")
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
    parser.add_argument("--last_steps_to_find", type=float, default=1.0)  #
    parser.add_argument("--max_try", type=int, default=50)  #

    parser.add_argument("--game_name", type=str, default="")

    args = parser.parse_args()
    return args


def main():
    game_code_folder = "./games"

    game_list = avail_games["games"]
    args = parse_args()

    for last_steps_to_verify in [args.last_steps_to_find]:
        game_result_dict = {}
        for game_name in game_list:
            print(game_name)

            max_try = args.max_try
            game_result_dict[game_name] = {}
            avail_embds = {}

            success_idx = []

            for idx in range(max_try):
                try:
                    if os.path.dirname(game_code_folder) not in sys.path:
                        sys.path.append(game_code_folder)
                    TextGame = importlib.import_module(game_name).TextGame
                    game_random_seed = importlib.import_module(game_name).randomSeed
                    real_random_seed = game_random_seed + 10 * idx

                    game = TextGame(randomSeed=real_random_seed)

                    args.game_name = game_name

                    demo_actions = game.get_demo_actions()

                    wm = WorldModel(args=args, game=game)

                    # executing the demo actions until the

                    demo_length = int(len(demo_actions) * (1 - last_steps_to_verify))
                    for step, action in enumerate(demo_actions):
                        if step < demo_length:
                            wm.step(action=action, with_predict=False)
                        else:
                            break
                    current_state, _ = wm.get_current_state_for_prompt()

                    return_actions = wm.planning(
                        max_steps=2 * (len(demo_actions) - demo_length),
                        current_state=current_state,
                    )

                    for action_idx, action in enumerate(return_actions):
                        real_avail_actions = list(
                            wm.game.generatePossibleActions().keys()
                        )

                        queried_actions, avail_embds = query_actions(
                            [action],
                            real_avail_actions,
                            pre_avail_embedding=avail_embds,
                        )

                        results_from_game = wm.step(
                            action=queried_actions[0], with_predict=False
                        )

                        if results_from_game["real"][1]["gameWon"]:
                            game_result_dict[game_name][real_random_seed] = True
                            break
                        else:
                            game_result_dict[game_name][real_random_seed] = False
                        print(f"{game_name},{real_random_seed}")
                        print(game_result_dict[game_name][real_random_seed])
                        success_idx.append(idx)

                        if len(success_idx) > 30:
                            break
                except:
                    continue

            # results_folder = "./results/results_data"
            # with open(
            #     f"{results_folder}/results_finding_{last_steps_to_verify}_{args.model}.json",
            #     "w",
            # ) as f:
            #     json.dump(game_result_dict, f, indent=4)

    print("end of the program")


if __name__ == "__main__":
    main()
