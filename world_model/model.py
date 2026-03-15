import os
import importlib
import sys
import json


class WorldModel:
    def __init__(self, args, game):
        print()

        self.game = game
        self.game_name = args.game_name
        self.task_desc = game.getTaskDescription()

        args.rule_folder = "./world/rules/llm_generated_rules"

        # load object rules
        with open(os.path.join(args.rule_folder, f"object_rules.json")) as f:
            self.obj_rules = json.load(f)[self.game_name]

        with open(os.path.join(args.rule_folder, f"action_rules.json")) as f:
            self.action_rules = json.load(f)[self.game_name]

        with open(os.path.join(args.rule_folder, f"score_rules.json")) as f:
            self.score_rules = json.load(f)[self.game_name]

        print("=" * 10 + "obj rules: start" + "=" * 10)
        print(self.obj_rules)
        print("=" * 10 + "obj rules: end" + "=" * 10)
        print("=" * 10 + "action rules: start" + "=" * 10)
        print(self.action_rules)
        print("=" * 10 + "action rules: end" + "=" * 10)
        print("=" * 10 + "score rules: start" + "=" * 10)
        print(self.score_rules)
        print("=" * 10 + "score rules: end" + "=" * 10)

    def build_prompt_for_prediction(self):
        prompt = (
            "You are a simulator of a text game. Read the task description of a text game. "
            "Given the current game state in JSON, "
            "you need to decide the new game state after taking an action including the game score.\n"
        )

        # Task
        prompt += "Here is the game that you need to simulate:\n"
        prompt += "Task Description:\n"
        prompt += f"{self.task_desc}\n"

        # prompt += "Here are the descriptions of all game objects properties:\n"
        # prompt += data_obj_desc.strip()
        # prompt += "\n"
        # if args.data_type == "action" or args.data_type == "full":
        #     prompt += "Here are the descriptions of all game actions:\n"
        #     prompt += data_action_desc.strip()
        #     prompt += "\n"
        # if args.data_type == "score" or args.data_type == "full":
        #     prompt += "Here is a description of the game score function:\n"
        #     prompt += data_score_desc.strip()
        #     prompt += "\n"
