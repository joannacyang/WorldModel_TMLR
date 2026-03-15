import os
import importlib
import json
from world.make_state import (
    make_game_state,
    make_game_state_partial,
    preprocess_obj_desc,
    get_state,
    recover_game_state_from_partial,
)
import copy

from world.utils import stream_llm_gpt, gpt_models


class WorldModel:
    def __init__(self, args, game):
        # print()

        self.args = args
        self.game = game
        self.game_name = args.game_name
        self.task_desc = game.getTaskDescription()

        args.rule_folder = "./world/rules/human_written_rules"
        args.example_data = "./world/examples.json"

        # load object rules
        with open(os.path.join(args.rule_folder, f"object_rules.json")) as f:
            self.obj_rules = json.load(f)

        with open(os.path.join(args.rule_folder, f"action_rules.json")) as f:
            self.action_rules = json.load(f)

        with open(os.path.join(args.rule_folder, f"score_rules.json")) as f:
            self.score_rules = json.load(f)

        with open(args.example_data) as f:
            self.example_lut = json.load(f)

        with open(args.state_change_file) as f:
            self.state_change_info = json.load(f)

        self.last_actions = []
        self.predicted_states = []
        self.predicted_scores = []

    def step(self, action, use_env=False, with_predict=True):
        self.game.generatePossibleActions()

        predicted_state, predicted_score = None, None
        if with_predict:
            if len(self.predicted_states) < 1:
                use_env = True
            # get predicted state
            # predicted_state, predicted_score = None, None
            if use_env:
                predicted_state, predicted_score = self.get_predicted_state(action)
            else:
                predicted_state, predicted_score = self.get_predicted_state(
                    action, self.predicted_states[-1]
                )
        self.predicted_states.append(copy.deepcopy(predicted_state))
        self.predicted_scores.append(copy.deepcopy(predicted_score))

        # get the true state
        obs, _, _, _, _ = self.game.step(action)
        # print('='*100)
        # print("This is the current step obs")
        # print(obs)
        # print('=' * 100)
        self.last_actions.append(action)
        last_action = "" if len(self.last_actions) == 0 else self.last_actions[-1]

        max_UUID = importlib.import_module(self.game_name).UUID
        real_state = get_state(self.game, last_action, max_UUID, self.game_name)
        real_state = make_game_state(real_state)

        # print(max_UUID)

        real_score = {
            "score": self.game.score,
            "gameOver": self.game.gameOver,
            "gameWon": self.game.gameWon,
        }

        return {
            "real": [real_state, real_score],
            "predict": [predicted_state, predicted_score],
        }

    def get_current_state_for_prompt(self):
        print()
        last_action = "" if len(self.last_actions) == 0 else self.last_actions[-1]
        max_UUID = importlib.import_module(self.game_name).UUID
        current_state = get_state(self.game, last_action, max_UUID, self.game_name)
        current_state_for_prompt = make_game_state(current_state)
        max_uuid = current_state["max_UUID"]

        return current_state_for_prompt, max_uuid

    def get_predicted_state(self, action, current_state=None):
        prompt = (
            "You are a simulator of a text game. Read the task description of a text game. "
            "Given the current game state in JSON, "
            "you need to decide the new game state after taking an action including the game score.\n"
        )
        prompt += (
            "Your response should be in the JSON format. "
            "It should have three keys: 'modified', 'removed', and 'score'. "
            "The 'modified' key stores a list of all the object states that are added or changed after taking the action. "
            "Keep it an empty list if no object is added or modified. "
            "The 'removed' key stores a list of uuids of the objects that are removed. "
            "Keep it an empty list if no object is removed. "
            "The 'score' key stores a JSON with three keys: "
            "'score', 'gameOver', and 'gameWon'. "
            "'score' stores the current game score, "
            "'gameOver' stores a bool value on whether the game is over, "
            "and 'gameWon' stores a bool value on whether the game is won. \n"
        )

        if current_state is None:
            current_state_for_prompt, max_uuid = self.get_current_state_for_prompt()
        else:
            # print("use the predicted state")

            current_state_for_prompt = current_state
            max_uuid = len(current_state["game_state"])

            # start adding examples
        example_prompt = self.build_examples()
        prompt += example_prompt
        # end of adding examples
        # Task
        prompt += "Here is the game that you need to simulate:\n"
        prompt += "Task Description:\n"
        prompt += f"{self.task_desc}\n"

        # load rules
        obj_desc = preprocess_obj_desc(self.obj_rules[self.game_name])
        action_desc = self.action_rules[self.game_name]
        score_desc = self.score_rules[self.game_name]

        prompt += "Here are the descriptions of all game objects properties:\n"
        prompt += obj_desc.strip()
        prompt += "\n"
        prompt += "Here are the descriptions of all game actions:\n"
        prompt += action_desc.strip()
        prompt += "\n"
        prompt += "Here is a description of the game score function:\n"
        prompt += score_desc.strip()
        prompt += "\n"

        # data_state, data_UUID_base, data_action = None, None, None
        prompt += "Here is the game state:\n"
        prompt += f"{current_state_for_prompt}\n"
        prompt += "\n"

        prompt += f"The current game UUID base is {max_uuid}\n"
        prompt += f"The action to take is:\n{action}\n"

        assert self.args.model in ["gpt-4o", "gpt-4o-mini"]

        model = gpt_models[self.args.model]

        response = stream_llm_gpt(
            prompt,
            model=model,
            response_format={"type": "json_object"},
            # temperature=0.7,
        )

        try:
            prediction = json.loads(response)
            game_score = prediction["score"]
            prediction = recover_game_state_from_partial(
                current_state_for_prompt, prediction, has_score=False
            )
            return prediction, game_score
        except Exception as e:
            # evaluate() will handle this format error
            print(e)
            return response

    def build_examples(self):
        # adding example to the prompt
        example_lut = self.example_lut
        state_change_info = self.state_change_info

        curr_state_key = "current_state"
        next_state_key = "tick_state"

        # Load one example that the game state is changed by an action
        example_action_change = example_lut["full"]["action"]
        current_score_state = example_action_change["current_score_state"]
        next_score_state = example_action_change["next_score_state"]
        example_state_a = make_game_state(example_action_change[curr_state_key])
        example_state_a["game_state"].append(current_score_state)
        example_target_state_a = make_game_state(example_action_change[next_state_key])
        example_target_state_a["game_state"].append(next_score_state)
        example_target_state_a_partial = ""
        if self.args.partial:
            example_target_state_a_partial = make_game_state_partial(
                example_state_a, example_target_state_a
            )

        example_action_a = example_action_change[next_state_key]["lastAction"]
        example_task_desc = example_action_change["current_state"]["taskDesc"]
        example_UUID_base_a = example_action_change[curr_state_key]["max_UUID"]

        example_game = "dishwasher"
        time_change_states = [
            s
            for s in state_change_info[example_game]["time_change"]
            if s not in state_change_info[example_game]["action_change"]
        ]

        # build the time change
        example_time_change = None
        example_state_t = ""
        # example_target_state_a_partial = ''
        example_target_state_t_partial = ""
        example_UUID_base_t = ""
        example_action_t = ""
        example_target_state_t = ""
        if len(time_change_states) > 0:
            example_time_change = example_lut["full"]["tick"]

            current_score_state = example_time_change["current_score_state"]
            next_score_state = example_time_change["next_score_state"]
            example_state_t = make_game_state(example_time_change[curr_state_key])
            example_state_t["game_state"].append(current_score_state)
            example_target_state_t = make_game_state(
                example_time_change[next_state_key]
            )
            example_target_state_t["game_state"].append(next_score_state)
            if self.args.partial:
                example_target_state_t_partial = make_game_state_partial(
                    example_state_t, example_target_state_t
                )

            example_action_t = example_time_change[next_state_key]["lastAction"]
            example_UUID_base_t = example_time_change[curr_state_key]["max_UUID"]

        # example_obj_desc = preprocess_obj_desc(self.obj_rules[example_game])
        # example_score_desc = self.score_rules[example_game]
        example_obj_desc = preprocess_obj_desc(self.obj_rules[example_game])

        example_action_desc = self.action_rules[example_game]

        example_score_desc = self.score_rules[example_game]

        prompt = ""

        prompt += "Note that while game states can be changed by actions, some game states may change over the time, which is described in the tick function of each object class. \n"
        if example_time_change is not None:
            prompt += "Here are two examples of both cases. Both examples are from the same example game.\n"

        prompt += "Example game task description:\n"
        prompt += f"{example_task_desc}\n"
        # if not args.no_rule:
        prompt += "Here are the descriptions of all game objects properties in the example game:\n"
        prompt += example_obj_desc.strip()
        prompt += "\n"
        prompt += "Here are the descriptions of all game actions in the example game:\n"
        prompt += example_action_desc.strip()
        prompt += "\n"
        prompt += "Here is a description of the score function of the example game:\n"
        prompt += example_score_desc.strip()
        prompt += "\n"

        if example_time_change is not None:
            # Example 1: a game state is changed by an action
            prompt += "In the first example, the game state is changed by an action:\n"

        prompt += "Here is the game state:\n"
        prompt += f"{example_state_a}\n"
        prompt += "\n"

        prompt += f"The current game UUID base is {example_UUID_base_a}\n"

        prompt += f"The action to take is: {example_action_a}\n"
        prompt += "The expected response is:\n"
        if self.args.partial:
            prompt += f"{example_target_state_a_partial}\n"
        else:
            prompt += f"{example_target_state_a}\n"
        prompt += "\n"

        # Example 2: a game state is changed over time
        if example_time_change is not None:
            prompt += "In the second example from the same example game, the game state is changed over the time. Note that while in this example the game state is changed by time only, it is possible that a game state is changed by both an action and time.\n"

            prompt += "Here is the game state:\n"
            prompt += f"{example_state_t}\n"
            prompt += "\n"

            prompt += f"The current game UUID base is {example_UUID_base_t}\n"
            prompt += f"The action to take is: {example_action_t}\n"
            prompt += "The expected response is:\n"
            if self.args.partial:
                prompt += f"{example_target_state_t_partial}\n"
            else:
                prompt += f"{example_target_state_t}\n"
            prompt += "\n"

        return prompt

    def get_action_proposals(self, current_state=None, k=5):
        prompt = (
            "You are a simulator of a text game. "
            "Read the task description and the descriptions of all game actions of a text game. "
            "Given the current game state in JSON, and the previous actions that lead to the current game state, "
            "you need to decide the most {} actions "
            "that can help to complete the task step by step at the current state.\n".format(
                k
            )
        )
        prompt += (
            "Each of your action should in one phrase with one verb and the objects it operates on. "
            "Examples of actions includes:\n"
            "move south" + ",\n"
            "detect with metal detector (ID: 15)" + ",\n"
            "dig with shovel (ID: 16)" + ",\n"
            "open freezer (ID: 2)" + ",\n"
            "put ice cube tray (ID: 3) in sink (ID: 4)" + ",\n"
            "dice patato (ID: 2) with knife (ID: 8)" + ",\n"
            "give Type O negative blood (ID: 3) to patient (ID: 2)" + ",\n"
            "read cook book (ID: 7)" + ".\n"
        )

        prompt += (
            "Your response should be in the JSON format. "
            "It should have one key: 'avail_actions', which includes the list of the recommended actions. \n"
        )

        last_action = "" if len(self.last_actions) == 0 else self.last_actions[-1]
        max_UUID = importlib.import_module(self.game_name).UUID
        if current_state is None:
            current_state = get_state(self.game, last_action, max_UUID, self.game_name)
            current_state_for_prompt = make_game_state(current_state)
            max_uuid = current_state["max_UUID"]
        else:
            # print("use the predicted state")

            current_state_for_prompt = current_state
            max_uuid = len(current_state["game_state"])

            # start adding examples
        # example_prompt = self.build_examples()
        # prompt += example_prompt
        # end of adding examples
        # Task
        prompt += "Here is the game that you need to simulate:\n"
        prompt += "Task Description:\n"
        prompt += f"{self.task_desc}\n"

        # load rules
        obj_desc = preprocess_obj_desc(self.obj_rules[self.game_name])
        action_desc = self.action_rules[self.game_name]
        score_desc = self.score_rules[self.game_name]

        prompt += "Here are the descriptions of all game objects properties:\n"
        prompt += obj_desc.strip()
        prompt += "\n"
        prompt += "Here are the descriptions of all game actions:\n"
        prompt += action_desc.strip()
        prompt += "\n"
        prompt += "Here is a description of the game score function:\n"
        prompt += score_desc.strip()
        prompt += "\n"

        # data_state, data_UUID_base, data_action = None, None, None
        prompt += "Here is the game state:\n"
        prompt += f"{current_state_for_prompt}\n"
        prompt += "\n"

        prompt += f"The current game UUID base is {max_uuid}\n"

        if len(self.last_actions) == 0:
            prompt += "There is no previous actions."
        else:
            prompt += "The previous actions {}:\n".format(
                "is" if len(self.last_actions) == 1 else "are"
            )
            for action in self.last_actions:
                prompt += action + "\n"
        response = stream_llm_gpt(
            prompt,
            model=self.args.model,
            response_format={"type": "json_object"},
        )

        try:
            action_proposals = json.loads(response)
            return action_proposals
        except Exception as e:
            print(e)
            return response

    def planning(self, current_state=None, max_steps=10):
        # planning_step_by_step = self.args.planning_step_by_step
        plan_actions = []

        # print(max_steps)

        for step in range(max_steps):
            # print()

            action = self.get_action_proposals(k=1)["avail_actions"][0]
            # print(action)

            next_state, score = self.get_predicted_state(action, current_state)

            current_state = next_state
            self.last_actions.append(action)
            plan_actions.append(action)
            if "gameOver" in score.keys():
                if score["gameOver"]:
                    break

        return plan_actions
