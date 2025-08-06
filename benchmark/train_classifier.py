import argparse
from collections import defaultdict
from functools import partial
import json
from os import path, environ
from random import choices, seed
import random
from copy import deepcopy

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, IntervalStrategy
from datasets import Dataset, Value
from peft import LoraConfig, TaskType, get_peft_model
import torch


# Helper function to process Task 1 data
def process_task1(scenario, precepts, values, force_binary, harder):
    results = []
    for precept, value in zip(precepts, values):
        score = str(precept['score']).strip()
        if score == "n/a":
            score = "0"
        score = float(score)
        if "additional_score" in precept:
            additional_score = str(precept['additional_score']).strip()
            if additional_score == "n/a":
                additional_score = "0"
            score = (score + float(additional_score)) / 2
        if force_binary:
            # this automatically handles "n/a" as 0
            score_boundary = 4 if harder else 3
            score = 1 if score >= score_boundary else 0
        results.append({
            'text': f"Scenario: {scenario}\nValue: {value}\nPrecept: {precept['conclusion']}\n[REASONING] {precept['reasoning']} [/REASONING]",
            'label': score,
            'task': 1,
            'scenario': scenario,
            'value': value,
            'precept': precept['conclusion'],
            'reasoning': precept['reasoning']
        })
    return results


# Helper function to process Task 2 data
def process_task2(scenario, action, consequence, precept, evaluation):
    results = []
    for key in ['satisfied', 'contradicted']:
        if evaluation[key]["score"] == "bad":
            continue
        additional_score = float(evaluation[key]["score"])
        if "additional_score" in evaluation[key]:
            additional_score = str(evaluation[key]['additional_score']).strip()
            if additional_score == "bad":
                continue
        score = (float(evaluation[key]["score"]) + float(additional_score)) / 2
        input_text = f"Scenario: {scenario}\nAction: {action}\nConsequence: {consequence}\nPrecept: {precept}\n[REASONING] {evaluation[key]['reasoning']} [/REASONING]"
        results.append({
            'text': input_text,
            'label': score,
            'task': 2,
            'scenario': scenario,
            'action': action,
            'consequence': consequence,
            'precept': precept,
            'reasoning': evaluation[key]['reasoning']
        })
    return results


# Load dataset from local JSON file
def load_dataset_from_json(
    file_path,
    task,
    force_binary,
    test_size=0.2,
    balanced=False,
    augment=False,
    augment_ratio=0.5,
    harder=False
):
    force_binary = force_binary
    with open(file_path, 'r') as f:
        data = json.load(f)

    processed_data = defaultdict(list)
    for item in data['data']:
        scenario = item['scenario']
        if task <= 1:
            processed_data[scenario].extend(process_task1(
                scenario,
                item['precepts'],
                data['moral_values'],
                force_binary,
                harder
            ))
        if task == 2 or task == 0:
            if "actions" not in item or "consequences" not in item or "evaluations" not in item:
                continue
            for action, consequence, evaluations in zip(item['actions'], item['consequences'], item['evaluations']):
                for precept, evaluation in zip(item['precepts'], evaluations):
                    processed_data[scenario].extend(process_task2(
                        scenario,
                        action,
                        consequence,
                        precept['conclusion'],
                        evaluation
                    ))

    train_data = []
    train_data_scenarios = []
    test_data = []

    if task == 0:
        task1_only_scenarios = list(filter(lambda x: not any(map(lambda y: y['task'] == 2, x)), processed_data.values()))
        task_both_scenarios = list(filter(lambda x: any(map(lambda y: y['task'] == 2, x)), processed_data.values()))
        seed(42)
        for step, items in enumerate(task1_only_scenarios):
            if step / len(processed_data) < test_size:
                test_data.extend(items)
            else:
                train_data_scenarios.append(items[0]['scenario'])
                train_data.extend(items)
        for step, items in enumerate(task_both_scenarios):
            task1_items = list(filter(lambda x: x['task'] == 1, items))
            task2_items = list(filter(lambda x: x['task'] == 2, items))
            if step / len(processed_data) < test_size:
                task2_items = choices(task2_items, k=len(task1_items))
                test_data.extend(task1_items)
                test_data.extend(task2_items)
            else:
                if balanced:
                    task2_items = choices(task2_items, k=len(task1_items))
                train_data.extend(task1_items)
                train_data.extend(task2_items)
    else:
        for step, (scenario, items) in enumerate(processed_data.items()):
            if step / len(processed_data) < test_size:
                test_data.extend(items)
            else:
                train_data.extend(items)

    # Augment data if requested
    if augment:
        seed(42)
        augmented_data = defaultdict(list)

        # Task 1 augmentation
        if task <= 1:
            augmented_data_task1 = []
            all_task1_data = [(scenario, item) for (scenario, items) in processed_data.items() for item in items if item['task'] == 1]

            # (i) Scramble within scenarios
            for scenario, items in processed_data.items():
                if scenario not in train_data_scenarios:
                    continue  # preserve test set
                task1_items = [item for item in items if item['task'] == 1]
                task1_items_values = [i['value'] for i in task1_items]
                task1_items_precepts = [i['precept'] for i in task1_items]
                task1_items_reasonings = [i['reasoning'] for i in task1_items]
                for item in task1_items:
                    new_item = deepcopy(item)
                    # Randomly reassign value, precept, and reasoning within the tuple
                    new_value = random.choice(task1_items_values)
                    new_precept = random.choice(task1_items_precepts)
                    new_reasoning = random.choice(task1_items_reasonings)
                    if new_value == item['value'] and new_precept == item['precept'] and new_reasoning == item['reasoning']:
                        continue
                    new_item['text'] = f"Scenario: {scenario}\nValue: {new_value}\nPrecept: {new_precept}\n[REASONING] {new_reasoning} [/REASONING]"
                    new_item['value'] = new_value
                    new_item['precept'] = new_precept
                    new_item['reasoning'] = new_reasoning
                    new_item["score"] = 0 if force_binary else "0"
                    augmented_data_task1.append((scenario, new_item))

            # (ii) Scramble across scenarios
            for scenario in processed_data.keys():
                scrambled = random.sample(all_task1_data, len(all_task1_data))
                for old_scenario, item in scrambled:
                    if old_scenario not in train_data_scenarios:
                        continue  # preserve test set
                    if scenario == old_scenario:  # Avoid identical matches
                        continue
                    new_item = deepcopy(item)
                    new_item['text'] = f"Scenario: {scenario}\nValue: {item['value']}\nPrecept: {item['precept']}\n[REASONING] {item['reasoning']} [/REASONING]"
                    new_item["score"] = 0 if force_binary else "0"
                    augmented_data_task1.append((scenario, new_item))

            # (iii) Scramble across scenarios and within tuples
            all_values = [item['value'] for _, item in all_task1_data]
            all_precepts = [item['precept'] for _, item in all_task1_data]
            all_reasonings = [item['reasoning'] for _, item in all_task1_data]
            for scenario in processed_data.keys():
                if scenario not in train_data_scenarios:
                    continue  # preserve test set
                for _ in range(len(all_task1_data)):
                    value = random.choice(all_values)
                    precept = random.choice(all_precepts)
                    reasoning = random.choice(all_reasonings)
                    new_item = deepcopy(random.choice([item for _, item in all_task1_data]))
                    if value == new_item['value'] and precept == new_item['precept'] and reasoning == new_item['reasoning']:
                        continue
                    new_item['text'] = f"Scenario: {scenario}\nValue: {value}\nPrecept: {precept}\n[REASONING] {reasoning} [/REASONING]"
                    new_item['value'] = value
                    new_item['precept'] = precept
                    new_item['reasoning'] = reasoning
                    new_item["score"] = 0 if force_binary else "0"
                    augmented_data_task1.append((scenario, new_item))

            len_data2keep = min(len(augmented_data_task1), int(len(all_task1_data) * augment_ratio * 2))
            augmented_data_task1 = random.choices(augmented_data_task1, k=len_data2keep)
            for scenario, item in augmented_data_task1:
                augmented_data[scenario].append(item)

        # Task 2 augmentation
        if task == 2 or task == 0:
            augmented_data_task2 = []
            for scenario, items in processed_data.items():
                if scenario not in train_data_scenarios:
                    continue  # preserve test set
                task2_items = [item for item in items if item['task'] == 2]

                # Group items by action and consequence
                action_consequence_groups = defaultdict(list)
                for item in task2_items:
                    action_consequence_groups[(item['action'], item['consequence'])].append(item)

                # (i) Scramble precept and reasoning within scenario
                for group in action_consequence_groups.values():
                    reasonings = [item['reasoning'] for item in group]
                    random.shuffle(reasonings)  # Scramble reasonings

                    for item, new_reasoning in zip(group, reasonings):
                        if item['reasoning'] != new_reasoning:  # Avoid identical matches
                            new_item = deepcopy(item)
                            new_item['text'] = f"Scenario: {scenario}\nAction: {item['action']}\nConsequence: {item['consequence']}\nPrecept: {item['precept']}\n[REASONING] {new_reasoning} [/REASONING]"
                            new_item['reasoning'] = new_reasoning
                            new_item["score"] = 0 if force_binary else "0"
                            augmented_data_task2.append((scenario, new_item))

                # (ii) Scramble precept-reasoning pairs across action-consequence pairs within the scenario
                all_precept_reasoning_pairs = [(item['precept'], item['reasoning']) for item in task2_items]
                random.shuffle(all_precept_reasoning_pairs)

                for (action, consequence), group in action_consequence_groups.items():
                    for item, (new_precept, new_reasoning) in zip(group, all_precept_reasoning_pairs):
                        if (item['precept'], item['reasoning']) != (new_precept, new_reasoning):  # Avoid identical matches
                            new_item = deepcopy(item)
                            new_item['text'] = f"Scenario: {scenario}\nAction: {action}\nConsequence: {consequence}\nPrecept: {new_precept}\n[REASONING] {new_reasoning} [/REASONING]"
                            new_item['precept'] = new_precept
                            new_item['reasoning'] = new_reasoning
                            new_item["score"] = 0 if force_binary else "0"
                            augmented_data_task2.append((scenario, new_item))
            len_data2keep = min(len(augmented_data_task2), int(len([item for items in processed_data.values() for item in items if item['task'] == 2]) * augment_ratio * 2))
            augmented_data_task2 = random.choices(augmented_data_task2, k=len_data2keep)
            for scenario, items in augmented_data_task2:
                augmented_data[scenario].append(items)

        # Combine original and augmented data
        for scenario in processed_data.keys():
            train_data.extend(augmented_data[scenario])

    return Dataset.from_dict({
        'text': [item['text'] for item in train_data],
        'label': [item['label'] for item in train_data],
        'task': [item['task'] for item in train_data]
    }), Dataset.from_dict({
        'text': [item['text'] for item in test_data],
        'label': [item['label'] for item in test_data],
        'task': [item['task'] for item in test_data]
    })


# Tokenize dataset
def tokenize_function(tokenizer, label2id, examples):
    labels = examples["label"]
    tasks = examples["task"]
    labels = [label2id(x, task) for x, task in zip(labels, tasks)]
    return_dict = tokenizer(examples["text"])  # , padding="max_length", max_length=2048, truncation=True)
    return_dict["label"] = labels
    return return_dict


def compute_metrics_base(pred, num_classes):
    # 1) Convert normalized‐float labels → integer class‐IDs
    #    labels_float ∈ {0, 1/(C-1), …, 1}
    labels_float = pred.label_ids
    labels = np.rint(labels_float * (num_classes - 1)).astype(int)

    # 2) Convert your model's predictions → integer class‐IDs
    scores = pred.predictions
    if scores.ndim == 2 and scores.shape[1] == num_classes:
        # usual multiclass probs/logits
        preds = np.argmax(scores, axis=1)
    else:
        # single continuous score in [0,1]
        preds = np.rint(scores * (num_classes - 1)).astype(int)
        preds = preds.clip(0, num_classes - 1)

    # 3) Compute accuracy + macro-averaged precision/recall/F1
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds,
        average='macro',
        zero_division=0
    )

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def prepare_peft(model):
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules="all-linear",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        modules_to_save=None
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def label2id_base(x, task, force_binary):
    if task == 1 and not force_binary:
        return float(x) / 4
    else:
        return float(int(x))


def id2label_base(x, task, force_binary):
    # fit between 0 and 4 for task 1, 0 and 1 for task 2
    if task == 1 and not force_binary:
        return int(float(x) * 4)
    else:
        return int(x)


def main():
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument("--task", type=int, required=True, help="Task number. 1 is precept reasoning, 2 is evaluation reasoning, 0 is both (will force binary).")
    parser.add_argument("--balanced", action="store_true", help="Balance task 1-2 dataset, in case of task 0")
    parser.add_argument("--augment", action="store_true", help="Augment data with more negative examples by mixing values, scenarios, etc.")
    parser.add_argument("--augment-ratio", type=float, default=0.5, help="Augmentation ratio")
    parser.add_argument("--force-binary", action="store_true", help="Force binary classification (for task 1)")
    parser.add_argument("--harder", action="store_true", help="Force binary classification for task 1, but only consider 4s as positive")
    parser.add_argument("--model", default="meta-llama/meta-llama-3-8b-instruct", help="Model name")
    parser.add_argument("--data-path", default="./benchmark/data", help="Location of the data directory")
    parser.add_argument("--output", "-o", default="./out", help="Checkpoints output dir")
    parser.add_argument("--learning-rate", "--lr", type=float, dest="lr", required=True, help="The maximum learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Maximum batch size")
    parser.add_argument("--virtual-batch-size", type=int, default=64, help="Virtual batch size, achieved through gradient accumulation")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--wandb", action="store_true", help="Use W&B")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--debug", action="store_true", help="Debug and use cpu")
    args = parser.parse_args()

    train_data, test_data = load_dataset_from_json(
        path.join(args.data_path, 'final_dataset.json'),
        args.task,
        args.force_binary,
        balanced=args.balanced,
        augment=args.augment,
        augment_ratio=args.augment_ratio,
        harder=args.harder
    )

    label2id = partial(label2id_base, force_binary=args.force_binary)
    # used for testing when running without force binary to get eval for binary task 1
    if not args.force_binary:
        label2id_other = partial(label2id_base, force_binary=True)
    id2label = partial(id2label_base, force_binary=args.force_binary)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # attention = "flash_attention_2" if not args.debug and "phi" in args.model.lower() else "eager"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        use_cache=False,
        # attn_implementation=attention,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        num_labels=1,
        # id2label=id2label,
        # label2id=label2id
    )
    model = prepare_peft(model)

    if "phi" in args.model.lower():
        tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right'
    elif "llama" in args.model.lower():
        tokenizer.pad_token_id = 0
        model.config.pad_token_id = 0
    else:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    train_data = train_data.map(lambda x: {"text": tokenizer.apply_chat_template(
        [
            {"role": "system", "content": get_task_prompt(x["task"])},
            {"role": "user", "content": x["text"]}
        ], tokenize=False, add_generation_prompt=False)}
    )
    test_data = test_data.map(lambda x: {"text": tokenizer.apply_chat_template(
        [
            {"role": "system", "content": get_task_prompt(x["task"])},
            {"role": "user", "content": x["text"]}
        ], tokenize=False, add_generation_prompt=False)}
    )

    train_data = train_data.map(partial(tokenize_function, tokenizer, label2id), batched=True, load_from_cache_file=False)
    test_data = test_data.map(partial(tokenize_function, tokenizer, label2id), batched=True, load_from_cache_file=False)
    if not args.force_binary:
        test_data_other = deepcopy(test_data)  # if force binary, also test not binary and vice versa
        test_data_other = test_data_other.map(partial(tokenize_function, tokenizer, label2id_other), batched=True, load_from_cache_file=False)
    # cast label to float (for some reason it is int64)
    train_data = train_data.cast_column("label", Value(dtype="float32"))
    test_data = test_data.cast_column("label", Value(dtype="float32"))

    training_args = TrainingArguments(
        save_steps=10,
        use_cpu=args.debug,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        save_total_limit=5,
        bf16=True,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        load_best_model_at_end=True,
        greater_is_better=True,
        metric_for_best_model="eval_f1",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 32,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.virtual_batch_size // args.batch_size,
        warmup_steps=100,
        logging_steps=10,
        weight_decay=0.01,
        eval_strategy=IntervalStrategy.STEPS,
        eval_steps=10,
        report_to="wandb" if args.wandb else "none",
        run_name=environ.get("RUN_NAME", "classification-test")
    )

    compute_metrics = partial(compute_metrics_base, num_classes=5 if args.task == 1 and not args.force_binary else 2)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    trainer.train(resume_from_checkpoint=args.resume)
    if args.task == 0:
        test_task1 = test_data.filter(lambda x: x['task'] == 1)
        test_task2 = test_data.filter(lambda x: x['task'] == 2)
        trainer.compute_metrics = partial(compute_metrics_base, num_classes=2 if args.force_binary else 5)
        eval_results = trainer.evaluate(test_task1)
        print(f"\n\nTask 1 (binary is {args.force_binary}) results: ", eval_results, "\n\n")
        if not args.force_binary:
            test_task1_other = test_data_other.filter(lambda x: x['task'] == 1)
            trainer.compute_metrics = partial(compute_metrics_base, num_classes=5 if args.force_binary else 2)
            eval_results = trainer.evaluate(test_task1_other)
            print(f"\n\nTask 1 (binary is {not args.force_binary}) results: ", eval_results, "\n\n")
        trainer.compute_metrics = partial(compute_metrics_base, num_classes=2)
        eval_results = trainer.evaluate(test_task2)
        print("\n\nTask 2 results: ", eval_results, "\n\n")
    else:
        eval_results = trainer.evaluate()
        print(eval_results)

    model.save_pretrained(args.output)


def get_task_prompt(task):
    prompts = {
        1: "You are a moral expert. You will be given a scenario and a precept. You will be given a [REASONING] reasoning [/REASONING] that from the scenario extracts a precept. You need to evaluate the quality of the reasoning.",
        2: "You are a moral expert. You will be given a scenario, an action, a consequence, and a precept. You will be given a [REASONING] reasoning [/REASONING] that establishes if the action/consequence satisfy or contradict the reasoning. You need to evaluate the quality of the reasoning."
    }
    if task not in prompts:
        raise ValueError("Invalid task number")
    return prompts[task]


if __name__ == "__main__":
    main()
