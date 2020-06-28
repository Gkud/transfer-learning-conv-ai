# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings

import torch
import torch.nn.functional as F

from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer, pipeline
from train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from utils import get_dataset, download_pretrained_model

import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
from nltk import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()

from generateRecipe import create_recipe
import random
summarizer = pipeline(task="summarization")

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output

def create_reflection(text):
    reflections = {"i": "you", "am": "are", "was": "were", "my": "your", "mine": "yours", "me": "you",
                   "i ' m": "you are", "'m": "are", "myself": "yourself", "im": "you are", "i ' hv": "you have",
                   "ive": "you have", "you ' m": "you are", "we": "you", "our": "your", "ours": "yours", "us": "you",
                   "your": "my", "yours": "mine"}
    extra_words = ["so, ", "ok, so you think ", "thus you say ", "ok, so you said ", "so you said, ",
                   "so you are saying "]
    words = word_tokenize(text.lower())
    reflected_sent = ' '.join([reflections[word] if word in reflections else word for word in words])
    reflected_sent = random.choice(extra_words) + reflected_sent
    return reflected_sent

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="openai-gpt", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")

    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--conv_limit", type=int, default=None, help="Length of conversation - number of times Speaker1 can respond")


    args = parser.parse_args()

    #logging.basicConfig(level=logging.INFO)
    #logger = logging.getLogger(__file__)
    #logger.info(pformat(args))




    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)



    print("Select type of chat:\n1. Counselling\n2. Task-Oriented")
    raw_text = input(">>> ")

    initial = ["Will you like to learn a new recipe?", "Do you want to learn a new recipe?", "Let us learn a new recipe."]
    sents = ["To sum up, ", "Thus, as I understand, ", "So, to summarize, "]

    history = []

    if raw_text == "1":
        if args.model_checkpoint == "":
            if args.model == 'gpt2':
                raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
            else:
                args.model_checkpoint = download_pretrained_model()

        #logger.info("Get pretrained model and tokenizer")
        tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
        tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
        model = model_class.from_pretrained(args.model_checkpoint)
        model.to(args.device)
        add_special_tokens_(model, tokenizer)


        #logger.info("Sample a personality")
        dataset = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)
        personalities = [dialog["personality"] for dialog in dataset]
        personality = random.choice(personalities)
        print("Selected personality: ", tokenizer.decode(chain(*personality)))

        if args.conv_limit:
          conv_len = args.conv_limit
        else:
          conv_len = -1

        utt = 0
        text_summary = []
        while utt != conv_len:
            raw_text = input(">>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input(">>> ")
            history.append(tokenizer.encode(raw_text))
            text_summary.append(raw_text)
            with torch.no_grad():
                out_ids = sample_sequence(personality, history, tokenizer, model, args)
            history.append(out_ids)
            history = history[-(2*args.max_history+1):]
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
            print(out_text)
            utt = utt + 1
            if utt == conv_len:
                if out_text.endswith("?"):
                    utt = utt - 1

        # generate emotion
        raw_text = 'exit chat'
        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = sample_sequence(personality, history, tokenizer, model, args)
        history.append(out_ids)
        history = history[-(2*args.max_history+1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print("\n" + "Chat Emotion: " + out_text)

        # generate summary
        text = ".".join(text_summary)
        summary = summarizer(text, max_length=50)
        print("\n" + "Summary:\n" + random.choice(sents) + create_reflection(summary[0]['summary_text']))

        # generate a supporting response to the summary
        raw_text = 'summarize-chat'
        history.append(tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = sample_sequence(personality, history, tokenizer, model, args)
        history.append(out_ids)
        history = history[-(2*args.max_history+1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        print("\n" + "Response:\n" + out_text)

    elif raw_text == "2":
        print(random.choice(initial))
        raw_text = input(">>> ")
        scores = sentiment.polarity_scores(raw_text)
        if scores['pos'] > scores['neg']:
            print("Great, here is a recipe for you ...")
            create_recipe()
            raw_text = input(">>> ")
        elif scores['neg'] > scores['pos']:
            print("ok, then maybe you will like to chat with the counsellor. Please choose option 1. Thank you.")
        else:
            print("I could not understand what you are asking.")

    else:
        print("Please select the correct choice.")

if __name__ == "__main__":
    run()
