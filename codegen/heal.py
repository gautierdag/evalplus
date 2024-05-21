import torch
from transformers import AutoTokenizer


class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_sequence = False
        self.idxs = set()


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.longest_sequence_length = 0

    def insert(self, sequence: list, idx: int):
        node = self.root
        node.idxs.add(idx)
        for num in sequence:
            if num not in node.children:
                node.children[num] = TrieNode()
            node = node.children[num]
            node.idxs.add(idx)
        node.is_end_of_sequence = True

        if len(sequence) > self.longest_sequence_length:
            self.longest_sequence_length = len(sequence)

    def starts_with(self, prefix: list) -> bool:
        node = self.root
        for num in prefix:
            if num not in node.children:
                return False
            node = node.children[num]
        return True

    def get_next(self, prefix: list) -> list:
        node = self.root
        for num in prefix:
            if num not in node.children:
                return []
            node = node.children[num]
        return list(node.children.keys())

    def get_idxs(self, prefix: list) -> set:
        node = self.root
        for num in prefix:
            if num not in node.children:
                return set()
            node = node.children[num]
        return node.idxs


def build_token_prefix_map(tokenizer: AutoTokenizer) -> Trie:
    """
    Build a map from token to index using a Trie datastructure
    """
    token_map = Trie()
    for i in range(len(tokenizer)):
        try:
            s = tokenizer.decode([i])
        except:
            print(f"token id {i} not found in tokenizer")
            continue
        token_map.insert(s, i)  # handle duplicate token encodings
    return token_map


def get_start_decoding(
    token_map: Trie, tokenizer: AutoTokenizer, prompt_tokens: list[int]
) -> list[tuple[int, list[int]]]:
    """
    Given encoded tokens, return the index of the start of token healing
    and the list of tokens that match the possible healing tokens.
    This builds the possible healing tokens by taking the longest subsequence
    that has matches, growing iteratively from the end of the prompt
    up to the max token length.

    Returns:
        list of tuples, with the first element being the index of the start of healing
        and the second element being the list of token ids that match the healing token.
    """
    subseq = ""
    matches = [(len(prompt_tokens), list(range(len(tokenizer))))]
    # matches = []
    i = len(prompt_tokens) - 1
    while len(subseq) < token_map.longest_sequence_length and i >= 0:
        subseq = tokenizer.decode(prompt_tokens[i:], skip_special_tokens=True)
        if token_map.starts_with(subseq):
            matches.append((i, list(token_map.get_idxs(prefix=subseq))))

        i -= 1
    # return matches in order of start index
    matches = sorted(matches, key=lambda x: x[0])
    return matches


def token_healing(
    model,
    tokenizer,
    matches,
    encoded,
    sample_constrained=False,
    sample_predictions=False,
) -> str:
    input_ids = torch.tensor([encoded])
    with torch.no_grad():
        outputs = model(input_ids.to(model.device))

    perplexities = []
    decoded_sequences = []
    for healing_window, (start_idx, token_ids) in enumerate(matches):
        input_ids = encoded[:start_idx]
        start_idx_logits = outputs.logits[0, start_idx - 1, :].cpu()

        mask = torch.full(start_idx_logits.shape, float("-inf"))
        mask[token_ids] = 0
        masked_logits = start_idx_logits + mask

        if sample_constrained:
            next_token_id = torch.multinomial(
                torch.softmax(masked_logits, dim=-1), 1
            ).item()

        else:
            # argmax mode
            next_token_id = torch.argmax(masked_logits).item()

        new_sequence = input_ids + [next_token_id]
        decoded_sequence = tokenizer.decode(new_sequence)

        # Calculate perplexity based on probability of next tokens in the sequence
        window_logits = outputs.logits[
            0, start_idx - healing_window - 1 : start_idx, :
        ].cpu()
        target_window_ids = encoded[start_idx - healing_window : start_idx] + [
            next_token_id
        ]
        loss = torch.nn.functional.cross_entropy(
            window_logits, torch.tensor(target_window_ids)
        )
        perplexity = torch.exp(loss).item()
        perplexities.append(perplexity)
        decoded_sequences.append(decoded_sequence)

    probabilities = 1.0 / torch.tensor(perplexities)
    probabilities = probabilities / torch.sum(probabilities)

    if sample_predictions:
        # sample using
        chosen_sequence = decoded_sequences[torch.multinomial(probabilities, 1).item()]
    else:
        # argmax mode
        chosen_sequence = decoded_sequences[torch.argmax(probabilities).item()]

    return chosen_sequence
