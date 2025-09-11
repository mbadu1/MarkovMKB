import random
import nltk
from collections import defaultdict, Counter

# Discount factor for stupid backoff
ALPHA = 0.4


def build_ngrams(corpus, n):
    """
    Build n-gram frequency models up to order n.
    Returns:
        dict: Returns a nested dictionary: ngram_models[n][context] = Counter of next words.
    """
    ngrams = defaultdict(lambda: defaultdict(Counter))

    # Unigrams (n=1): context = empty tuple
    for word in corpus:
        ngrams[1][()][word] += 1

    # Higher-order n-grams bigrams, trigrams from 2 to n
    for size in range(2, n + 1):
        for i in range(len(corpus) - size + 1):
            context = tuple(corpus[i : i + size - 1])
            next_word = corpus[i + size - 1]
            ngrams[size][context][next_word] += 1

    return ngrams


def get_next_word_probabilities(ngrams, context, n):
    """
    Given a context,
    using stupid backoff smoothing the probabilities for the next word
    Args:
        context (tuple): previous words (up to n-1)
    Returns:
         probability distribution of next words dict[str, float]
    """
    context_len = len(context)

    # first context - longer
    if context_len > 0 and context in ngrams[context_len + 1]:
        counts = ngrams[context_len + 1][context]
        total = sum(counts.values())
        return {word: count / total for word, count in counts.items()}

    # shorter context - backoff
    if context_len > 1:
        shorter_context = context[1:]
        backed_off = get_next_word_probabilities(ngrams, shorter_context, n)
        return {word: prob * ALPHA for word, prob in backed_off.items()}

    # unigram distribution
    unigram_counts = ngrams[1][()]
    total = sum(unigram_counts.values())
    return {word: (count / total) * ALPHA for word, count in unigram_counts.items()}


def finish_sentence(sentence, n, corpus, randomize=False):
    """
    Markov-based text model with backoff.

    Args:
        sentence (list[str]): seed words
        n (int): order of n-gram model
        corpus (list[str]): tokenized training text
        randomize (bool): if True, sample randomly; if False, pick max prob

    Returns:
        list[str]: extended sentence (up to 10 words or until punctuation)
    """
    extended_sentence = list(sentence)  # make sure it's a list
    ngrams = build_ngrams(corpus, n)

    while len(extended_sentence) < 10:
        # Stop if we reach sentence-ending punctuation
        if extended_sentence[-1] in [".", "?", "!"]:
            break

        # Context = last n-1 words
        context_length = min(len(extended_sentence), n - 1)
        context = tuple(extended_sentence[-context_length:])

        # Get probabilities for the next word
        next_word_probs = get_next_word_probabilities(ngrams, context, n)
        if not next_word_probs:
            break

        if not randomize:
            # Deterministic: pick most probable (alphabetical tiebreaker)
            sorted_words = sorted(next_word_probs.keys())
            next_word = max(sorted_words, key=lambda word: next_word_probs[word])
        else:
            # Random: sample by probability
            words = list(next_word_probs.keys())
            weights = list(next_word_probs.values())
            next_word = random.choices(words, weights=weights, k=1)[0]

        extended_sentence.append(next_word)

    return extended_sentence
