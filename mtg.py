import nltk
import random
from collections import defaultdict, Counter

# Stupid Backoff parameter
ALPHA = 0.4

def build_ngrams(corpus, n):
    """
    Builds frequency distributions for n-grams up to the specified size.
    Returns a dictionary of Counters for each n.
    ngrams[k][context] = Counter of next words
    """
    ngrams = defaultdict(lambda: defaultdict(Counter))
    
    # Unigrams (n=1): treat as context = empty tuple
    for word in corpus:
        ngrams[1][()][word] += 1
    
    # Higher-order n-grams (n=2,3,...)
    for i in range(2, n + 1):
        for j in range(len(corpus) - i + 1):
            context = tuple(corpus[j : j + i - 1])
            next_word = corpus[j + i - 1]
            ngrams[i][context][next_word] += 1
            
    return ngrams

def get_next_word_probabilities(ngrams, context, n):
    """
    Calculates probabilities for the next word using stupid backoff.
    """
    context_len = len(context)
    
    # Try longest possible context first
    if context_len > 0 and context in ngrams[context_len + 1]:
        next_word_counts = ngrams[context_len + 1][context]
        total_count = sum(next_word_counts.values())
        return {word: count / total_count for word, count in next_word_counts.items()}
    
    # Backoff to a shorter context
    if context_len > 1:
        shorter_context = context[1:]
        prob_dict = get_next_word_probabilities(ngrams, shorter_context, n)
        return {word: prob * ALPHA for word, prob in prob_dict.items()}
    
    # Base case: unigram distribution
    unigram_counts = ngrams[1][()]
    total_words = sum(unigram_counts.values())
    return {word: (count / total_words) * ALPHA for word, count in unigram_counts.items()}
    
def finish_sentence(sentence, n, corpus, randomize=False):
    """
    Extends a sentence using a Markov text generator with stupid backoff.
    Input: list of words
    Output: list of words
    """
    extended_sentence = list(sentence)  # ensure it's a list
    ngrams = build_ngrams(corpus, n)
    
    while len(extended_sentence) < 10:
        # Stop if sentence-ending punctuation
        if extended_sentence[-1] in ['.', '?', '!']:
            break
            
        # Get the context for prediction
        context_length = min(len(extended_sentence), n - 1)  # ✅ FIXED
        context = tuple(extended_sentence[-context_length:])
        
        # Get the probabilities for the next word
        next_word_probs = get_next_word_probabilities(ngrams, context, n)
        
        if not next_word_probs:
            break  # no prediction available
            
        if not randomize:
            # Deterministic: choose most probable word (alphabetical tie-breaker)
            sorted_words = sorted(next_word_probs.keys())
            next_word = max(sorted_words, key=lambda word: next_word_probs[word])
        else:
            # Stochastic: sample by probability
            words = list(next_word_probs.keys())
            weights = list(next_word_probs.values())
            next_word = random.choices(words, weights=weights, k=1)[0]
            
        extended_sentence.append(next_word)
        
    return extended_sentence

