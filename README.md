# Markov Text Generator (with Stupid Backoff)

This project implements a simple **n-gram language model** with **stupid
backoff** for text generation. The model is built using the [NLTK
Gutenberg corpus](https://www.nltk.org/book/ch02.html) and can
deterministically or stochastically generate continuations of a given
input phrase.

------------------------------------------------------------------------

## Features

-   Builds **n-gram frequency models** up to order `n`\
-   Uses **stupid backoff** (Jelinek, 1985) for smoothing and handling
    unseen contexts\
-   Supports:
    -   **Deterministic generation**: always picks the most probable
        next word\
    -   **Stochastic generation**: samples the next word based on
        probability distribution

------------------------------------------------------------------------

## How It Works

1.  **`build_ngrams(corpus, n)`**
    -   Constructs unigram, bigram, trigram, ... up to `n`-gram models\
    -   Stores counts in nested `defaultdict(Counter)` structures
2.  **`get_next_word_probabilities(ngrams, context, n)`**
    -   Looks up probabilities for the next word given the current
        context\
    -   If the context is unseen, recursively backs off to a shorter
        context\
    -   At the unigram level, applies scaling factor `ALPHA = 0.4`
3.  **`finish_sentence(sentence, n, corpus, randomize=False)`**
    -   Extends a starting sequence of words up to 10 tokens\
    -   Stops early if it generates punctuation (`.`, `?`, `!`)\
    -   Returns a list of words

------------------------------------------------------------------------

## Example Usage

``` python
import nltk
from mtg import finish_sentence

# Download NLTK resources if not already installed
nltk.download("gutenberg")
nltk.download("punkt")

# Load corpus
corpus = nltk.word_tokenize(
    nltk.corpus.gutenberg.raw("austen-sense.txt").lower()
)

# Deterministic generation
start = ["she", "was", "not"]
result = finish_sentence(start, n=3, corpus=corpus, randomize=False)
print("Deterministic:", " ".join(result))

# Stochastic generation
start2 = ["the", "old", "man"]
result2 = finish_sentence(start2, n=2, corpus=corpus, randomize=True)
print("Stochastic:", " ".join(result2))
```

------------------------------------------------------------------------

## Example Output

    Deterministic: she was not in the world .
    Stochastic: the old man was very happy .

*(Your output may vary depending on `randomize` and corpus size.)*

------------------------------------------------------------------------


