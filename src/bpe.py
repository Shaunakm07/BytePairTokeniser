import regex as re
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_tokeniser(input_file, vocab_size, special_tokens):
    pre_tokensied = re.findall(PAT, open(input_file).read())
    merges = []
    vocab = special_tokens
    for i in range(256):
        vocab.append(chr(i).encode("utf=8"))
    
    word_counts = {}

    for w in pre_tokensied:
        w = tuple(c.encode() for c in (list(w)))
        word_counts[w] = word_counts.get(w, 0) + 1

    byte_pairs = {} 
    for k,v in word_counts.items():
        for k1, k2 in zip(k, k[1:]):
            pairs = (k1,k2)
            byte_pairs[pairs] = byte_pairs.get(pairs, 0) + v
    while len(vocab) < vocab_size:


        max_val = max([v for k,v in byte_pairs.items()])
        max_list = []
        for k,v in byte_pairs.items():
            if v == max_val:
                max_list.append(k)
        heighest_lexical = max(max_list)
        merges.append(heighest_lexical)
        joined = heighest_lexical[0] + heighest_lexical[1]
        vocab.append(joined)

        new_word_counts = {}
        new_byte_pairs = {}
        
        for k,v in word_counts.items():
            word = list(k)
            i  = 0
        
            while i < len(word) - 1:
                pairs = (word[i], word[i + 1])
                if pairs == heighest_lexical:
                    if i > 0:
                        prev_pairs = (word[i - 1], word[i])
                        new_byte_pairs[prev_pairs] = new_byte_pairs.get(prev_pairs, byte_pairs[prev_pairs]) - v
                        new_byte_pairs[(word[i - 1], joined)] = new_byte_pairs.get((word[i - 1], joined),0) + v
                    
                    if i < len(word) - 2:
                        next_pairs = (word[i + 1], word[i + 2])
                        new_byte_pairs[next_pairs] = new_byte_pairs.get(next_pairs, byte_pairs[next_pairs]) - v
                        new_byte_pairs[(joined, word[i + 2])] = new_byte_pairs.get((joined, word[i + 2]),0) + v

                    word[i] = joined
                    word.pop(i + 1)
                    i += 1
                else:
                    new_byte_pairs[pairs] = new_byte_pairs.get(pairs, 0) + v
                i+= 1
            new_word_counts[tuple(word)] = v

        word_counts = new_word_counts
        byte_pairs = new_byte_pairs

    return {i:s for i,s in enumerate(vocab)}, merges


vocab, merges = train_tokeniser("text.txt", 500, ["<token>"])
print(merges)