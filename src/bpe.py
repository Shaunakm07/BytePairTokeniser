import regex as re 
import time

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"

def pre_tokenise(text):
    return re.findall(PAT, text)

def create_corpus(special_tokens):
    corpus = special_tokens

    for i in range(256):
        char = chr(i)
        corpus.append(char)
    return corpus

def get_word_counts(words):
    word_counts = {}
    for word in words:
        w = tuple([chr(c) for c in list(word.encode("utf-8"))])
        count = word_counts.get(w, 0)
        count += 1
        word_counts[w] = count
    return word_counts

def find_byte_pairs(word_count):
    word_keys = word_count.keys()
    byte_pairs = {}

    for k in word_keys:
        for i in range(len(k) - 1):
            pairs = (k[i], k[i + 1])
            current_count = byte_pairs.get(pairs, 0)
            weight = word_count[k]
            current_count += weight
            byte_pairs[pairs] = current_count
    
    return byte_pairs

def find_lexically_highest(byte_pairs):
    max_count = max([v for k,v in byte_pairs.items()])
    max_pairs = []

    for k,v in byte_pairs.items():
        if v == max_count:
            max_pairs.append(k)

    heighest_lexical = max(max_pairs)
    
    return "".join(heighest_lexical), heighest_lexical

def merge_word_count(byte_pairs, heighest_lexical):
    new_byte_pairs = {}
    for k,v in byte_pairs.items():
        v1, v2 = k
        if v1 == heighest_lexical[1]:
            v1 = heighest_lexical
        elif v2 == heighest_lexical[0]:
            v2 = heighest_lexical
        new_byte_pairs[(v1, v2)] = v
    return new_byte_pairs
        
def train_tokeniser(file, vocab_size, special_tokens):
    f = open(file, "r")
    text = f.read()
    merges = []

    pre_tokenised = pre_tokenise(text)
    corpus = create_corpus(special_tokens)
    word_counts = get_word_counts(pre_tokenised)
    byte_pairs = find_byte_pairs(word_counts)

    while len(corpus) < vocab_size:

        heighest_lexical, merge = find_lexically_highest(byte_pairs)
        merges.append(merge)
        corpus.append(heighest_lexical)
    
        merged = merge_word_count(byte_pairs, heighest_lexical)
        byte_pairs = merged

    return ({i:c for i,c in enumerate(list(set(corpus)))}, merges)


start_time = time.time()
vocab, merges = train_tokeniser("text.txt", 500, [])
print(merges)
end_time = time.time()

print(end_time - start_time)
