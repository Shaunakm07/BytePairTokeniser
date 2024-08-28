import regex as re 

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
        occurences = {}
        for i in range(len(k) - 1):
            pairs = (k[i], k[i + 1])
            count = occurences.get(pairs, 0)
            count += 1
            occurences[pairs] = count

        for j in occurences.keys():
            count = occurences[j]
            count = count * word_count[k]
            
            pairs_count = byte_pairs.get(j, 0)
            pairs_count += count
            byte_pairs[j] = pairs_count
    
    return byte_pairs

def find_lexically_highest(byte_pairs):
    max_count = max([v for k,v in byte_pairs.items()])
    max_pairs = []
    
    for k,v in byte_pairs.items():
        if v == max_count:
            max_pairs.append(k)

    heighest_lexical = max(max_pairs)
    heighest_lexical_string = "".join(heighest_lexical)
    
    return heighest_lexical_string, heighest_lexical

def merge_word_count(word_counts, heighest_lexical):
    new_word_counts = {}
    for k,v in word_counts.items():
        k = list(k)
        k_string = "".join(k)
        if heighest_lexical in k_string:
            idx = k_string.find(heighest_lexical)
            k.pop(idx)
            k[idx] = heighest_lexical
            new_word_counts[tuple(k)] = v
        else:
            new_word_counts[tuple(k)] = v
    return new_word_counts
        
def train_tokeniser(file, vocab_size, special_tokens):
    f = open(file, "r")
    text = f.read()

    merges = []

    pre_tokenised = pre_tokenise(text)
    corpus = create_corpus(special_tokens)
    word_counts = get_word_counts(pre_tokenised)
    while len(corpus) < vocab_size:
        byte_pairs = find_byte_pairs(word_counts)

        heighest_lexical, merge = find_lexically_highest(byte_pairs)
        merges.append(merge)
        corpus.append(heighest_lexical)
    
        merged = merge_word_count(word_counts, heighest_lexical)
        word_counts = merged
    return merged, merges
