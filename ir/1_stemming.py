import re

def stem(word):
    if len(word) < 4:
        return word  # too short to stem safely
    
    # Plurals
    if word.endswith('sses'):
        word = word[:-2]          # passes → pass
    elif word.endswith('ies') and len(word) > 3:
        word = word[:-2] + 'y'    # studies → study
    elif not word.endswith('ss') and word.endswith('s'):
        word = word[:-1]          # cats → cat
    
    # -ed: played → play, worked → work
    elif word.endswith('ed') and len(word) > 3:
        word = word[:-2]
    
    # -ing: running → run, playing → play, making → make
    elif word.endswith('ing') and len(word) > 4:
        word = word[:-3]
        # Double the consonant back: runn → run
        if len(word) >= 2 and word[-1] == word[-2] and word[-1] not in 'aeiou':
            word = word[:-1]
    
    return word

# Test it
words = ['running', 'playing', 'making', 'cats', 'passes', 'studies', 'played', 'quickly', 'beautiful']
for w in words:
    print(f"{w:12} → {stem(w)}")