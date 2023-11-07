---
Created: 2023-10-30
Programming language: "[[Python]]"
Related:
  - "[[list]]"
Completed:
---
---
## Introduction
Quello delle **sliding window** è un problema che è bene saper risolvere in quanto è molto frequente ma anche molto versatile infatti si può applicare anche ad immagini a strutture bidimensionali

```python
'''
corpo pipppippopipipipipippppppippo
1)    pippo
2)     pippo
3)      pippo
4)       pippo
5)        pippo
'''

query = 'pippo'
corpo = 'pipppippopipipipipippppppippo'

def count_sub_string(query, corpo):
    offset, count = len(query), 0
    # mi fermo non appena query inizia
    # a sbordare dal corpo. Mi serve il +1
    # perche' altrimenti non controllo ultimo
    # elemento quando faccio slicing [start:end]
    N = len(corpo) - offset + 1  # <---- COSA DA RICORDARE
    for i in range(N):
        # mi sposto su corpo da i fino a i+offset escluso
        if query == corpo[i:i+offset]: # <---- COSA DA RICORDARE
               count += 1
    return count
```