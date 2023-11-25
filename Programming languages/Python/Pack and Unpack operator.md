---
Created: 2023-11-22
Programming language: "[[Python]]"
Related:
  - "[[dict]]"
  - "[[list]]"
Completed:
---
---
## Introduction
Questo tipo di operatore mi permette di assegnare velocemente più variabili su una sola riga
- **nulla** → tupla
- **`*`**  → lista
- **`**`** → dizionario

```python
first_name, middle_name, last_name = 'iacopo', 'lizardking', 'masi'

# sto "impacchettando" i tre valori dentro una stessa lista
*full_name, = 'iacopo', 'lizardking', 'masi'
print(full_name) # ['iacopo', 'lizardking', 'masi']

*full_name, last = 'iacopo', 'lizardking', 'masi'
print(full_name, last) # -> ['iacopo', 'lizardking'] 'masi'
```

---
## Funzioni con argomenti variabili
Possiamo anche definire delle funzioni con argomenti variabili e lo possiamo fare attraverso gli operatori `*` e `**`.
Definendo infatti una funzione che riceve come argomento `**kwargs` questa riceverà un dizionario composto da `key = nome_dell_variabila` e `value = valore passato`

```python
def func(**kwargs):
	for key, value in kwargs.items():
		print(f'{key} -> {value}')

func(opt_1=True, deactivate_log=False, alpha=1.0)

def f(*args):
	pass # impacchetta gli n valori passati dentro la lista args
```

Mentre se chiamo una funzione passandogli `*arg` vorrà dire che la funzione verrà eseguita su ogni singolo valore della lista

```python
L = [1, 2, 3, 4]
print(*L) # -> 1 2 3 4
```

In definitiva l’utilizzo dell’operatore pack o unpack in una funzione:
- **se la stiamo definendo**, vuol dire che i valori passati devono essere impacchettati dentro `*args` o `**kwargs`
- **se la stiamo chiamando**, vuol dire che gli passiamo “al volo” tutti gli elementi contenuti in un iterable (usualmente lista)