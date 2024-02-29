---
Created: 2023-10-23
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Index

1. [[#Introduction|Introduction]]
2. [[#Casi d’uso|Casi d’uso]]
3. [[#Accedere e manipolare un dizionario|Accedere e manipolare un dizionario]]
4. [[#Iterazione su un dizionario|Iterazione su un dizionario]]
5. [[#Metodi|Metodi]]

---
## Introduction
I dizionari sono delle **mappature** o array associativi, posso infatti usare altri tipi diversi da interi come indici **univoci** (detti `keys`) per un dizionario (basta che siano non mutabili).
Dunque i dizionari contengono:
- raccolta di indici, chiamati chiavi
- una raccolta di valori

La sintassi di base di un dizionario è:
```python
nome_dizionario = { <chiave1> : <valore1>, <chiave2> : <valore2>}

my_dict = {'P': 'p',
           'Y': 'y',
           'T': 't',
           'H': 'h',
           'O': 'o',
           'N': 'n', }

print(my_dict) # -> {'P': 'p', 'Y': 'y', 'T': 't', 'H': 'h', 'O': 'o', 'N': 'n'}
```

Possono essere anche definiti attraverso la funzione `dict()` direttamente (che però risulta poco flessibile) oppure tramite una lista di tuple, dove ogni tupla è la coppia `(chiave, valore)` (più flessibile)

```python
a = dict(uno=1, two=2, three=3)
print(a) # -> {'uno': 1, 'two': 2, 'three': 3}

b = dict([('uno',1), ('tre',3), ('quattro',4)])
print(b) # -> {'uno': 1, 'tre': 3, 'quattro': 4}
```

---
## Casi d’uso
- sono molto potenti per memorizzare risultati nel caso li avete già calcolati **(caching)**
- in alcuni casi possono sostituire **if elif molto densi**
- utili per creare istogrammi o applicazioni dove devo **contare la frequenza, le occorrenze**

---
## Accedere e manipolare un dizionario
Per poter accedere ad un dizionario ho la necessità di avere una `key` (se non presente mi verrà restituito un `KeyError`, per controllare la appartenenza di una chiave ad un dizionario uso `in`)

```python
my_dict = {'P': 'p', 'Y': 'y', 'T': 't', 'H': 'h', 'O': 'o', 'N': 'n'}
chiave = 'P'
print(f'{chiave} --> {my_dict[chiave]}') # -> P --> p

for chiave in 'pP':
    if chiave in my_dict:  # NB: controlla la chiave NON il valore
        print(f'{chiave} --> {my_dict[chiave]}') # chiave -> valore
    else:
        print(f'Key {chiave} non è presente')
# -> Key p non è presente
# -> P --> p
```

Per poter aggiungere un elemento al dizionario mi basta definire una nuova coppia chiave → valore

```python
my_dict = {'mille': '1k', 'cento': 100, 'uno': 1, 'zero': '0'}
my_dict['T'] = 3
print(my_dict) # -> {'mille': '1k', 'cento': 100, 'uno': 1, 'zero': '0', 'T': 3}
```

---
## Iterazione su un dizionario
```python
my_dict = {'P': 'p', 'Y': 'y', 'T': 't', 'H': 'h', 'O': 'o', 'N': 'n', 'p': 3}
```

### Per coppia (chiave, valore)
```python
for key, val in my_dict.items():
    print(f'{key} --> {val}')

# P --> p
# Y --> y
# T --> t
# H --> h
# O --> o
# N --> n
# p --> 3
```

### Per chiave
```python
for key in my_dict.keys():
	print(f'{key}', end=' ')

# -> P Y T H O N p
```

### Per valore
```python
for val in my_dict.values():
	print(f'{val}', end=' ')

# -> p y t h o n 3
```

---
## Metodi
![[dict methods]]
