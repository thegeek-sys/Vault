---
Created: 2023-10-10
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Index

1. [[#Introduction|Introduction]]
2. [[#Casi d’uso delle `tuple`|Casi d’uso delle `tuple`]]
---
## Introduction
Il tipo `tuple` in Python che ci permette di memorizzare più elementi in una singola variabile.
I loro valori possono essere di qualsiasi tipo e sono indicizzate tramite interi, supportano dunque l’operatore `[:]` e `len`.

```python
t = 1, 0, 0
print(t,type(t)) # -> (1, 0, 0) <class 'tuple'>

t = tuple('Python')
print(t) # -> ('P', 'y', 't', 'h', 'o', 'n')
print(t[1]) # -> 'y'
```

Anche se sono rappresentabili solo con la virgola, è uso rappresentarle **racchiuse tra parentesi tonde**. Possiamo inoltre “spacchettare” delle tuple e assegnare un valore ad ogni valore della tupla

```python
t = ('mario', 'rossi', 2108912, 28)
nome, cognome, matricola, voto = t
print(nome) # -> 'mario'
```

Gli operatori di confronto con le tuple iniziano a confrontare il primo elemento di ciascuna sequenza:
- Se sono uguali passa all’elemento successivo
- Se elementi di diversi li confronta e restituisce immediatamente il risultato
Mentre il `+` le concatena

```python
(0, 1, 200) < (0, 3, 4)  # -> True
# 0, 0 ignorato
# 1 < 3 True
# 200, 4 ignorato

(2,)+(3,) # -> (2, 3)
```

---
## Casi d’uso delle `tuple`
Tramite l’utilizzo delle tuple posso **assegnare contemporaneamente più valori** a diverse variabili

```python
nome, cognome, matricola, voto = 'mario', 'rossi', 2108912, 28
```

Oppure vengono utilizzare per **ritornare più valori da un funzione**

```python
def div_mod(a,b):
	q = a//b
	r = a%b
	return q, r

quot, resto = div_mod(10,5)
print(quot,resto) # -> 2 0

tup_qr = div_mod(7,2)
type(tup_qr) # -> <class 'tuple'>
print(tup_qr[0], tup_qr[1]) # -> 3 1
```
