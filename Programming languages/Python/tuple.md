---
Created: 2023-10-10
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
Il tipo `tuple` in Python che ci permette di memorizzare più elementi in una singola variabile.
I loro valori possono essere di qualsiasi tipo e sono indicizzate tramite interi. 

> [!INFO]
> Nonostante possono sembrare molto simili a liste, queste sono immutabili
```python
t = 1, 0, 0
print(t,type(t)) # -> (1, 0, 0) <class 'tuple'>
```

Nonostante sono rappresentabili anche solo con la virgola, è uso rappresentarle racchiuse tra parentesi tonde. Possiamo inoltre “spacchettare” delle tuple e assegnare un valore ad ogni valore della tupla

```python
t = ('mario', 'rossi', 2108912, 28)
nome, cognome, matricola, voto = t
print(nome) # -> 'mario'
```