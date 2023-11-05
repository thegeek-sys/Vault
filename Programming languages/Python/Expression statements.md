---
Created: 2023-10-06
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## `pass`
Istruzione che non dà alcun output (Python fa finta che non ci sia nulla ma ci può evitare errori dati dalla sola definizione di una funzione)

---
## `del`

---
## `return`

---
## `import`

---
## `continue`
Salta l’iterazione corrente

---
## `break`
Questa keyword permettere di interrompere un ciclo o l’iterazione corrente quando si verifica una determinata condizione

---
## `assert`
Usato dai programmatori per controllare il loro codice. Se stiamo facendo qualcosa di intricato e
vogliamo essere sicuri che una certa condizione sia vera possiamo inserire `assert` `condizione` per capire se le nostra ipotesi è corretta.
- se `<condizione_che_vogliamo_sia_vera>` e' vera il flusso del programma prosegue
- altrimenti assert rende un `AssertionError`
```python
''' asser_main.py '''
i = 0
assert i > 0, 'Attenzione i è negativo o zero, i = ' + str(i)

python ‐O assert_main.py # per non eseguire gli assert
```