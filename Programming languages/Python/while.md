---
Created: 2023-10-12
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
Il ciclo while si utilizza quando non sappiamo quante volte iterare ma possiamo testare la condizione di fine iterazione.
La sua sintassi è:

```python
while <condizione_booleana>:
	pass
```

---
## Esempi
Assumiamo di avere una lista che contiene interi ma ogni tanto contiene -100 che vogliamo togliere, non sappiamo a priori quanti -100 ci sono

```python
numbers = [5, -100, 4, 3, -100, -100, -100, 1, -2, 99, 0]

for x in numbers:
	if x == -100:
		numbers.remove(x)
## !ATTENZIONE! ##
##  non funziona ##

# OPPURE

while -100 in numbers
	numbers.remove(-100)
```

> [!WARNING]
> Attenzione con gli indici `IndexError` quando si manipola una struttura dati soprattutto quando: si itera si una struttura dati e allo stesso tempo si rimuove/cancella elementi nella struttura. Mi si potrebbe infatti modificare la lista durante l’iterazione stessa

Rimuovere dalla lista le stringhe che contengono la stringa `'th'`
```python
stringhe  = ['python','th','thhe','the','thee','ttthtt','aatata','th','pippoth','the show','h','t','t    h']
out = []

''' soluzione 1 '''
# questa soluzione non modifica distruttivamente
# la stringa ma ne crea una nuova
for x in stringhe:
	if 'th' not in x:
		out.append(x)
stringhe[:] = out
print(stringhe)


''' soluzione 2 '''
for i in range(len(stringhe)-1,-1,-1):
	if 'th' in stringhe[i]:
		stringhe.pop(i)
print(stringhe)


''' soluzione 3 '''
while i < len(stringhe): 
# il controllo del while con len() è eseguito ogni volta
    if 'th' in stringhe[i]: 
        stringhe.pop(i)
        # tolgo elemento i-esimo; gli elementi
        # a dx di i-th shiftano tutti quindi in
        # questo caso NON devo incrementare i
    else:
        i+=1
		# se non ho tolto, posso andare avanti
```
