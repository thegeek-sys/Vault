---
Created: 2023-10-05
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
Il tipo `float` (floating point) in Python che ci permette di rappresentare numeri con la virgola.

```python
type(5.5) # -> <class 'float'>
print(float(5)) # -> 5.0
```

Per rappresentare numeri estremamente complessi Python utilizza una particolare compressione che tramite una serie di matrici concatenate mi permette di rappresentare numeri superiori a quello che mi permetterebbe l'architettura del mio pc.
Proprio per questo motivo spesso Python commette spesso errori di approssimazione in caso di numeri troppo piccoli o troppo grandi.

```python
format(1e-20, '.100f') # -> '0.0000000000000000000099999999999999994515...' (1 + 1e-20) > 1 # -> False
```
