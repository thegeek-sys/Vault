---
Created: 2023-11-15
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
A differenza della programmazione strutturata, in cui i dati e le funzioni sono separati. Nella programmazione **object oriented** i dati diventano un attributo dell’oggetto e le funzioni diventano i metodi associati all’oggetto (i dati e le funzioni sono incapsulati). Invocare un metodo su un oggetto provoca dei side effect (cambiamenti) di stato ai suoi attributi se mutabile, altrimenti crea un nuovo oggetto in uscita dal metodo (se immutabile).

![[oop int.png]]
```python
list.append(L,4)    L.append(5)
# obj oriented      # strutturale

''' programmazione strutturale '''
a = 1
b = 100
def somma(x, y):
	return x + y
somma(a,b) # -> 101

''' programmazione ad oggetti '''
a = int(1)
b = int(100)
int.__add__(a,b) # -> 101
# __add__ è un metodo della classe int a cui passo i parametri a e b
```

---
## Classi e Oggetti

![[Screenshot 2023-11-15 at 22-46-33 5) Oggetti Iteratori e Generatori Analisi testi.png|300]]
![[Screenshot 2023-11-15 at 22-47-21 5) Oggetti Iteratori e Generatori Analisi testi.png|300]]

Le classi forniscono un mezzo per costruire nuove «strutture dati» dove i dati e le funzioni «vanno insieme» (incapsulamento). Progettare una nuova classe definisce un nuovo tipo di oggetto. Ci fornisce la possibilità di creare nuove istanze di quel tipo (oggetto è l’istanza di una classe)