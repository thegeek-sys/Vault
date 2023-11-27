---
Created: 2023-11-15
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
A differenza della programmazione strutturata, in cui i dati e le funzioni sono separati. Nella programmazione **object oriented** i dati diventano un attributo dell’oggetto e le funzioni diventano i metodi associati all’oggetto (i dati e le funzioni sono incapsulati).
Invocare un metodo su un oggetto provoca dei side effect (cambiamenti) di stato ai suoi attributi se mutabile, altrimenti crea un nuovo oggetto in uscita dal metodo (se immutabile). Ogni volta che infatti usiamo un punto in python stiamo accedendo ai metodi o agli attributi di un oggetto
Questo tipo di programmazione mi permette di tutelare la riusabilità del codice.

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

![[employee 1.png|300]]
![[employee 2.png|300]]

Le classi forniscono un mezzo per costruire nuove «strutture dati» dove i dati e le funzioni «vanno insieme» (incapsulamento). Progettare una nuova classe definisce un nuovo tipo di oggetto. Ci fornisce la possibilità di creare nuove istanze di quel tipo (oggetto è l’istanza di una classe)

![[color class.png]]

```python
class Color:
	'''
	Definisco una classe che descrive attributo e metodo per definire e
	modificare un pixel
	'''

	def __init__(self, r, g, b):
	'''
	__init__ -> specifica cosa fare nel’ inizializzazione di un oggetto. In
				questo caso lo crea e memorizza r, g, b.
	self     -> va aggiunto nei metodi di una classe. Si riferisce all’oggetto
				istanza che verrà creato successivamente quando facciamo
				c1 = Color(0,0,0).
	'''
	self._r = r
	self._g = g
	self._b = b

	def __repr__(self):
	'''
	questo è il metodo che riscriviamo per mostrare la rappresentazione a video
	dell'oggetto quando lo sampiamo, es. print()
	'''
	return f'Color ({self._r}, {self._g}, {self._b})'

c1 = Color(55, 200, 128)
print(c1._r) # -> 55
```

### Ereditarietà


---
## Oggetti mutabili e immutabili
Ogni cosa in python in realtà è un oggetto e questi possono essere di tipo mutabile (liste, dizionari, set) o immutabile (numeri, stringhe tuple). Un qualcosa per essere un oggetto deve possedere:
- **identità** (puntatore alla memoria)
- **tipo**
- **valore** (anche le funzioni stesse sono oggetti)

### Passaggio per riferimento (immutabile)
In Python, i parametri alle funzioni sono passati per riferimento (una sorta di puntatore). La funzione vede in ingresso un riferimento alla variabile passata (non una copia)

### Passaggio per riferimento (mutabile)
In Python, i parametri alle funzioni sono passati per riferimento. La funzione vede in ingresso una riferimento alla variabile passata (non una copia)

### Side effect
La mutabilità degli oggetti causano
queste «insidie». In python, gli argomenti di default (`target=[]`) sono valutati una volta sola quando la funziona è definita come se fossero una proprietà della
funzione.
> [!WARNING]
> NON vengono valutati ogni volta che la funzione è eseguita

```python
def add_to(num, target=[]):
	target.append(num)
	return target

add_to(1) # -> [1]
add_to(2) # -> [1, 2]
add_to(3) # -> [1, 2, 3]
```

