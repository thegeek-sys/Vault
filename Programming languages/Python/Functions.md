---
Created: 2023-10-05
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Index

1. [[#Introduction|Introduction]]
1. [[#Funzione ricorsiva|Funzione ricorsiva]]
1. [[#Funzioni come parametri|Funzioni come parametri]]

---
## Introduction
Le **funzioni** in Python sono dei particolari costrutti sintattici che ci permettono di raggruppare, all'interno di un programma, una sequenza di istruzioni in un unico blocco, espletando così una specifica operazione. La loro sintassi generale:

```python
def nome_della_funzione(parametri_formali_ingresso):   # INGRESSO
    # corpo della funzione
    # inserisci qui il codice che vuoi che la funzione 
    # svolga processando i parametri_formali_ingresso
    return calcoli_effettuati                           # USCITA
```

Il comando `return` ci permette di restituire al chiamante (invocazione della funzione) la variabile di output che è stata generata; facendo direttamente `print`, questa ritornerà l'output solamente a video, senza la possibilità di istanziare alcuna variabile all'interno del programma.

```python
# creo una funzione tale che dato un numero di row e col, questa mi genererà una tabella in MarkDown Syntax
def crate_table(row, col):
	hed = "| hed "*col+'\n'
	sep = "| --- "*col+'\n'
	val = ("| val "*col+'\n')*row
	table = hed+sep+val
	return table

final = crate_table(3,8)
print(final)

# iPython
%whos # -> create_table	function
```

In questo modo inoltre non verranno memorizzare le variabili all'interno della funzione e l'unica variabile memorizzata sarà la funzione stessa. Mentre esegue passo passo la funzione verranno memorizzate le variabili dichiarate, per poi essere cancellate non appena si uscirà dalla funzione

---
## Funzione ricorsiva
E’ definita **funzione ricorsiva** una funzione che si invoca all’interno della stessa (concetto che ha a che fare con i metodi induttivi di matematica)
Ne è un chiaro esempio il fattoriale:
$$
n! = \prod_{i=1}^n
$$
$$
5! = 5*4! = 5*4*3! = 5*4*3*2! = 5*4*3*2*1
$$
```python
def fact(n):
	if n == 1
		return 1
	return fact(n-1)*n

fact(5) # -> 120
```

Quello che avviene in realtà è:
$$
fact(n) = n * fact(n-1)
$$
$$
fact(n) = n * ((n-1)*fact(n-2))
$$
$$fact(n) = n * ((n-1)*((n-2)*fact(n-3)))$$
$$fact(n) = n * ((n-1)*((n-2)*fact(n-3)))$$

Nel tentativo di mostrare il processo logico a schermo:

```python
def fact(n, level):
    print('=='*level + f'> chiamo fact({n})')
    
    if n == 1:
        print('=='*level + f'> blocco la ricorsione con ({n})')
        return 1
    
    fact_n = fact(n-1, level+1) * n
    
    print('=='*level + f'> rit. da fact({n}), fact_n {fact_n}')
    return fact_n

fact(5, 1)

#stack
==> chiamo fact(5)
====> chiamo fact(4)
======> chiamo fact(3)
========> chiamo fact(2)
==========> chiamo fact(1)
==========> blocco la ricorsione con (1)
========> rit. da fact(2), fact_n 2
======> rit. da fact(3), fact_n 6
====> rit. da fact(4), fact_n 24
==> rit. da fact(5), fact_n 120
120
```

---
## Funzioni come parametri
Eventualmente posso anche passare direttamente una funzione stessa come parametro di una funzione

```python
def statistics(func, L):
	return func(L)

lista = [5,7,43,-87,12]
M = statistics(max, lista)
print(M) # -> 43
m = statistics(min, lista)
print(m) # -> -87
```

---
## Decoratori
Un decoratore è una funzione che modifica le “funzionalità” di una funzione (la potenzia senza modificarne il codice).
Assumiamo infatti di avere una funzione che non possiamo né toccare né modificare, ma vogliamo aggiungere delle funzionalità ad essa. Come facciamo? Il trucco è ricordarsi che possono passare funzioni a funzioni
```python
# decoratore
def before_and_after_decorator(func):
	# senza questa seconda funzione dentro verrà immediatamente esguita
	def my_wrapping_func():
		print('Decorator: before')
		func()
		print('Decorator: after')
	return my_wrapping_func

# funzione da estendere
def my_locked_function():
	print('Main code of the locked function 1')

# il mio decoratore è fatto in modo tale che func() non è deciso a priori e
# dipende da ciò che passiamo alla funzione
dec = before_and_after_decorator(my_locked_function)
dec()
# -> Decorator: before
# -> Main code of the locked function 1
# -> Decorator: after

@before_and_after_decorator
def my_locked_function_two():
	print('Main code of the locked function 2')
```