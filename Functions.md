---
Created: 2023-10-05
Programming language: "[[Python]]"
Related: 
Completed:
---
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