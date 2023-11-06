---
Created: 2023-10-23
Programming language: "[[Python]]"
Related:
  - "[[dict]]"
Completed:
---
---
#### `d1.update(d2)`
Il metodo `update` mi permette di aggiungere in-place un dizionario ad un altro dizionario (in caso di elementi con stessa chiave questa verrà sostituita)

---
#### `dict.get(key, val_def)`
Il metodo `get` rende il valore mappato della chiave `key` **se esiste la chiave altrimenti rende** `val_def`
```python
stringa = 'supercalifragilistichespiralidoso'

counts_b = {}
for char in stringa:
    counts_b[char] = counts_b.get(char,0) + 1
counts_b
assert counts_b == counts
```

---
## Ricerca binaria
Sapendo che bisogna fare tante ricerche con uno stesso campo, mi conviene spendere un po’ di tempo inizialmente per ordinare il dizionario per poi poter fare una ricerca binaria (decisamente più veloce di una ricerca lineare, quella a cui siamo abituati a fare).
Possiamo interpretare la ricerca binaria come quando cerchiamo la parola in un dizionario. Cercando una parola in un dizionario ordinato ci viene molto più semplice rispetto a cercare in un dizionario disordinato. In Python come concetto possiamo pensare di avere una lista ordinata di numeri da 0 a 50, e di dover cercare il valore 12. Il programma deve dividere la lista in 2 e controllare se il valore di mezzo è maggiore o minore di quello ricercato. In questo caso sarà più piccolo quindi farò lo stesso procedimento sulla porzione di sinistra della lista e continuo così finché il valore non sarà uguale a quello ricercato. Questo in Python ci viene molto semplice da fare con la **ricorsione**
Questa tecnica risulta molto più veloce, infatti mentre la ricerca lineare ha complessità $O(N)$ la ricerca binaria ha complessità $O(N\log_2(N))$ per ordinare il dizionario e $O(N\log_2(N))$ per la ricerca.
```python
def bin_search(L, query):
	if not L:
		return False
	half = len(L)//2
	if L[half] == query:
		print(f'{L[half]}!')
		return True
	elif query < L[half]:
		return bin_search(L[:half], query)
	else:
		return bin_search(L[half+1:], query)

print(bin_search([1,2,3,4,5], 2))
```

Ricerca lineare: 
$$
\begin{matrix}
Q = \left\{\text{numero di query}\right\}\\
N = \left\{\text{numero di record}\right\}\\
O(N ⋅ Q)
\end{matrix}
$$

Ricerca binaria:
$$
\begin{matrix}
O(N\log_{2}(N)+Q\log_{2}(N))= \\
= (N + Q) ⋅ log_{2}(N)
\end{matrix}
$$

---
## Indice invertito
Nel caso in cui ci servisse di cercare per campi ogni volta diversi ci potrebbe tornare utile la ricerca ad indice invertito, che ci permette di iterare ad una velocità prossima a quella lineare ma dovremmo pagare un prezzo a livello spaziale (di memoria).
Per fare questo dovremmo creare una sequenza ordinata di valori, che associa a ciascun valore l’indice che contiene quel valore `{ valore : lista_di_indici }`

```python
def crea_indice_second(agenda, campo):
	indice = {} # questo mappa valore ‐‐> lista_indice
	for idx, record in enumerate(agenda):
		if campo in record:
			# accedo al valore
			valore = record[campo]
			indice[valore] = indice.get(valore,[]) + [idx]
			#.append(idx) NON funziona
	return indice

print(crea_indice_second(agenda,'cognome'))

{'Paolino': [0],
'Paperone': [1],
"de' Paperoni": [2],
'Pitagorico': [3],
'Gambadilegno': [4, 5],
'Mouse': [6, 7],
"de' Pippis": [8]}
```

Quindi creo la funzione che mi restituisca tutti gli output con un determinato cognome:

```python
def records_per_campo(indice, agenda, campo):
	return [ agenda[idx] for idx in indice[campo]]
```