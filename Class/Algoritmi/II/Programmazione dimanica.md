---
Created: 2025-04-25
Class: "[[Algoritmi]]"
Related:
---
---
## Dal divide et impera alla programmazione dinamica
Sappiamo che gli algoritmi basati sulla tecnica del divide et impera seguono i 3 passi di questo schema:
1. dividi il problema in sottoproblemi di taglia inferiore
2. risolvi (ricorsivamente) i sottoproblemi di taglia inferiore
3. combina le soluzioni dei sottoproblemi in una soluzione del problema originale

Negli esempi finora visti i sottoproblemi che si ottenevano dall’applicazione del passo $1$ erano tutti diversi, pertanto ciascuno di essi veniva individualmente risolto dalla relativa chiamata ricorsiva del passo $2$. In molte situazioni i sottoproblemi ottenuti al passo $1$ possono risultare uguali. In tal caso, l’algoritmo basato sulla tecnica del divide et impera risolve lo stesso problema più volte svolgendo lavoro inutile

>[!example]
>La sequenza $f_{0},f_{1},f_{2},\dots$ dei numeri di Fibonacci è definita dall’equazione di ricorrenza:
>$$f_{i}=f_{i-1}+f_{i-2}\qquad \text{con }f_{0}=f_{1}=1$$
>
>Il primo algoritmo che viene in mente per calcolare l’$n$-esimo numero di Fibonacci è basato sul divide et impera e sfrutta la definizione stessa di numero di Fibonacci
>
>```python
>def Fib(n):
>	if n<=1: return 1
>	a = Fib(n-1)
>	b = Fib(n-2)
>	return a+b
>```
>
>La relazione di ricorrenza per il tempo di calcolo dell’algoritmo  è:
>$$T(n)=T(n-1)+T(n-2)+O(1)\Longrightarrow T(n)\geq 2T(n-2)+O(1)$$
>risolvendo tramite il metodo iterativo otteniamo:
>$$T(n)\geq \Theta(2^{n/2})$$
>
>Il motivo di tale inefficienza sta nel fatto che il programma viene chiamato sullo stesso input molte volte (ridondante)
>![[Pasted image 20250425224006.png]]

Individuata la causa dell’inefficienza dell’algoritmo è facile individuare la cura. Basta memorizzare in una lista i valori $fib(i)$ quando li si calcola la prima volta cosicché nelle future chiamate ricorsive a $fib(i)$ non ci sarà più bisogno di ricalcolarli, ma potranno essere ricavati dalla lista. Questa tecnica prende il nome di **memoizzazione**

Si risparmia così tempo di calcolo al costo di un piccolo incremento di occupazione di memoria

```python
def Fib(n):
	F = [-1]*(n+1)
	return memFib(n, F)

def memFib(n, F):
	if n<=1:
		return 1
	if F[n] == -1:
		a = memFib(n-1, F)
		b = memFib(n-2, F)
		F[n] = a+b
	return F[n]
```
La novità di questo secondo algoritmo è che esso, prima di attivare la ricorsione per il calcolo di qualche $f_{i}$, con $i<n$, controlla se quel valore è stato calcolato precedentemente e posto in $F[i]$. In caso affermativo la ricorsione non viene effettuata ma viene restituito direttamente il valore $F[i]$
In questo modo l’algoritmo effettuerà esattamente $n$ chiamate ricorsive (una sola  chiamata per il calcolo di ogni $f_{i}$ con $i<n$)

Tenendo conto che ogni chiamata ricorsiva costa $O(1)$ il tempo di calcolo di `Fib` è $O(n)$, un miglioramento esponenziale rispetto alla versione da cui eravamo partiti

A questo punto è ormai semplice eliminare la ricorsione:
```python
def Fib2(n):
	F=[-1]*(n+1)
	F[0] = F[1] = 1
	for i in range(2, n+1):
		F[i] = F[i-2]+F[i-1]
	return F[n]
```
La complessità asintotica rimane $\Theta(n)$ ma abbiamo un risparmio di tempo e spazio (per la gestione della ricorsione)

E’ possibile inoltre ridurre utilizzare complessità di spazio $O(1)$ (mantenendo la complessità di tempo $\Theta(n)$), ci è infatti necessario mantenere solo gli ultimi due valori calcolati
```python
def Fib3(n):
	if n<=1:
		return n
	a=b=1
	for i in range(2, n+1):
		a,b = b, a+b
	return b
```