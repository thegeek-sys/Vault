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



