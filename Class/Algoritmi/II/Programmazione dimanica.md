---
Created: 2025-04-25
Class: "[[Algoritmi]]"
Related:
---
---
## Index
- [[#Dal divide et impera alla programmazione dinamica|Dal divide et impera alla programmazione dinamica]]
	- [[#Dal divide et impera alla programmazione dinamica#In sintesi|In sintesi]]
- [[#Esercizi|Esercizi]]
- [[#Algoritmi pseudopolinomiali|Algoritmi pseudopolinomiali]]
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

### In sintesi
- siamo partiti da un algoritmo ricorsivo e non efficiente ottenuto applicando la tecnica del divide et impera al problema in esame
- ci siamo accorti che il motivo dell’inefficienza era la presenza di overlapping di sottoproblemi
- abbiamo risolto il problema del ricalcolo di soluzioni allo stesso sottoproblema mediante la tecnica della memoizzazione e quindi ricorrendo a ”tabelle” per conservare i risultati a sottoproblemi già calcolati
- abbiamo sviluppato una versione dell’algoritmo iterativa che ha permesso di sbarazzarsi della ricorsione, permettando una approccio **bottom-up** (la versione ricorsiva usa l’approccio top-down)
- abbiamo ottimizzato lo spazio di memoria mantenendo memorizzata nel corso dell’algoritmo solo la parte della tabella che sarebbe servita nel seguito

---
## Esercizi

>[!question] Vogliamo contare il numero di stringhe binarie lunghe $n$ senza 2 zeri consecutivi
>>[!done]
>>Per questo tipo di esercizi è necessario tendenzialmente precalcolarsi i primi valori, per poi capire il pattern per la costruzione dei successivi
>>![[Pasted image 20250426115415.png|300]]
>>
>>$$T[i]=\text{il numero di stringhe binaria lunghe }i\text{ senza 2 zeri consecutivi}$$
>>
>>Il problema dunque si limita a definire quante stringhe si aggiungono aumentando di un elemento la lunghezza (dando per scontato che tutte le stringhe fino a $i-1$ sono valide)
>>![[Pasted image 20250426114844.png|200]]
>>
>>Se alla posizione $i$ ci sta un $1$, allora aggiungo $T[i-1]$ modi (non ci sono vincoli sui valori precedenti)
>>Se alla posizione $i$ ci sta uno $0$, vuol dire che necessariamente alla posizione $i-1$ ci deve essere un $1$. Aggiungo quindi $T[i-2]$ modi
>>
>>In totale si ha dunque:
>>$$T[i]=T[i-1]+T[i-2]$$
>>e posso iniziare ad applicare la formula a partire da $T[1]$

>[!question] Vogliamo contare il numero di stringhe binarie lunghe $n$ senza 3 zeri consecutivi
>>[!done]
>>Come prima precalcoliamo i primi valori, per poi capire il pattern per la costruzione dei successivi
>>![[Pasted image 20250426123938.png|300]]
>>
>>$$T[i]=\text{il numero di stringhe binaria lunghe }i\text{ senza 3 zeri consecutivi}$$
>>
>>Il problema dunque si limita a definire quante stringhe si aggiungono aumentando di un elemento la lunghezza (dando per scontato che tutte le stringhe fino a $i-1$ sono valide)
>>![[Pasted image 20250426114844.png|200]]
>>
>>Se alla posizione $i$ ci sta un $1$, allora aggiungo $T[i-1]$ modi (non ci sono vincoli sui valori precedenti)
>>Se alla posizione $i$ ci sta uno $0$, è necessario controllare anche il valore precedente ($i-1$):
>>- se ci sta uno $0$, allora $i-2$ deve necessariamente essere un $1$ → aggiungo $T[i-3]$
>>- se ci sta un $1$, allora sulla posizione $i-2$ non ci sono vincoli → aggiungo $T[i-2]$
>>
>>In totale si ha dunque:
>>$$T[i]=T[i-1]+T[i-2]+T[i-3]$$
>>e posso iniziare ad applicare la formula a partire da $T[2]$

>[!question] Abbiamo $n$ ($n\geq 1$) persone da distribuire in un albero con stanze singole o doppie. In quanti modi si possono distribuire le persone?
>>[!done]
>>Come prima precalcoliamo i primi valori, per poi capire il pattern per la costruzione dei successivi
>>![[Pasted image 20250426124538.png|300]]
>>- $n=1$ → $1$
>>- $n=2$ → $2$ ($[[1],[2]],[[1,2]]$)
>>
>>$$T[i]=\text{il numero di modi in cui posso sistemare }i\text{ persone}$$
>>
>>Abbiamo quindi due casi: se la persona aggiunta viene inserita in una camera singola o in una camera doppia
>>Mettendo la persona nella camera singola dovremo aggiungere $T[i-1]$ possibilità. Mettendo invece la persona nella camera doppia avremo $i-1$ modi per poter scegliere il suo compagno di stanza a cui dovremo aggiungere le $T[i-2]$ possibili combinazioni precedenti
>>$$T[i]=\underset{ \text{sing} }{ ? }+\underset{ \text{doppia} }{ ? }=T[i-1]+(i-1)T[i-2]$$
>>
>>Implementazione:
>>```python
>>def es(n):
>>	T=[0]*(n+1)
>>	T[1],T[2] = 1,2
>>	for i in range(3, n+1):
>>		T[i]=T[i-1]+(i-1)T[i-2]
>>```

>[!question] Abbiamo $n$ file di varie dimensioni ciascuna inferiore a $C$ ed un disco di capacità $C$, bisogna trovare il sottoinsieme di file che può essere memorizzato sul disco e che massimizza lo spazio occupato. Progettare un algoritmo che, dati $C$ e la lista $A$, dove $A[i]$ è la dimensione del file $i$, risolva il problema
>>[!done]
>>Non è difficile rendersi conto che gli algoritmi greedy non trovano sempre la soluzione ottima.
>>Tramite la programmazione dinamica è possibile trovare la soluzione ottima in tempo **pseudopolinomiale**
>>
>>Ci è infatti possibile creare una tabella $C\times len(A)$ dove $T[i,j]$ è il massimo spazio che posso occupare con i primi $i$ file in un blocco grande $j$
>>Inoltre, poiché la compilazione della tabella è sequenziale, quando calcolo $T[i,j]$ ho già calcolato tutti i precedenti
>>
>>Per compilare la tabella:
>>$$T[i,j]=\begin{cases}T[i-1,j]&\text{se non prendo} \\A[i]+T[i-1,j-A[i]]&\text{se prendo}\end{cases}$$
>>>[!info]
>>>Quando prendo il file, per calcolare la nuova posizione $T[i,j]$, utilizzo la dimensione del file $i$ più $T$ della riga precedente e colonna corrispondente a $j-A[i]$ (devo avere sufficiente spazio)
>>
>>Quindi:
>>$$T[i,j]=\text{max}\{T[i-1,j],A[i]+T[i-1,j-A[i]]\}$$
>>e la soluzione si troverà nella posizione $T[C,len(A)]$
>>
>>Questa tabella ci permette anche di ricostruire le scelte prese per poter arrivare alla soluzione
>>>[!example] $C=10$ e $A=[1,5,3,4,2,2]$
>>>![[Pasted image 20250427130630.png]]
>>
>>Dunque la complessità risulta essere $\Theta(n\cdot c)$, ma come già detto è pseudopolinomiale. Infatti $C$ potrebbe essere molto più grande di $len(A)$

>[!question] Dato l’intero $n$ vogliamo contare il numero di differenti tassellamenti di una superficie di dimensione $n\times2$ tramite tessere di dominio di dimensione $1\times 2$
>Ad esempio:
>- per $n=1$ la risposta dell’algoritmo deve ovviamente essere $1$
>- per $n=2$ la risposta deve essere $2$ perché sono possibili i soli due tassellamenti seguenti
>![[Pasted image 20250429102731.png|300]]
>- per $n=3$ la risposta deve essere $3$
>![[Pasted image 20250429103612.png|300]]
>L’algoritmo deve avere complessità $O(n)$
>>[!done]
>>Utilizzeremo una tabella monodimensionale di dimensioni $n+1$ e definiamo il contenuto delle celle come segue:
>>$$T[i]=\text{numero di tassellamenti possibili per la superficie di dimensione}\,i\times 2$$
>>
>>Una volta riempita la tabella la soluzione al nostro problema la troveremo nella locazione $T[n]$
>>
>>Resta definire la rgeola ricorsiva con cui calcolare i valori $T[i]$ nella tabella
>>$$T[i]=\begin{cases}1&\text{se}\,i=1 \\2&\text{se}\,i=2 \\T[i-1]+T[i-2]&\text{altrimenti}\end{cases}$$
>>Dove $T[i-1]$ se sto aggiungendo una piastrella orizzontale, e $T[i-2]$ se sto aggiungendo una piastrella verticale (all’$i$-esimo livello ce ne sono $2$ verticali che quindi coprono il livello $i$ e $i-1$)
>>
>>Implementazione:
>> ```python
>>def es(n):
>>	m = max(3,n)
>>	T=[0]*(n+1)
>>	T[1] ,T[2] = 1, 2
>>	for i in range(3, n+1):
>>		T[i] = T[i-1]+T[i-2]
>>	return T[n]
>>```

>[!question] Il problema del massimo sottovettore
>Data una lista $A$ di $n$ interi, vogliamo trovare una sottolsita (una sequenza di elementi consecutivi della lista) la somma dei cui elementi è massima
>![[Pasted image 20250429104738.png|450]]
>
>Progettare una algoritmo che risolva il problema in tempo $O(n)$
>>[!done]
>>Tentiamo un approccio basato sulla programmazione dinamica. Come è tipico di questa tecnica ci concentriamo sul calcolare il valore della soluzione. In un secondo momento, grazie alla tabella utilizzata, sarà possibile ottenere anche la sottosequenza
>>
>>Cominciamo con l’individuare i sottoproblemi dalla composizioni delle cui soluzioni sarà poi possibile risolvere il problema originario
>>Una scelta che ci porta a definire $n$ sottoproblemi è la seguente:
>>$$T[i]=\text{massima somma possibile per le sottoliste di}\,A\,\text{che terminano nella posizione}\,i$$
>>
>>Poiché la sottolista a valore massimo deve terminare in una qualche posizione, il valore della soluzione che cerchiamo sarà poi dato da:
>>$$\underset{ 0\leq i<n }{ \text{max} }\,T[i]$$
>>
>>Resta da definire la regola ricorsiva con cui calcolare i valori di $T[i]$ della tabella
>>$$T[i]=\begin{cases}A[0]&\text{se}\,i=0 \\\text{max}\bigl\{A[i],\,\,A[i]+T[i-1]\bigr\}&\text{altrimenti}\end{cases}$$
>>- $T[0]=A[0]$ in quanto esiste una sola sottosequenza nell’array di un solo elemento
>>- per $T[i]$ con $i>0$, la sottolista di valore massimo che termina con $A[i]$ può essere di due soli tipi:
>>	- consiste del solo elemento $A[i]$ → in questo caso ha valore $A[i]$
>>	- ha lunghezza superiore ad $1$ → in questo caso vale $A[i]+S$ dove $S$ è la massima somma di un sottovettore che termina in $i-1$ (il che significa che $S=T[i-1]$)
>>
>>Pertanto si ha:
>>$$T[i]=\text{max}\bigl\{A[i],\,\,A[i]+T[i-1]\bigr\}$$
>>
>>Implementazione:
>>```python
>>def es(A):
>>	n=len(A)
>>	if n==0:
>>		T = [0]*n
>>		T[0] = A[0]
>>		for i in range(i,n):
>>			T[i] = max(A[i], A[i]+T[i-1])
>>		return max(T)
>>```

>[!question] Data una sequenza $S$ di elementi di una sottosequenza di $S$ si ottiene eliminando zero o più elementi da $S$
>Ad esempio: data la sequenza $9,3,2,4,1,5,8,6,7,2$ una sua sottosequenza è $3,1,5,6$ (si ottiene eliminando i seguenti elementi $\textcolor{red}{9},3,\textcolor{red}{2},\textcolor{red}{4},1,5,\textcolor{red}{8},6,\textcolor{red}{7},\textcolor{red}{2}$)
>
>>[!info]
>>Le sottosequenze possibili per una sequenza di $n$ elementi sono $\Theta(2^n)$
>
>Una sottosequenza è detta crescente se i suoi elementi risultano ordinati in modo crescente
>Data una sequenza di interi, vogliamo trovare la lunghezza massima per le sottosequenze crescenti presenti in $S$
>
>Ad esempio per $S=9,3,2,4,1,5,8,6,7,2$ la risposta è $5$ infatti la sottosequenza crescente più lunga in $S$ è $3,4,5,6,7$ (non è l’unica, un’altra possibile soluzione è $2,4,5,6,7$)
>
>Progettare un algoritmo che data una sequenza $S$ di $n$ elementi, in tempo $O(n^2)$ risolve il problema


---
## Algoritmi pseudopolinomiali
Viene detto **pseudopolinomiale** un algoritmo che risolve un problema in tempo polinomiale quando i numeri presenti nell’input sono codificati in unario


