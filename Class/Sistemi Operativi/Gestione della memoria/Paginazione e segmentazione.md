---
Created: 2024-11-12
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Index
- [[#Paginazione (semplice)|Paginazione (semplice)]]
- [[#Segmentazione (semplice)|Segmentazione (semplice)]]
	- [[#Segmentazione (semplice)#Indirizzi Logici|Indirizzi Logici]]
---
## Paginazione (semplice)
La paginazione semplice in quanto tale non è stata sostanzialmente mai usata, ma è importante a livello concettuale per introdurre la memoria virtuale.
Con la paginazione sia la memoria che i processi vengono “spacchettati” in pezzetti di dimensione uguale. Ogni pezzetto del processo è chiamato **pagina**, mentre i pezzetti di memoria sono chiamati **frame**.
Ogni pagina, per essere usata, deve essere collocata in un frame ma pagine contigue di un processo possono essere messe in un qualunque frame (anche distanti)

I SO che la adottano però devono mantenere una tabella delle pagine per ogni processo che associa ogni pagina del processo al corrispettivo frame in cui si trova.

>[!info] Quando c’è un process switch, la tabella delle pagine del nuovo processo deve essere ricaricata ed aggiornata

A differenza di prima in cui l’hardware doveva solamente intervenire e aggiungere un offset, qui deve intervenire sulle pagine stesse, infatti un indirizzo di memoria può essere visto come un numero di pagina e uno spiazzamento al suo interno (indirizzo logico)

>[!example]
>![[Pasted image 20241021233338.png|250]]
>
>Deve essere caricato un processo A che occupa 4 frame
>![[Pasted image 20241021233436.png|250]]
>
>Ne arrivano altri due da 3 (B) e 4 (C) frame 
>![[Pasted image 20241021233523.png|250]]
>
>Quindi viene swappato B
>![[Pasted image 20241021233610.png|250]]
>
>E sostituito con D (con il partizionamento dinamico, non sarebbe stato possibile caricare D in memoria)
>![[Pasted image 20241021233636.png|250]]
>
>Tabelle delle pagine risultanti
>![[Pasted image 20241021233804.png|450]]

>[!example] Esempio di traduzione
>Supponiamo che la dimensione di una pagina sia di $100 \text{ bytes}$. Quindi la RAM dell’esempio precedente è di solo $1400 \text{ bytes}$
>Inoltre i processi A, B, C, D richiedono solo $400$, $300$, $400$ e $500 \text{ bytes}$ rispettivamente (comprensivi di codice - program, dati globali ed heap - data e stack delle chiamate).
>Nelle istruzioni dei processi, i riferimenti alla RAM sono relativi all’inizio del processo (quindi, ad esempio, per D ci saranno riferimenti compresi nell’intervallo $[0, 499]$)
>
>Supponiamo ora che attualmente il processo D sia in esecuzione, e che occorra eseguire l’istruzione `j 343` (vale lo stesso anche per istruzioni di load o store, anche se devono passare per registri)
>Ovviamente non si tratta dell’indirizzo $343$ della RAM: lì c’è il processo A.
>Bisogna capire in quale pagina di D si trova $343$: basta fare `343 div 100 = 3`
>
>Poi occorre guardare la tabella delle pagine di D: la pagina $3$ corrisponde al frame di RAM numero $11$
>Il frame $11$ ha indirizzi che vanno da $1100$ a $1199$: qual è il numero giusto? Basta fare `343 mod 100 = 43` quindi il 44-esimo byte
>L’indirizzo vero è pertanto $11\cdot 100+43=1143$

>[!info]
>Per ogni processo, il numero di pagine è al più il numero di frames (non sarà più vero con la memria virtuale)
>![[Pasted image 20241024195255.png|440]]
>Per ottenere l’indirizzo vero dunque punto alla pagina formata dai $6 \text{ bit}$ più significativi, controllando la corrispondenza con il frame, e utilizzo i restanti $10 \text{ bit}$ (ogni pagina è grande $2^{10} \text{ bit}$) come offset all’interno della pagina

---
## Segmentazione (semplice)
La differenza tra paginazione e segmentazione sta nel fatto che nella paginazione le pagine sono tutte di ugual dimensione, mentre i segmenti hanno **lunghezza variabile**.
In questo risulta simile al partizionamento dinamico ma è il programmatore a decidere come deve essere segmentato il processo (tipicamente viene fatto un segmento per il codice sorgente, uno per i dati condivisi e uno per lo stack delle chiamate)

### Indirizzi Logici
![[Pasted image 20241024200117.png]]

>[!info]
>Qui si suppone che non possano esserci segmenti più grandi di $2^{12} \text{ bytes}$
>![[Pasted image 20241024201839.png|440]]
>In questo caso nella tabella delle corrispondenze oltre all’indirizzo base del segmento, ci sta anche la sua lunghezza
