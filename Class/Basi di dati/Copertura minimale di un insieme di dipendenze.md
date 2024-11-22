---
Created: 2024-11-21
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
	- [[#Introduction#Che si fa ora?|Che si fa ora?]]
	- [[#Introduction#Prima di continuare|Prima di continuare]]
- [[#Copertura minimale|Copertura minimale]]
- [[#Come calcolare la copertura minimale|Come calcolare la copertura minimale]]
- [[#Esempi|Esempi]]
---
## Introduction
fino ad ora abbiamo parlato del perché possa essere necessario decomporre uno schema di relazione $R$, su cui è definito un insieme di dipendenze funzionali $F$, soprattutto in relazione a violazioni della 3NF che causano diversi tipi di anomalie

Abbiamo detto più volte che, qualunque sia il motivo che ci porta a decomporre lo schema, la decomposizione deve soddisfare tre requisiti fondamentali:
- ogni sottoschema deve essere 3NF
- la decomposizione deve preservare le dipendenze funzionali
- deve essere possibile ricostruire ogni istanza legale dello schema originale tramite join naturale di istanze della decomposizione

Nelle lezioni precedenti abbiamo visto come verificare che una decomposizione data (non ci interessa come sia sta prodotta) soddisfi tutte le indicazioni, in particolare abbiamo parlato di come verificare:
- se la decomposizione preserva le dipendenze funzionali ([[Decomposizioni che preservano le dipendenze|qui]])
- se sarà possibile ricostruire ogni istanza legale dello schema originale tramite join naturale di istanze della decomposizione ([[Decomposizioni che hanno un join senza perdita|qui]])

### Che si fa ora?
Ora affrontiamo il problema di come ottenere una decomposizione che soddisfi le nostre condizioni.
Prima di tutto: è sempre possibile ottenerla? Si, è sempre possibile, dato uno schema $R$ su cui è definito un insieme di dipendenze funzionali $F$, decomporlo in modo da ottenere che:
- ogni sottoschema è 3NF
- la decomposizione preserva le dipendenze funzionali
- è possibile ricostruire ogni istanza legale dello schema originale tramite join naturale di istanze della decomposizione

Presenteremo un algoritmo che raggiunge questo scopo

>[!info]
>Per la 3NF è sempre possibile applicare l’algoritmo di decomposizione (dopo aver trovato la copertura minimale), per la Boyce-Codd non esiste

>[!warning]
>- La decomposizione che si ottiene dall’algoritmo che studieremo non è l’unica possibile che soddisfi le condizioni richieste
>- Lo stesso algoritmo, a seconda dell’input di partenza (di cui parleremo) può fornire risultati diversi e tuttavia corretti
>- Attenzione a non confondere l’algoritmo per la decomposizione con quelli per la verifica
>- Proprio perché non esiste **la** decomposizione giusta, ma ci sono diverse possibilità, potrebbe succedere che la decomposizione da verificare non sia stata ottenuta tramite l’algoritmo, quindi usare l’algoritmo di decomposizione per controllare se produce la decomposizione da verificare, e ottenerne invece una diversa, non ci autorizza a concludere che la decomposizione da verificare non possegga le proprietà richieste 

### Prima di continuare
Prima di continuare dobbiamo introdurre il concetto di “copertura minimale” di un insieme $F$ di dipendenze funzionali. Sarà proprio una copertura minimale di $F$ a costruire l’input dell’algoritmo di decomposizione. Dato un insieme di dipendenze funzionali $F$, possono esserci più coperture minimali equivalenti (nel senso di avere tutte la stessa chiusura, che poi è uguale anche a quella di $F$). E’ proprio per questo motivo che l’algoritmo di decomposizione può produrre risultati diversi

---
## Copertura minimale
Dato un insieme di dipendenze $F$, la copertura minimale ha la sua stessa chiusura, ma è rimosso di tutte le ridondanze 

>[!info] Definizione
>Sia $F$ un insieme di dipendenze funzionali. Una *copertura minimale* di $F$ è un insieme $G$ di dipendenze funzionali equivalente ad $F$ tale che:
>- per ogni dipendenza funzionale in $G$ **la parte destra è un singleton**, cioè costituita da un unico attributo (ogni attributo nella parte destra è non ridondante)
>- per nessuna dipendenza funzionale $X\to A$ in $G$ esiste $X'\subset X$ tale che $G\equiv (G-\{X\to A\})\cup \{X'\to A\}$ (ogni attributo nella parte sinistra è non ridondante)
>- per nessuna dipendenza funzionale $X\to A$ in $G$, $G\equiv G-\{X\to A\}$ (ogni dipendenza è non ridondante)

Riformulando la definizione in modo più informale:
1. i dipendenti devono essere singleton
2. $\not\exists X \to A \text{ t.c. }F\equiv F-\{X\to A\}\cup \{X'\to A\} \text{ con }X'\subset X$
	$AB\to C$ può trovarsi nella copertura minimale se e solo se nella chiusura di $A$ e di $B$ non è presente $C$ (in caso contrario viene sostituito da $A\to C$ oppure da $B\to C$ in base a dove trovo la chiusura)
3. $\not\exists X\to A \text{ t.c. }F\equiv F-\{X\to A\}$
	posso eliminare una dipendenza se è possibile ricostruirla in $F^+$ per transitività

---
## Come calcolare la copertura minimale
Per trovare la copertura minimale su un insieme di dipendenze $F$ devo:
1. applico la decomposizione → $AB\to C\Rightarrow A\to C,B\to C$
2. data $X\to A$ devo verificare se $\forall X'\subset X$ ho che $F\equiv F-\{X\to A\}\cup \{X'\to A\}$
	![[Pasted image 20241122000018.png]]
3. data $X\to A$ devo verificare se $F\equiv F-\{X\to A\}$
	![[Pasted image 20241122000528.png|250]]
	$\text{DIP1, DIP2, DIP3}$ appartengono ad entrambi, dunque ci basta verificare che $X\to A\in G^+\overset{\text{lemma 1}}{\implies}A\in X^+_{G}$

>[!warning]
>E’ importante rispettare l’ordine dei passi 2 e 3 in quanto, se generalmente il risultato è comunque corretto, ci sono casi in cui questo non è vero

---
## Esempi

>[!example]- Esempio 1
>$$R=(A,B,C,D,E,H)$$
>$$F=\{AB\to CD,C\to E,AB\to E,ABC\to D\}$$
>
>Trovare una copertura minimale $G$ di $F$
>
>##### Passo 1
>Prima di tutto riduciamo le parti destre a singleton
>$$F=\{AB\to C, AB\to D,C\to E,AB\to E,ABC\to D\}$$
>
>##### Passo 2
>Ora dobbiamo verificare se nelle dipendenze ci sono ridondanze nelle parti sinistre.
>Cominciamo dalla dipendenza $AB\to C$ e controlliamo se $A\to C$ appartiene a $F^+$, cioè se $C \in(A)^+_{F}$. Ma $(A)^+_{F}=\{A\}$ e stessa cosa per $B$ infatti $(B)^+_{F}=\{B\}$, quindi la parte sinistra non può essere ridotta
>
>Lo stesso si verifica per le dipendenze $AB\to D$ e $AB\to E$
>Proviamo allora a ridurre $ABC\to D$, poiché nell’insieme di dipendenze esiste $AB\to D$, possiamo non solo eliminare l’attributo $C$ ma anche tutta la dipendenza risultante che è un duplicato.
>Alla fine di questo passo abbiamo un insieme
>$$G=\{AB\to C,AB\to D,C\to E,AB\to E\}$$
>
>##### Passo 3
>Vediamo ora se questo insieme contiene delle dipendenze ridondanti
>Intanto possiamo considerare che $C$ viene determinato unicamente da $AB$, quindi eliminando la dipendenza $AB\to C$ non riusciremmo più ad inserirlo nella chiusura di $AB$ rispetto al nuovo insieme di dipendenze. Lo stesso vale per $D$
>
>Proviamo allora ad eliminare la dipendenza $C\to E$. Rispetto al nuovo insieme di dipendenze di prova $G=\{AB\to C,AB\to D,AB\to E\}$ abbiamo che $(C)^+_{G}=\{C\}$ in cui non compare $E$. La dipendenza deve dunque rimanere
>
>Proviamo infine ad eliminare $AB\to E$. Rispetto a $G=\{AB\to C,AB\to D,C\to E\}$ abbiamo che $(AB)^+_{G}=\{A,B,C,D,E\}$ in cui $E$ compare. Ciò significa che $E$ rientra comunque nella chiusura di $AB$ perché la dipendenza $AB\to E$, pur non comparendo in $G$, si trova in $G^+$, e quindi può essere eliminata
>
>La **copertura minimale** di $F$ è:
>$$G=\{AB\to C,AB\to D,C\to E\}$$

>[!example]- Esempio 2
>$$F=\{BC\to DE,C\to D,B\to D,E\to L,D\to A,BC\to AL\}$$
>
>Trovare una copertura minimale $G$ di $F$
>
>##### Passo 1
>Prima di tutto riduciamo le parti destre a singleton
>$$F=\{BC\to D, BC\to E,C\to D,B\to D,E\to L,D\to A,BC\to A, BC\to L\}$$
>
>##### Passo 2
>Ora dobbiamo verificare se nelle dipendenze ci sono ridondanze nelle parti sinistre.
>Cominciamo dalla dipendenza $BC\to D$ e controlliamo se $B\to D$ oppure $C\to D$ appartengono a $F^+$, cioè se $D \in(B)^+_{F}$ oppure $D\in(C)^+_{F}$. Notiamo però che in $F$ abbiamo sia $C\to D$ che $B\to D$, quindi $BC\to D$ è sicuramente ridondante. La eliminiamo e il nostro $F$ diventa
>$$F=\{BC\to E,C\to D,B\to D,E\to L,D\to A,BC\to A,BC\to L\}$$
>
>Continuiamo con la dipendenza $BC\to E$; dobbiamo controllare se $B\to E$ oppure $C\to E$ appartengono a $F^+$, cioè se $E\in(B)^+_{F}$ oppure $E\in(C)^+_{F}$. Applicando l’algoritmo di chiusura otteniamo: $(B)^+_{F}=\{B,D,A\}$ e $(C)^+_{F}=\{C,D,A\}$, quindi non possiamo eliminare elementi a sinistra
>
>>[!hint]
>>In effetti bastava osservare che $E$ compare a destra solo du questa dipendenza (è determinato funzionalmente solo da questa coppia di attributi) e quindi non avremmo potuto inserirlo nelle chiusure dei singoli attributi in nessun altro modo
>
>Continuiamo con $BC\to A$. Abbiamo già calcolato le chiusure di $B$ e $C$ ($F$ non è cambiamo oppure sarebbe un insieme equivalente), e in entrambe troviamo l’attributi $A$
>
>>[!warning]
>>Questa volta le dipendenze $B\to A$ e $C\to A$ non sono in $F$, quindi non possiamo semplicemente eliminare $BC\to A$ ma va effettuata la sostituzione con una delle due.
>
>Scegliamo per esempio, come dipendenza da tenere $B\to A$
>Abbiamo quindi:
>$$F=\{BC\to E,C\to D,B\to D,E\to L,D\to A,B\to A,BC\to L\}$$
>
>Continuiamo con la dipendenza $BC\to L$; dobbiamo controllare se $B\to L$ oppure $C\to L$ appartengono a $F^+$, cioè se $L\in(B)^+_{F}$ oppure $L\in(C)^+_{F}$. Abbiamo già calcolato le chiusure di $B$ e $C$, e verifichiamo che in nessuna delle due troviamo l’attributo $L$, quindi non possiamo eliminare elementi a sinistra
>
>Alla fine del passo 2 abbiamo quindi:
>$$F=\{BC\to E,C\to D,B\to D,E\to L,D\to A,B\to A,BC\to L\}$$
>
>##### Passo 3
>Vediamo ora se questo insieme contiene delle dipendenze ridondanti
>Intanto possiamo considerare che $E$ viene determinato unicamente da $BC$, quindi eliminando la dipendenza $BC\to E$ non riusciremmo più ad inserirlo nella chiusura di $BC$ rispetto al nuovo insieme di dipendenze.
>
>Proviamo allora ad eliminare la dipendenza $C\to D$. Rispetto al nuovo insieme di dipendenze di prova $G=\{BC\to E,B\to D,E\to L,D\to A,B\to A,BC\to L\}$ abbiamo che $(C)^+_{G}=\{C\}$ in cui non compare $D$. La dipendenza deve dunque rimanere
>
>Proviamo allora ad eliminare la dipendenza $B\to D$. Rispetto al nuovo insieme di dipendenze di prova $G=\{BC\to E,C\to D,E\to L,D\to A,B\to A,BC\to L\}$ abbiamo che $(B)^+_{G}=\{B,A\}$ in cui non compare $D$. La dipendenza deve dunque rimanere
>
>Proviamo allora ad eliminare la dipendenza $E\to L$. Rispetto al nuovo insieme di dipendenze di prova $G=\{BC\to E,C\to D,B\to D,D\to A,B\to A,BC\to L\}$ abbiamo che $(E)^+_{G}=\{E\}$ in cui non compare $L$. La dipendenza deve dunque rimanere
>
>Proviamo allora ad eliminare la dipendenza $D\to A$. Rispetto al nuovo insieme di dipendenze di prova $G=\{BC\to E,C\to D,B\to D,E\to L,B\to A,BC\to L\}$ abbiamo che $(D)^+_{G}=\{D\}$ in cui non compare $A$. La dipendenza deve dunque rimanere
>
>Proviamo allora ad eliminare la dipendenza $B\to A$. Rispetto al nuovo insieme di dipendenze di prova $G=\{BC\to E,C\to D,B\to D,E\to L,D\to A,BC\to L\}$ abbiamo che $(B)^+_{G}=\{B,D,A\}$ in cui compare $A$. La dipendenza dunque può essere eliminata.
>Così la nostra $F$ diventa:
>$$F=\{BC\to E,C\to D,B\to D,E\to L,D\to A,BC\to L\}$$
>
>Proviamo infine ad eliminare $BC\to L$. Rispetto a $G=\{BC\to E,C\to D,B\to D,E\to L,D\to A\}$ abbiamo che $(BC)^+_{G}=\{A,B,C,D,E,L\}$ in cui $L$ compare. La dipendenza dunque può essere eliminata.
>
>La **copertura minimale** di $F$ è:
>$$G=\{BC\to E,C\to D,B\to D,E\to L,D\to A\}$$

>[!example]- Esempio 3
>$$F=\{AB\to C,A\to E,E\to D,D\to C,B\to A\}$$
>
>Trovare una copertura minimale $G$ di $F$
>
>##### Passo 1
>Non c’è bisogno di decomporre le parti destre delle dipendenze in $F$ infatti sono già singleton
>
>##### Passo 2
>Ora dobbiamo verificare se nelle dipendenze ci sono ridondanze nelle parti sinistre.
>Verifichiamo se la dipendenza $AB\to C$ ha attributi ridondanti a sinistra. Verifichiamo se $C\in(A)^+_{F}$ oppure $C\in(B)^+_{F}$; $(A)^+_{F}=\{A,E,D,C\}$ e $(B)^+_{F}=\{B,A,E,D,C\}$. Quindi $AB\to C$ può essere sostituito con $A\to C$ oppure con $B\to C$.
>Per questo esempio sceglieremo $A\to C$
>Abbiamo quindi:
>$$F=\{A\to C,A\to E,E\to D,D\to C,B\to A\}$$
>
>##### Passo 3
>Vediamo ora se questo insieme contiene delle dipendenze ridondanti
>
>Proviamo allora ad eliminare la dipendenza $A\to C$. Rispetto al nuovo insieme di dipendenze di prova $G=\{A\to E,E\to D,D\to C,B\to A\}$ abbiamo che $(A)^+_{G}=\{A,C,D,E\}$ in cui compare $C$. La dipendenza deve dunque può essere rimossa
>
>Notiamo che nessun attributo compare a destra di più di una dipendenza, quindi non potrebbe rientrare nelle chiusure delle parti sinistre per transitività, quindi nessuna altra dipendenza può essere eliminata
>
>La **copertura minimale** di $F$ è:
>$$G=\{A\to E,E\to D,D\to C,B\to A\}$$
