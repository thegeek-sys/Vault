---
Created: 2024-10-27
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Index
- [[#Quali sono i problemi di uno schema mal progettato?|Quali sono i problemi di uno schema mal progettato?]]
	- [[#Quali sono i problemi di uno schema mal progettato?#Considerazioni|Considerazioni]]
	- [[#Quali sono i problemi di uno schema mal progettato?#Conclusioni|Conclusioni]]
- [[#Terza forma normale|Terza forma normale]]
	- [[#Terza forma normale#Definizione|Definizione]]
	- [[#Terza forma normale#Esempi|Esempi]]
- [[#Dipendenza parizale|Dipendenza parizale]]
	- [[#Dipendenza parizale#Definizione|Definizione]]
- [[#Dipendenza transitiva|Dipendenza transitiva]]
	- [[#Dipendenza transitiva#Definizione|Definizione]]
- [[#Definizione alternativa di 3NF (teorema)|Definizione alternativa di 3NF (teorema)]]
- [[#Cosa vogliamo ottenere?|Cosa vogliamo ottenere?]]
	- [[#Cosa vogliamo ottenere?#La 3NF non basta|La 3NF non basta]]
		- [[#La 3NF non basta#Esempi|Esempi]]
- [[#Forma normale di Boyce-Codd|Forma normale di Boyce-Codd]]
	- [[#Forma normale di Boyce-Codd#Esempio|Esempio]]
---
## Quali sono i problemi di uno schema mal progettato?
Ritorniamo al nostro esempio di base di dati che contiene le informazioni sugli studenti e sugli esami sostenuti, e ripartiamo dalla soluzione “buona” tornata alla fine

![[Progettazione di una base di dati relazionale - Problemi e vincoli#Ipotesi 3]]

### Considerazioni
Poiché il numero di matricola identifica univocamente uno studente, ad ogni numero di matricola corrisponde:
- un solo codice fiscale ($\text{Matr}\to \text{CF}$)
- un solo cognome ($\text{Matr}\to \text{Cogn}$)
- un solo nome ($\text{Matr}\to \text{Nome}$)
- una sola data di nascita ($\text{Matr}\to \text{Data}$)
- un solo comune di nascita ($\text{Matr} \to \text{Com}$)

Quindi un’istanza di $\text{Studente}$ per essere legale deve soddisfare la dipendenza funzionale
$$
\text{Matr}\to \text{Matr, CF, Cogn, Nome, Data, Com}
$$
Con considerazioni analoghe abbiamo che un’istanza di $\text{Studente}$ per essere legale deve soddisfare la dipendenza funzionale
$$
\text{CF}\to \text{Matr, CF, Cogn, Nome, Data, Com}
$$
Pertanto sia $\text{Matr}$ che $\text{CF}$ sono chiavi di $\text{Studente}$

D’altra parte possiamo osservare che ci possono essere due studenti con lo stesso cognome e nomi differenti quindi possiamo avere due istanze di $\text{Studente}$ che **non** soddisfano le dipendenza funzionale $\text{Cogn}\to \text{Nome}$. Inoltre possiamo avere istanze di $\text{Studente}$ che non soddisfano: $\text{Cogn}\to \text{Nome}$, $\text{Cogn}\to \text{Data}$, $\text{Cogn}\to \text{Com}$ ecc.

Con ciò possiamo concludere che le **uniche dipendenze funzionali non banali** che devono essere soddisfatte da un’istanza legale di $\text{Studente}$ sono del tipo
$$
K\to X
$$
dove $K$ contiene una chiave ($\text{Matr}$ o $\text{CF}$)

>[!hint]
>Vedremo che questa è la prima condizione che però va ulteriormente rifinita per arrivare ad una definizione precisa di terza forma normale (3NF)


$$
\text{Esame(Matr, C\#, Data, Voto)}
$$
Uno studente può sostenere l’esame relativo ad un corso una sola volta; pertanto per ogni esame esiste:
- una sola data (in cui è stato sostenuto)
- un solo voto
Quindi ogni istanza legale di $\text{Esame}$ deve soddisfate la dipendenza funzionale
$$
\text{Matr, C\#}\to \text{Data, Voto}
$$
D’altra parte uno studente può sostenere esami in dati differenti e riportare voti diversi nei vari esami. Pertanto esistono istanze di $\text{Esame}$ che non soddisfano una o entrambe le dipendenze funzionali: $\text{Matr}\to \text{Data}$, $\text{Matr}\to \text{Voto}$

Inoltre l’esame relativo ad un certo corso non può essere superato da diversi studenti in date diverse e con voti diversi. Pertanto esistono istanze di $\text{Esame}$ che non soddisfano una o entrambe le dipendenze funzionali: $\text{C\#}\to \text{Data}$, $\text{C\#}\to \text{Voto}$
Pertanto $\text{Matr, C\#}$ è una chiave per $\text{Esame}$

>[!hint]
>Vedremo in seguito delle procedure rigorose per identificare la/le chiavi

### Conclusioni
Per ciascun schema di relazione:
- $\text{Studente (Matr, CF, Cogn, Nome, Data, Com)}$
- $\text{Corso (C\#, Tit, Doc)}$
- $\text{Esame (Matr, C\#, Data, Voto)}$
- $\text{Comune (Com, Prov)}$

>[!info] Stiamo continuando ad assumere che $\text{Com}\to \text{Prov}$, cioè che non ci sono comuni omonimi

Le **uniche dipendenze funzionali non banali** che devono essere soddisfatte da ogni istanza legale di sono del tipo
$$
K\to X
$$
dove $K$ contiene una chiave

---
## Terza forma normale
Uno schema di relazione è in **3NF** se le uniche dipendenze funzionali non banali (non riflessive) che devono essere soddisfatte da ogni istanza legale sono del tipo
$$
K\to X
$$
dove $K$ contiene una chiave oppure $X$ è contenuto in una chiave

### Definizione
Dati uno shema di relazione $R$ e un insieme di dipendenze funzionali $F$ su $R$, $R$ è in **3NF** se
$$
\forall X\to A\in F^+ \,\,\,\,\,\,\,A\not\in X
$$
- $A$ appartiene ad una chiave (è **primo**, applicabile solo ai cingleton)
- $X$ contiene una chiave (è una superchiave)

>[!warning]
>- è sbagliato scrivere $\forall X\to A\in F$, perché non sapremmo se e come valutare una dipendenza del tipo $X\to AB$ (due o più attributi a destra)
>- se sostituisco $\forall X\to A\in F$ con $\forall X\to Y\in F$, non so come comportarmi se $Y$ contiene sia attributi primi che non
>- la condizione $A \not\in X$ è importante. Infatti, per l’assioma della riflessività, se $A\in X$ avremo sempre $X\to A \in F^A$ e quindi in $F^+$, anche quando $A$ non è primo e $X$ non è superchiave, e quindi se considerassimo questo tipo di dipendenze nessuno schema risulterebbe in 3NF

### Esempi
>[!example] Esempio 1
>$$
\begin{flalign}R=A,B,C,D&& F=\{A\to B, B\to CD\}\end{flalign}
>$$
>
>Ho come chiavi:
>- $K_{1}=A$ → $A\to CD$ per transitività e $A\to BCD$ per [[Chiusura di un insieme di dipendenze funzionali#Regola dell’unione|unione]]
>
>$A\to B$ → rispetta le condizioni per essere in 3NF
>$B\to BC$ → $B$ non è superchiave, controllo quindi i determinati. $C$ e $D$ violano la 3NF perché in entrambi i casi non fanno parte di una chiave
>
>Lo schema non è in 3NF

>[!example] Esempio 2
>$$
\begin{flalign}R=A,B,C,D&& F=\{AB\to CD, AC\to BD, D\to BC\}\end{flalign}
>$$
>
>Ho come chiavi:
>- $K_{1}=AB$
>- $K_{2}=AC$
>- $K_{3}=AD$ (per il teorema dell’aumento sull’ultima dipendenza funzionale)
>
>
>D nell’ultima dipendenza non è una chiave (ma un pezzo di una chiave) però il determinato è composto da $B$ (un attributo della chiave $AB$) e da $C$ (un attributo della chiave $AC$); dunque lo schema è in **3FN**.
>
>Ho infatti decomposto in $D\to B$ ($B$ è parte di una chiave) e in $D\to C$ ($C$ è una parte di una chiave)


---
## Dipendenza parizale
Siano $R$ uno schema di relazione e $F$ un insieme di dipendenze funzionali su $R$.
$X\to A\in F^+\mid A\not\in X$ è una **dipendenza parziale** su $R$ se $A$ non è primo ed $X$ è contenuto propriamente in una chiave di $R$
![[Pasted image 20241027185741.png|150]]

### Definizione
$A$ dipende parzialmente da una chiave $K$ se $\exists X\subset R$ tale che $K\to R\in F^+$ con $A\not\in X$ e tale che $X\subset K$ e $A$ non è parte di una chiave

>[!example]
>$$\text{Curriculum (}\mathbf{Matr}\text{, CF, Cogn, Nome, DataN, Com, Prov, }\mathbf{C\#}\text{, Tit, Doc, DataE, Voto)}$$
>
>Ad un numero di matricola corrisponde un solo cognome: $\text{Matr}\to \text{Cogn}$
>Quindi ad una coppia costituita da un numero di matricola e da un codice di corso corrisponde un solo cognome: $\text{Matr, C\#}\to \text{Cogn}$
>
>L’attributo $\text{Cogn}$ dipende parzialmente dalla chiave $\text{Matr, C\#}$ ($\text{Matr, C\#}\to \text{Cogn}$ è una conseguenza di $\text{Matr}\to \text{Cogn}$)
>
>>[!info]
>>$\text{Matr}$ è propriamente contenuto in una chiave

---
## Dipendenza transitiva
Siano $R$ uno schema di relazione e $F$ un insieme di dipendenze funzionali su $R$.
$X\to A\in F^+\mid A\not\in X$ è una **dipendenza transitiva** su $R$ se $A$ non è primo e per ogni chiave $K$ di $R$ si ha che $X$ non è contenuto propriamente in $K$ e $K-X\neq\varnothing$
![[Pasted image 20241027185912.png|550]]

### Definizione
$A$ dipende parzialmente da una chiave $K$ se $\exists X\subset R$ tale che $K\to R\in F^+$ con $A\not\in X$ e $X\to A\in F^+$ e $A$ non è parte di una chiave

>[!example]
>$$\text{Studente (}\mathbf{Matr}\text{, CF, Cogn, Nome, Data, Com, Prov)}$$
>Ad un numero di matricola corrisponde un solo comune di nascita: $\text{Matr}\to \text{Com}$
>Un comune si trova in una sola provincia: $\text{Com}\to \text{Prov}$
>Quindi ad un numero di matricola corrisponde una sola provincia: $\text{Matr}\to \text{Prov}$
>
>L’attributo $\text{Prov}$ dipende transitivamente dalla chiave $\text{Matr}$ ($\text{Matr}\to \text{Prov}$ è una conseguenza di $\text{Matr}\to \text{Com}$ e $\text{Com}\to \text{Prov}$)
>
>>[!info]
>>Per ogni chiave $K$ di $R$ ($\text{Matr}$ e $\text{CF}$) $\text{Com}$ non è contenuto propriamente nella chiave ($\text{Com}$ non è sottoinsieme né di $\text{Matr}$ né di $\text{CF}$) e $K-\text{Com}\neq\varnothing$ ($\text{Matr}-\text{Com}=\text{Matr}$ e $\text{CF}-\text{Com}=\text{CF}$)


---
## Definizione alternativa di 3NF (teorema)
Dato uno schema $R$ e un insieme funzionali $F$ su $R$, $R$ è in 3NF se e solo se in $F$ non ci sono **né dipendenze parziali né dipendenze transitive**

>[!info] Dimostrazione
>##### Parte $\Rightarrow$
>Lo schema $R$ è in 3NF, quindi $\forall X\to A\in F^+, \,A\not\in X$
>$A$ appartiene ad una chiave (è primo) oppure $X$ contiene una chiave (è superchiave)
>
>- Se $A$ è parte di una chiave (è primo), viene a mancare la prima condizione per avere una dipendenza parziale o transitiva
>- Se $A$ non è primo (non fa parte di nessuna chiave), allora $X$ è superchiave. Dunque si ha che $X\supset K$ facendo mancare la seconda condizione per la dipendenza parziale (non può essere che $X\subset K$); per lo stesso motivo non si può verificare che $K-X\neq\varnothing$ quindi la dipendenza non può essere transitiva
>
>##### Parte $\Leftarrow$
>Supponiamo per assurdo che non ci sono dipendenze parziali e transitive e lo schema non sia in 3NF. In tal caso vuol dire che ci sta almeno una dipendenza che viola la 3NF (ovvero che $A$ non è primo e $X$ non è una superchiave).
>Siccome $A$ non è primo devo verificare il secondo punto della 3NF
>
>- Se $X$ non è superchiave vuol dire che $\forall K \in R$, $X\not\subset K$ e $K-X\neq \varnothing$ e quindi vuol dire che ci sta una dipendenza transitiva. **CONTRADDIZIONE**
>- Se $X$ è una porzione di una chiave, ovvero se $\exists K\in R \mid X\subset K$, in tal caso $X\to A$ è una dipendenza parziale su $R$. **CONTRADDIZIONE**


---
## Cosa vogliamo ottenere?
Abbiamo visto che uno schema in 3NF ha delle buone proprietà che lo rendono preferibile ad uno che non è in 3NF. Un obbiettivo da tener presente quando si progetta una base di dati è quello di produrre uno schema in un ogni relazione sia in 3NF.
Normalmente nella fase di progettazione concettuale si usa il modello Entità-Associazione e si individuano per l’appunto i concetti che devono essere rappresentati nella base di dati

Se il lavoro di individuazione è fatto accuratamente lo schema relaziona può essere derivato con opportune regole, è in 3NF. Se tuttavia, dopo tale processo, ci ritrovassimo a produrre uno schema che non è in 3NF dovremmo procedere ad una fare di **decomposizione** di tale schema in maniera analoga a quella esaminata nell’esempio sui dati di un’Università ([[Progettazione di una base di dati relazionale - Problemi e vincoli#Introduzione|qui]])
### La 3NF non basta
Uno schema che non è in 3NF può essere decomposto in più modi in un insieme di schemi in 3NF. Ad esempio lo schema $R=ABC$ con l’insieme di dipendenze funzionali $F=\{A\to B, B\to C\}$ non è in 3NF per la presenza in $F^+$ della dipendenza transitiva $B\to C$, dato che la chiave è evidentemente $A$.
$R$ può essere decomposto in:
- $R_{1}=AB$ con $\{A\to B\}$
- $R_{2}=BC$ con $\{B\to C\}$
oppure:
- $R_{1}=AB$ con $\{A\to B\}$
- $R_{2}=AC$ con $\{A\to C\}$

Entrambi gli schemi sono in 3NF, tuttavia la seconda soluzione non è soddisfacente.
Consideriamo due istanze legali degli schemi ottenuti
![[Pasted image 20241027212903.png|250]]
L’istanza dello schema originario $R$ che posso ricostruire da questa attraverso il join naturale è la seguente
![[Pasted image 20241027213020.png|170]]
Ma non è un’istanza legale di $R$ in quanto **non soddisfa la dipendenza funzionale** $B\to C$

>[!warning]
>Occorre preservare tutte le dipendenze in $F^+$
>Per **perdita di dati** si intende sia la perdita delle tuple originali sia l’aggiunta di tuple non presenti originariamente

In conclusione, quando si decompone uno schema per ottenerne uno in 3NF occorre tenere presenti altri due requisiti dello schema decomposto:
- deve **preservare le dipendenze funzionali** che valvono su ogni isntanza legasle dello schema originario
- deve permettere di **ricostruire mediante il join naturale** ogni **istanza legale dello schema originario** (senza aggiunta di informazione estranea)

>[!hint]
>Se ci sono delle dipendenze che rispettano la seconda condizione del 3NF (determinato primo) devo fare attenzione ad aggiungere dei contrains nella decomposizione dello schema in quanto si potrebbe violare una dipendenza funzionale

#### Esempi
>[!example]
>Consideriamo lo schema $R=\{\text{Matricola, Comune, Provincia}\}$ con l’insieme di dipendenze funzionali $F=\{\text{Matricola}\to \text{Comune, }\text{Comune}\to \text{Provincia}\}$
>Lo schema non è in 3NF per la presenza in $F^+$ della dipendenza transitiva $\text{Comune}\to \text{Provincia}$, dato che la chiave è evidentemente $\text{Matricola}$ ($\text{Provincia}$ dipende transitivamente da $\text{Matricola}$)
>
>$R$ può essere decomposto in:
>- $R_{1}=(\text{Matricola, Comune})$ con $\{\text{Matricola}\to \text{Comune}\}$
>- $R_{2}=(\text{Matricola, Provincia})$ con $\{\text{Comune}\to \text{Provincia}\}$
>
>oppure:
>- $R_{1}(\text{Matricola, Comune})$ con $\{\text{Matricola}\to \text{Comune}\}$
>- $R_{2}(\text{Matricola, Provincia})$ con $\{\text{Matricola}\to \text{Provincia}\}$
>
>Entrambi gli schemi sono in 3NF, tuttavia la seconda soluzione non è soddisfacente
>
>Consideriamo le istanze legali degli schemi ottenuti
>![[Pasted image 20241031113211.png|440]]
>L’istanza dello schema originario $R$ che posso ricostruire da questa (facendo un join naturale) è la seguente
>![[Pasted image 20241031113401.png|350]]
>Ma non è un’istanza legale di $R$, in quanto non soddisfa la dipendenza funzionale $\text{Comune}\to \text{Provincia}$
>
>>[!hint]
>>E’ evidente che ci sia stato un errore di inserimento, ma non abbiamo potuto rivelarlo

>[!example]
>Consideriamo ora lo schema $R=ABC$ con l’insieme di dipendenze funzionali $F=\{A\to B,B\to C\}$. Lo schema non è in 3NF per la presenza in $F^+$ delle dipendenze funzionali parziali $A\to B$ e $C\to B$, dato che la chiave è $AC$ tale schema può essere decomposto in:
>- $R_{1}=AB$ con $\{A\to B\}$
>- $R_{2}=BC$ con $\{C\to B\}$
>
>Tale schema pur preservando tutte le dipendenze in $F^+$ non è soddisfacente.
>
>Consideriamo l’istanza legale di $R$
>![[Pasted image 20241102223833.png|150]]
>In base alla decomposizione data, questa istanza di decompone in:
>![[Pasted image 20241102224009.png|200]]
>E dovrebbe essere possibile ricostruirla esattamente tramite join e invece se si effettua il join si avrà come risultato il prodotto cartesiano
>![[Pasted image 20241102224108.png|150]]

>[!example]
>Consideraimo ora lo schema 
>- $R=(\text{Matricola, Progetto, Capo)}$ 
>- $F=\{\text{Matricola}\to \text{Progetto}, \text{Capo}\to \text{Progetto}\}$
>
>Il progetto ha più capi ma ogni capo ha un solo progetto, e un impiegato su un progetto  dà conto ad un solo capo (ogni capo segue un gruppo).
>Lo schema non è in 3NF per la presenza in $F+$ delle dipendenze parziali $\text{Matricola}\to \text{Progetto}$ e $\text{Capo}\to \text{Progetto}$.
>
>Dato che la chiave è $(\text{Matricola, Capo})$ tale schema può essere decomposto in:
>- $R_{1}=(\text{Matricola, Progetto})$ con $\{\text{Matricola} \to \text{Progetto}\}$
>- $R_{2}=(\text{Progetto, Capo})$ con $\{\text{Capo}\to \text{Progetto}\}$
>
>Tale schema pur preservando tutte le dipendenze in $F^+$ non è soddisfacente
>
>Consideriamo l’istanza legale di $R$
>![[Screenshot 2024-11-02 alle 22.45.28.png|380]]
>In base alla decomposizione data, questa istanza si decompone in:
>![[Pasted image 20241102225443.png|480]]
>Ma quando la si ricostruisce tramite join si ottiene uno schema in cui sono presenti delle tuple estranee (perdita di informazione)
>![[Screenshot 2024-11-02 alle 22.55.46.png|300]]

---
## Forma normale di Boyce-Codd
La 3NF non è la più restrittiva che si può ottenere ma ne esistono altre tra cui la forma normale di **Boyce-Codd**

>[!info] Definizione
>Una relazione è in forma normale di Boyce-Codd (BCNF) quando in essa **ogni determinante è una superchiave** (ricordiamo che una chiave è anche superchiave)

Una relazione che rispetta la forma normale di Boyce-Codd è anche in terza forma normale, ma non il contrario

### Esempio
>[!example]
>Consideriamo una relazione che descrive l’allocazione delle sale operatorie di un ospedale. Le sale operatorie sono prenotate, giorno per giorno, in orari previsti, per effettuare interventi su pazienti ad opera dei chirurghi dell’ospedale.
>Nel corso di una giornata una sala operatoria è occupata sempre dal medesimo chirurgo che effettua più interventi, in ore diverse.
>Noti i valori di $\text{Paziente}$ e $\text{DataIntervento}$, sono noti anche: ora dell’intervento, chirurgo e sala operatoria utilizzata
>
>Schema: $\text{Interventi(Paziente, DataIntervento, OraIntervento, Chirurgo, Sala)}$
>In base alla presedente descrizione, nella relazione $\text{Interventi}$ valgono le dipendenze funzionali:
>- $\{\text{Paziente, DataIntervento}\}\to \text{OraIntervento, Chirurgo, Sala}$
>- $\{\text{Chirurgo, DataIntevento, OraIntervento}\}\to \text{Paziente, Sala}$
>- $\{\text{Sala, DataIntervento, OraIntervento}\}\to \text{Paziente, Chirurgo}$
>- $\{\text{Chirurgo, DataIntervento}\}\to \text{Sala}$
>
>Ho come chiavi:
>- $K_{1}=\{\text{Paziente, DataIntervento}\}$
>- $K_{2}=\{\text{Chirurgo, DataIntervento, OraIntervento}\}$
>- $K_{3}=\{\text{Sala, DataIntervento, OraIntervento}\}$
>
>Qualunque sia la chiave primaria che scegliamo, i determinanti nelle prime 3 dipendenze funzionali sono insiemi di attributi che possono svolgere la funzione di chiave e quindi la BCNF non è sicuramente violata in questi casi.
>La BCNF non è invece soddisfatta per la quarta dipendenza funzionale che ha come determinante un insieme di attributi non chiave. Ne segue che la relazione $\text{Interventi}$ **non è in BCNF**
>Ma $\text{Interventi}$ è in 3NF in quanto la quarta dipendenza funzionale non viola la definizione, perché l’attributo $\text{Sala}$ è un attributo che fa parte della chiave $\{\text{Sala, DataIntervento, OraIntervento}\}$ e quindi è primo
>
>Conseguenza: la relazione $\text{Interventi}$, pur essendo in terza forma normale, presenta una certa ridondanza nei dati che può creare problemi in fase di aggiornamento.
>Se per qualche ragione si deve cambiare la sala operatoria utilizzata da in chirurgo in una certa data, bisogna aggiornate più righe: per esempio, per spostare $\text{Romano}$ dalla $\text{Sala2}$ alla $\text{Sala3}$, bisogna modificare due righe della tabella
>![[Pasted image 20241102234004.png|450]]
>
>La tabella $\text{Interventi}$ può essere normalizzata, ottenendo i due schemi:
>- $\text{OccupazioneSale(Chirurgo, DataIntervento, Sala)}$
>- $\text{Interventi(Paziente, DataInterventi, OraIntervento, Chirurgo)}$
>
>L’attributo $\text{Sala}$ viene tolto da $\text{Interventi}$ e compare in una nuova tabella che ha come chiave il determinante della dipendenza funzionale che non rispettava la BCNF
>![[Pasted image 20241102234231.png|400]]
>
>#### Problema
>Supponiamo di voler tenere traccia dei pazienti che devono essere sottoposti a più interventi chirurgici, in diversi reparti, per la cura di patologie più complicate.
>Una relazione che rappresenta questa esigenza è mostrata nella tabella sotto
>![[Pasted image 20241102234427.png|250]]
>Ogni n-upla della relazione $\text{ChirurgieMultiple}$ associa un paziente al chirurgo che lo ha operato e al reparto nel quale è avvenuto l’intervento. Valgono le dipendenze funzionali:
>- $\text{Chirurgo}\to \text{Reparto}$
>- $\{\text{Paziente, Reparto}\}\to \text{Chirurgo}$
>
>$\{\text{Paziente, Reparto}\}$ è chiave e la prima dipendenza viola Boyce-Codd. Proviamo quindi a procedere come prima
>![[Pasted image 20241102235202.png|370]]
>Questa decomposizione non conserva la seconda delle due dipendenze funzionali. Se, per esempio, si volesse registrare il fatto (errato) che il paziente Bianchi è stato operato da Lanzetta nel reparto di Cardiochirurgia, l’interimento nella tabella $\text{Paziente}$ della coppia di valori: $\text{(Bianchi, Lanzetta)}$ sarebbe consentito.
>
>Solo quando si cerca di ricostruire i dati della relazione $\text{ChirurgieMultiple}$ con un join tra $\text{Pazienti}$ e $\text{Chirurghi}$ la n-upla: $(\text{Bianchi, Lanzetta, Chir.Generale})$ poltrebbe evidenziare l’errore dei dati perché $<\text{Bianchi}-\text{Chir. Generale}>$ dovrebbe essere associato a $\text{Romano}$


