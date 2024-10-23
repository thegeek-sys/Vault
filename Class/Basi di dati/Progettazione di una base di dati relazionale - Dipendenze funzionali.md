---
Created: 2024-10-14
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Index
- [[#Tupla|Tupla]]
- [[#Istanza di relazione|Istanza di relazione]]
- [[#Dipendenze funzionali|Dipendenze funzionali]]
	- [[#Dipendenze funzionali#Nota|Nota]]
	- [[#Dipendenze funzionali#Esempio|Esempio]]
- [[#Istanza legale|Istanza legale]]
	- [[#Istanza legale#Osservazione|Osservazione]]
- [[#Esempio|Esempio]]
- [[#Chiusura di un insieme di dipendenze funzionali|Chiusura di un insieme di dipendenze funzionali]]
	- [[#Chiusura di un insieme di dipendenze funzionali#$\textcolor{lightgreen}{\text{F}}$ ed $\textcolor{lightgreen}{\text{F}^+}$|$\textcolor{lightgreen}{\text{F}}$ ed $\textcolor{lightgreen}{\text{F}^+}$]]
- [[#Chiave|Chiave]]
	- [[#Chiave#Esempio|Esempio]]
	- [[#Chiave#Chiave primaria|Chiave primaria]]
- [[#Dipendenze funzionali banali|Dipendenze funzionali banali]]
	- [[#Dipendenze funzionali banali#Dipendenze funzionali (proprietà)|Dipendenze funzionali (proprietà)]]
---
## Schema di relazione
Uno **schema di relazione** R è un insieme di attributi $\{A_{1}, A_{2}, \dots, A_{n}\}$
Notazione:
- $\text{R}=A_{1}, A_{2}, \dots A_{n}$
- le prime lettere dell’alfabeto ($\text{A,B,C,}\dots$) denotano i singoli attributi
- le ultime lettere dell’alfabeto ($\text{X, Y,}\dots$) denotano insiemi di attributi
- Se $\text{X}$ e $\text{Y}$ sono insiemi di attributi $\text{XY}$ denota $X\cup Y$

---
## Tupla
Dato uno schema di relazione $\text{R}=A_{1}, A_{2}, \dots A_{n}$ una **tupla** $\text{t}$ su $\text{R}$ è una funzione che associa ad ogni attributo $A_{i}$ in $\text{R}$ un valore $\text{t}[A_{i}]$ nel corrispondente dominio $\text{dom}(A_{i})$
![[Pasted image 20241014140836.png|450]]

Se $\text{X}$ è un sottoinsieme di $\text{R}$ e $t_{1}$ e $t_{2}$ sono due tuple su $\text{R}$, $t_{1}$ e $t_{2}$ coincidono su $\text{X}$ ($t_{1}[X]=t_{2}[X]$) se $\forall A \in X \,(t_{1}[A]=t_{2}[A])$
![[Pasted image 20241014141231.png|450]]

---
## Istanza di relazione
Dato uno schema di relazione $\text{R}$ una **istanza** di $\text{R}$ è un insieme di tuple su $\text{R}$

>[!info]
>Tutte le “tabelle” che abbiamo visto finora negli esempi sono istanze di qualche schema di relazione

---
## Dipendenze funzionali
Dato uno schema di relazione $\text{R}$ una **dipendenza funzionale** su $\text{R}$ è una coppia ordinata di sottoinsiemi non vuoti $\text{X}$ ed $\text{Y}$ di $\text{R}$
Notazione:
- $\text{X} \rightarrow \text{Y}$ → si legge X determina funzionalmente Y oppure Y dipende funzionalmente da X
- $\text{X}$ → parte sinistra della dipendenza o determinante
- $\text{Y}$ → parte destra della dipendenza o dipendente

Dati uno schema $\text{R}$ e una dipendenza funzionale $\text{X}\rightarrow\text{Y}$ su $\text{R}$ un’istanza $r$ di $\text{R}$ **soddisfa** la dipendenza funzionale $\text{X}\rightarrow\text{Y}$ se:
$$
\forall t_{1},t_{2}\in r \,(t_{1}[X]=t_{2}[X]\rightarrow t_{1}[Y]=t_{2}[Y])
$$

> [!info] Le dipendenze funzionali non fanno altro che esprimere dei vincoli sui dati

### Nota
Nella relazione che rappresenta gli esami, non abbiamo $\text{Voto}\rightarrow\text{Lode}$ perché se $t_{1}[\text{Voto}]=t_{2}[\text{Voto}]=27$ allora sicuramente deve essere $t_{1}[\text{Lode}]=t_{2}[\text{Lode}]=\text{'No'}$.
Ma se $t_{1}[\text{Voto}]=t_{2}[\text{Voto}]=30$ e $t_{1}[\text{Lode}]=\text{'Si'}$ questo non determina il valore che deve avere $t_{2}[\text{Lode}]$ (può essere $\text{'Si'}$ oppure $\text{'No'}$ senza compromettere la correttezza del dato)

E’ possibile dire che $\text{Lode} \rightarrow \text{Voto}$?
Se $t_{1}[\text{Lode}]=t_{2}[\text{Lode}]=\text{'Si'}$ allora sicuramente deve essere $t_{1}[\text{Voto}]=t_{2}[\text{Voto}]=30$.
Ma se $t_{1}[\text{Lode}]=t_{2}[\text{Lode}]=\text{'No'}$ e $t_{1}[\text{Voto}]=27$ questo non determina il valore che deve avere $t_{2}[\text{Voto}]$ (può essere un qualsiasi voto tra $18$ e $30$)

### Esempio
![[Pasted image 20241014142931.png|300]]
**Soddisfa** da dipendenza funzionale $\text{AB} \rightarrow \text{C}$

![[Pasted image 20241014143009.png|300]]
**Non soddisfa** la dipendenza funzionale $\text{AB} \rightarrow \text{C}$

---
## Istanza legale
Dati uno schema di relazione $\text{R}$ e un insieme $\text{F}$ di dipendenze funzionali, un’istanza di $\text{R}$ è **legale** se soddisfa **tutte** le dipendenze in $\text{F}$

### Osservazione
![[Pasted image 20241010155529.png|400]] 
L’istanza soddisfa la dipendenza funzionale $\text{A} \rightarrow \text{B}$ (e quindi è un’istanza legale) e anche la dipendenza funzionale $\text{A}\rightarrow\text{C}$ ma $\text{A}\rightarrow\text{C}$ non è in $\text{F}$ e non è detto che debba sempre essere soddisfatta

![[Pasted image 20241010155714.png|400]]
La nuova istanza soddisfa la dipendenza funzionale $\text{A}\rightarrow\text{B}$ (e quindi è anch’essa un’istanza legale) ma non soddisfa la dipendenza funzionale $\text{A}\rightarrow\text{C}$, d’altra parte $\text{A}\rightarrow\text{C}$ non è in F quindi perché dovrebbe essere comunque sempre soddisfatta?

![[Pasted image 20241014143259.png|400]]
Ogni istanza legale (cioè ogni istanza che soddisfa sia $\text{A}\rightarrow\text{B}$ che $\text{B}\rightarrow\text{C}$ soddisfa sempre anche la dipendenza funzionale $\text{A}\rightarrow\text{C}$). Possiamo considerarla allora “come se fosse in $\text{F}$”?

Dunque dato uno schema di relazione $\text{R}$ e un insieme $\text{F}$ di dipendenze funzionali su $\text{R}$ ci sono delle dipendenze funzionali **che non sono in $\text{F}$**, ma che **sono soddisfatte da ogni istanza legale di $\text{R}$**

---
## Esempio

$$
\text{Matricola} \rightarrow \text{CodiceFiscale} \rightarrow \text{DataNascita}
$$
devono essere sempre soddisfatte da ogni istanza legale ma allora sarà sempre soddisfatta anche $\text{Matricola}\rightarrow\text{DataNascita}$


$$
\text{CodiceFiscale}\rightarrow\text{Nome, Cognome}
$$
deve essere soddisfatta da ogni istanza legale ma allora saranno sempre soddisfatte anche:
- $\text{CodiceFiscale}\rightarrow\text{Nome}$
- $\text{CodiceFiscale}\rightarrow\text{Cognome}$

---
## Chiusura di un insieme di dipendenze funzionali
Dato uno schema di relazione $\text{R}$ e un insieme $\text{F}$ di dipendenze funzionali su $\text{R}$ la **chiusura di $\text{F}$** è l’insieme delle dipendenze funzionali che sono soddisfatte da ogni istanza legale di $\text{R}$
Notazione:
- $\text{F}^+$

### $\text{F}$ ed $\text{F}^+$
Se $\text{F}$ è un insieme di dipendenze funzionali su $\text{R}$ ed $r$ è un’istanza di $\text{R}$ che soddisfa **tutte** le dipendenze in $\text{F}$, diciamo che $r$ è un’**istanza legale** di $\text{R}$
La chiusura di $\text{F}$, denotata con $\text{F}^+$, è l’insieme di dipendenze funzionali che sono soddisfatte **da ogni** istanza legale di $\text{R}$
Banalmente si ha che $\text{F}\subseteq \text{F}^+$

Due insiemi di dipendenze funzionali che hanno la stessa chiusura avranno le stesse istanze legali
$$
F\subseteq F^+=G^+\supseteq G
$$

---
## Chiave
Dati uno schema di relazione $\text{R}$ e un insieme $\text{F}$ di dipendenze funzionali, un sottoinsieme $\text{K}$ di uno schema di relazione $\text{R}$ è una **chiave** se $\text{K}\rightarrow\text{R}\in \text{F}^+$ e non esiste un sottoinsieme proprio $\text{K}'$ di $\text{K}$ tale che $\text{K}'\rightarrow\text{R}\in \text{F}^+$

### Esempio
Consideriamo lo schema
$$
\text{Studente=Matr, Cognome, Nome, Data}
$$
Il numero di matricola viene assegnato allo studente per identificarlo
$$
\Downarrow
$$
Quindi non i possono essere due studenti con lo stesso numero di matricola
$$
\Downarrow
$$
Quindi un’istanza di $\text{Studente}$ per rappresentare correttamente la realtà non può contenere due tuple con lo stesso numero di matricola
$$
\Downarrow
$$
Quindi $\text{Matr}\rightarrow \text{Matr, Cognome, Nome, Data}$ deve essere soddisfatta da ogni istanza legale
$$
\Downarrow
$$
Quindi $\text{Matr}$ è una chiave per $\text{Studente}$

### Chiave primaria
Dati uno schema di relazione $\text{R}$ e un insieme $\text{F}$ di dipendenze funzionali, possono esistere più chiavi di $\text{R}$. In SQL una di esse verrà scelta come **chiave primaria** (non può assumere valore nullo)

ESEMPIO: $\text{Studente}=\text{Matr, }\textbf{CF}\text{, Cognome, Nome, Data}$
Se prendiamo $\textbf{CF}$ come chiave primaria $\text{Matr}$ deve essere UNIQUE

---
## Dipendenze funzionali banali
Dati uno schema di relazione $\text{R}$ e due sottoinsiemi non vuoti $\text{X, Y}$ di $\text{R}$ tali che $\text{Y}\subseteq \text{X}$ si ha che ogni istanza $r$ di $\text{R}$ soddisfa la dipendenza funzionale $\text{X}\rightarrow\text{Y}$
![[Pasted image 20241016160527.png|440]]

Pertanto se $\text{Y}\subseteq \text{X}$ allora $\text{X}\rightarrow\text{Y}\in \text{F}^+$
Una tale dipendenza funzionale è detta **banale**

### Dipendenze funzionali (proprietà)
Dati uno schema di relazione $\text{R}$ e un insieme di dipendenze funzionali $\text{F}$, si ha:
$$
\text{X}\rightarrow\text{Y}\in \text{F}^+ \Leftrightarrow \forall \text{A} \in \text{Y}\, (\text{X} \rightarrow \text{A} \in \text{F}^+)
$$
$\text{X} \rightarrow \text{Y}$ deve essere soddisfatta sa **ogni** istanza legale di $\text{R}$
- se $t_{1}[\text{X}]=t_{2}[\text{X}]$ allora deve essere $t_{1}[\text{Y}]=t_{2}[\text{Y}]$
- ovviamente se $\text{A}\in\text{Y}$ e $t_{1}[\text{A}]\neq t_{2}[\text{A}]$, non può essere $t_{1}[\text{Y}]=t_{2}[\text{Y}]$
- ovviamente se $\forall\text{A}\in\text{Y}\, t_{1}[\text{A}]= t_{2}[\text{A}]$, avremo $t_{1}[\text{Y}]=t_{2}[\text{Y}]$
![[Pasted image 20241017004420.png|440]]