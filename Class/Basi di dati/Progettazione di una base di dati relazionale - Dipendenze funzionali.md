---
Created: 2024-10-14
Class: "[[Basi di dati]]"
Related: 
Completed:
---
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
