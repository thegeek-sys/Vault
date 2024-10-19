---
Created: 2024-10-17
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduciamo $\textcolor{Peach}{\text{F}^\text{A}}$
Ricordiamo che il nostro problema è calcolare l’insieme di dipendenze $F^+$ che viene **soddisfatto da ogni istanza legale** di uno schema $R$ su cui è definito un insieme di dipendenze funzionali $F$.
Abbiamo concluso che banalmente $F\subseteq F^+$ in quanto una istanza è legale solo se soddisfa tutte le dipendenze in $F$

---
## Assiomi di Armstrong
Denotiamo con $F^A$ l’insieme di dipendenze funzionali definito nel modo seguente:
- se $f \in F$ allora $f \in F^A$
- se rispetta l’**assioma della riflessività** (determina le dipendenze funzionali banali)
- se rispetta l’**assioma dell’aumento**
- se rispetta l’**assioma della transitività**

Dimostreremo che $\mathbf{F^+=F^A}$, cioè la chiusura di un insieme di dipendenze funzionali $F$ può essere ottenuta a partire da $F$ applicando ricorsivamente gli assiomi della riflessività, dell’aumento e della transitività, conosciuti come **assiomi di Armstrong**

### Assioma della riflessività
$$
\text{se } Y\subseteq X\subseteq R \text{ allora } X\rightarrow Y \in F^A
$$
#### Esempio
$\text{Nome}\subseteq(\text{Nome, Cognome})$ quindi ovviamente se due tuple hanno uguale la coppia $(\text{Nome, Cognome})$ allora sicuramente avranno uguale l’attributo $\text{Nome}$ (idem per $\text{Cognome}$), quindi $(\text{Nome, Cognome}) \rightarrow \text{Nome}$ viene sempre soddisfatta

### Assioma dell’aumento
$$
\text{se } X \to
$$
se $X\rightarrow Y \in F^A$ allora $XZ \rightarrow YZ \in F^A$, per ogni $Z \subseteq R$ (**assioma dell’aumento**)
$\text{CodFiscale}\rightarrow\text{Cognome}$ è soddisfatta quando, se due tuple hanno $\text{CodFiscale}$ uguale, allora hanno anche $\text{Cognome}$ uguale.
Se la dipendenza è soddisfatta, e aggiungo l’attributo $\text{Indirizzo}$, avrò che se due tuple sono uguali su $(\text{CodFiscale, Indirizzo})$ lo devono essere anche su $(\text{Cognome, Indirizzo})$ ($\text{Indirizzo}$ è incluso nella porzione di tuple che è uguale), quindi se viene soddisfatta $\text{CodFiscale}\rightarrow\text{Cognome}$ viene soddisfatta anche $\text{CodFiscale, Indirizzo}\rightarrow\text{Cognome, Indirizzo}$

se $X\rightarrow Y \in F^A$ e $Y\rightarrow Z \in F^A$ allora $X\rightarrow Z \in F^A$ (**assioma della transitività**)
$\text{Matricola}\rightarrow\text{CodFiscale}$ è soddisfatta quando, se due tuple hanno $\text{Matricola}$ uguale, allora hanno anche $\text{CodFiscale}$ uguale
$\text{CodFiscale}\rightarrow\text{Cognome}$ è soddisfatta quando, se due tuple hanno $\text{CodFiscale}$ uguale, allora hanno anche $\text{Cognome}$ uguale
Allora se entrambe le dipendenze sono soddisfatte, e due tuple hanno $\text{Matricola}$ uguale, allora hanno anche $\text{CodFiscale}$ uguale, ma allora hanno anche $\text{Cognome}$ uguale, quindi se entrambe le dipendenze sono soddisfatte, ogni volta che due tuple hanno $\text{Matricola}$ uguale avranno anche $\text{Cognome}$ uguale, e quindi viene soddisfatta anche $\text{Matricola}\rightarrow\text{Cognome}$

---
## Conseguenze degli assiomi di Armstrong
Prima di procedere introduciamo altre tre regole conseguenza degli assiomi che consentono di derivare da dipendenze funzionali in $F^A$ altre dipendenze funzionali in $F^A$

### Regola dell’unione
se $X\rightarrow Y \in F^A$ e $X\rightarrow Z \in F^A$ allora $X \rightarrow YZ \in F^A$

>[!info] Dimostrazione
>Se $X\rightarrow Y \in F^A$, per l’assioma dell’aumento si ha $X\rightarrow XY \in F^A$.
>Analogamente se $X\rightarrow Z \in F^A$, per l’assioma dell’aumento si ha $XY \rightarrow YZ \in F^A$. Quindi poiché $X\rightarrow XY \in F^A$ e $XY \rightarrow YZ \in F^A$, per l’assioma della transitività si ha $X\rightarrow YZ \in F^A$

### Regola della decomposizione
se $X \rightarrow Y \in F^A$ e $Z \subseteq Y$ allora $X\rightarrow Z \in F^A$
>[!info]- Dimostrazione
>a

### Regola della pseudotransitività
se $X \rightarrow Y \in F^A$ e $WY \rightarrow Z \in F^A$ allora $WX \rightarrow Z \in F^A$
>[!info]- Dimostrazione
