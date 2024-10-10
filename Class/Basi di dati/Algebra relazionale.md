---
Created: 2024-09-29
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Index
- [[#Introduction|Introduction]]
- [[#Proiezione|Proiezione]]
	- [[#Proiezione#Esempio|Esempio]]
- [[#Selezione|Selezione]]
	- [[#Selezione#Esempio|Esempio]]
- [[#Unione|Unione]]
	- [[#Unione#Esempi|Esempi]]
		- [[#Esempi#1.|1.]]
		- [[#Esempi#2|2]]
		- [[#Esempi#3|3]]
		- [[#Esempi#4|4]]
- [[#Differenza|Differenza]]
	- [[#Differenza#Esempio|Esempio]]
- [[#Intersezione|Intersezione]]
	- [[#Intersezione#Esempio|Esempio]]
- [[#Informazioni in più relazioni|Informazioni in più relazioni]]
- [[#Prodotto cartesiano|Prodotto cartesiano]]
	- [[#Prodotto cartesiano#Esempio|Esempio]]
- [[#Join naturale|Join naturale]]
	- [[#Join naturale#Esempio 1|Esempio 1]]
	- [[#Join naturale#Esempio 2|Esempio 2]]
		- [[#Esempio 2#Alternativa|Alternativa]]
	- [[#Join naturale#Casi limite|Casi limite]]
	- [[#Join naturale#Possibili errori|Possibili errori]]
- [[#θ-join|θ-join]]
- [[#Condizioni negative|Condizioni negative]]
---
## Introduction
L’algebra relazione è un linguaggio **formale** per interrogare un database relazionale: consiste di un insieme di operatori che possono essere applicati a una (operatori unari) o due (operatori binari) istanze di relazione e forniscono come risultato una nuova istanza di relazione (che può essere “salvata” in una ”variabile”).
Ma è anche un linguaggio **procedurale** in quando l’interrogazione consiste in un’espressione in cui compaiono operatori dell’algebra e istanze di relazioni della base di dati, in una sequenza che prescrive in maniera puntuale l’ordine delle operazioni e i loro operandi

---
## Proiezione
La **proiezione** consente di effettuare un taglio verticale su una relazione cioè di selezionare solo alcune colonne (attributi).
$$
\pi_{\text{A1, A2, } \dots, \text{ Ak}}(r)
$$
Seleziona quindi le colonne di $r$ che corrispondono agli attributi $\text{A1, A2, }\dots, \text{Ak}$

### Esempio
![[Screenshot 2024-09-26 alle 15.40.18.png|center|500]]

> [!warning] Attenzione
> Si seguono le regole insiemistiche. Nella relazione risultato **non** ci sono **duplicati**.
> Se vogliamo conservare i clienti omonimi dobbiamo aggiungere un ulteriore attributo in questo caso la **chiave**
> ![[Screenshot 2024-09-26 alle 15.42.26.png|400]]

---
## Selezione
La **selezione** consente di effettuare un “taglio orizzontale” su una relazione, cioè di selezionare solo le righe (tuple) che soddisfano una data condizione
$$
\sigma_{\text{C}}(r)
$$
Seleziona le tuple di $r$ che soddisfano la condizione $\text{C}$ la quale è un’espressione booleana composta in cui i termini semplici sono del tipo:
- $\text{A }\theta \text{ B}$
- $\text{A }\theta \text{ 'nome'}$
dove:
- $\theta$ → un operatore di confronto ($\theta \in \{<, =, >, \leq, \geq\}$)
- A e B → due attributi con lo stesso dominio ($\text{dom(A) = dom(B)}$)
- nome → un elemento di $\text{dom(A)}$ (costante o espressione)

### Esempio
![[Screenshot 2024-09-29 alle 17.35.02.png|center|550]]


---
## Unione
L’**unione** serve a costruire una relazione contenente tutte le ennuple che appartengono ad almeno uno dei due operandi
$$
r_{1} \cup r_{2}
$$

> [!warning]
> L’unione può essere applicata a due istanze **union compatibili**, ovvero solo se:
> 1. hanno lo stesso numero di attributi
> 2. gli attributi ordinatamente (corrispondenti) sono definiti sullo stesso dominio
> 3. ordinatamente hanno lo stesso significato (es. matricola ≠ numero di telefono)

### Esempi
#### 1.
![[Screenshot 2024-10-02 alle 15.37.22.png|440]]
sono union compatibili
$$
\text{Personale}=\text{Docenti}\cup \text{Amministrativi}
$$
![[Screenshot 2024-10-02 alle 15.38.51.png|440]]

#### 2
![[Screenshot 2024-10-02 alle 15.43.21.png|440]]
In questo caso non posso fare l’unione (in $\text{Amministrativi}$ ci sta un attributo in più). Per risolvere dunque devo prima fare una proiezione per poter poi fare l’unione. (non era necessario fare la proiezione sui docenti)
$$
\text{Personale}=\text{Docenti}\cup \pi_{\text{Nome, CodDoc, Dipartimento}}(\text{Amministrativi})
$$

#### 3
![[Screenshot 2024-10-02 alle 15.49.56.png|440]]
In questo esempio non è possibile unire le due relazioni in quanto non sono union compatibili (attributi corrispondenti sono definiti su domini diversi $\text{Dipartimento}$ e $\text{AnniServizio}$). Devo per questo fare una proiezione su entrambe le relazioni
$$
\text{Personale} = \pi_{\text{Nome, CodDoc}}(Docente)\cup \pi_{\text{Nome,  CodAmm}}(\text{Amministrativi})
$$

#### 4
![[Screenshot 2024-10-02 alle 15.53.36.png|440]]
In questo esempio le due relazioni sono union compatibili ma gli attributi anche se definiti sugli stessi domini hanno un significato diverso ($\text{Dipartimento}$ e $\text{Mansioni}$). Devo dunque fare una proiezione su entrambe le relazioni
$$
\text{Personale} = \pi_{\text{Nome, CodDoc}}(Docente)\cup \pi_{\text{Nome,  CodAmm}}(\text{Amministrativi})
$$

---
## Differenza
La **differenza** consente di costruire una relazione contentente tutte le tuple che appartengono al primo operando e non appartengono al secondo operando e si applica a operandi union compatibili
$$
r_{1}-r_{2}
$$
>[!warning] La differenza non è commutativa

### Esempio
![[Screenshot 2024-10-02 alle 16.00.34.png|440]]
$$
\text{Studenti}-\text{Amministrativi}=\text{studenti  che non sono anche amministrativi}
$$
$$
\text{Amministrativi} - \text{Studenti} = \text{amministrativi che non sono anche studenti}
$$
![[Screenshot 2024-10-02 alle 16.03.03.png|440]]

Nascerebbe però un problema se avessi degli studenti che sono amministrativi in dipartimenti diversi da quelli in cui studiano (e viceversa). In questo caso infatti dovremmo fare una proiezione su $\text{Nome}$ e $\text{CodFiscale}$ per poter avere gli stessi risultati

---
## Intersezione
 L’intersezione consente di costruire una relazione contenente tutte le tuple che appartengono  ad entrambi gli operandi e si applica a operandi union compatibili.
 $$
r_{1}\cap r_{2}=(r_{1}-(r_{1}-r_{2}))
$$

### Esempio
![[Screenshot 2024-10-02 alle 16.11.54.png|440]]
$$
\text{Studenti}\cap \text{Amministrativi} = \text{studenti che sono anche amministrativi}
$$
![[Screenshot 2024-10-02 alle 16.13.14.png|440]]

---
## Informazioni in più relazioni
Vedremo che per garantire determinate ”buone” qualità di una relazione occorre rappresentare **separatamente** (in relazioni diverse) **concetti diversi**
Capita che molto spesso che le informazioni che interessano per rispondere ad una interrogazione sono **distribuite** in più relazioni, in quanto coinvolgono **più oggetti** in qualche modo associati. Occorre quindi individuare le relazioni in cui si trovano le informazioni che ci interessano, e combinare queste informazioni in maniera opportuna

---
## Prodotto cartesiano
Il **prodotto cartesiano** permette di costruire una relazione che contiene tutte le ennuple ottenute unendo tutte le ennuple di una relazione e tutte le ennuple di una seconda relazione
$$
r_{1}\times r_{2}
$$
Si usa quando le informazioni che occorrono a rispondere a una query si trovano in **relazioni diverse**

> [!warning] Non sempre il prodotto cartesiano ha un significato

### Esempio
![[Screenshot 2024-10-02 alle 16.37.52.png|440]]
In questo caso però non posso fare direttamente $\text{Cliente}\times \text{Ordine}$ in quanto ho un attributo identico nelle due relazioni. Per questo motivo abbiamo necessità di utilizzare la **ridenominazione** ($\rho$)
$$
\text{OrdineR}=\rho_{\text{CC\#} \leftarrow \text{C\#}}(\text{Ordine})
$$
Dunque posso fare:
$$
\text{Dati dei clienti e degli ordini}=(\text{Cliente}\times \text{OrdineR})
$$
![[Screenshot 2024-10-02 alle 16.42.00.png|440]]
Questa relazione però risulta essere sbagliata; ho infatti la seconda tupla ha due codici diversi per $\text{Cliente}$ e $\text{Ordine}$. Il che vuol dire che devo effettuare una selezione sul codice cliente
$$
\sigma_{\text{C\#}=\text{CC\#}}(\text{Cliente}\times \text{OrdineR})
$$
![[Screenshot 2024-10-02 alle 20.06.39.png|440]]
A questo punto mi ritrovo sostanzialmente con un attributo duplicato. Lo rimuovo quindi con una proiezione
$$
\pi_{\text{Nome, C\#, Città, O\#, A\#, N-pezzi}}(\sigma_{\text{C\#}=\text{CC\#}}(\text{Cliente}\times \text{OrdineR}))
$$
![[Screenshot 2024-10-02 alle 20.08.46.png|440]]
Adesso voglio rimuovere tutti gli elementi in cui il $\text{N-pezzi}$ è ≤ a 100
$$
\pi_{\text{Nome C\# Città O\# A\# N-pezzi}}(\sigma_{\text{C\#=CC\# }\land \text{ N-pezzi}>100}(\text{Cliente}\times \text{OrdineR}))
$$
![[Screenshot 2024-10-02 alle 20.10.02.png|440]]

---
## Join naturale
Il join naturale consente di selezionare automatiche le tuple del prodotto cartesiano dei due operandi che soddisfano la condizione
$$
(R_{1}. A_{1}=R_{2}. A_{1}) \land (R_{1}. A_{2}=R_{2}. A_{2})\land \dots \land (R_{1}.A_{k} = R_{2}. A_{k})
$$
(dove $R_{1}$ ed $R_{2}$ sono i nomi delle relazioni operando e $A_{1}, A_{2}, \dots,A_{k}$ sono gli attributi comuni, cioè **con lo stesso nome**, delle relazioni operando) eliminando le ripetizioni degli attributi
$$
r_{1}\bowtie r_{2} = \pi_{\text{XY}}(\sigma_{\text{C}}(r_{1}\times r_{2}))
$$
dove:
- $\text{C}$ → $(R_{1}. A_{1}=R_{2}. A_{1})\land \dots \land (R_{1}. A_{k} = R_{2}. A_{k})$
- $\text{X}$ → è l’insieme di attributi di $r_{1}$
- $\text{Y}$ → insieme di attributi di $r_{2}$ che non sono attributi di $r_{1}$

> [!info]
> Nel join naturale gli attributi della condizione che consente di unire solo le ennuple giuste che hanno lo **stesso nome**
> Vengono unite le ennuple in cui questi attributi hanno lo **stesso valore**
### Esempio 1
![[Screenshot 2024-10-02 alle 16.58.41.png|440]]
$$
\text{Dati dei clienti e dei loro ordini} = \text{Cliente}\bowtie \text{Ordine}
$$
![[Screenshot 2024-10-02 alle 17.00.21.png|440]]
Adesso voglio rimuovere tutti gli elementi in cui il $\text{N-pezzi}$ è ≤ a 100
$$
\sigma_{\text{N-pezzi}>100}(\text{Cliente}\bowtie \text{Ordine})
$$
![[Screenshot 2024-10-02 alle 20.37.48.png|440]]
Come prima ma stavolta voglio solo i nomi dei clienti
$$
\pi_{\text{Nome}}(\sigma_{\text{N-pezzi}>100}(\text{Cliente}\bowtie \text{Ordine}))
$$
![[Screenshot 2024-10-02 alle 20.40.10.png|250]]
>[!warning]
>Dato che $\text{Nome}$ non identifica il cliente, i duplicati vengono cancellati (in questo caso ho solo un $\text{Rossi}$). Sarebbe stato meglio utilizzare anche una **chiave** ($\text{C\#}$)

Adesso oltre che il nome voglio anche vedere la città
$$
\pi_{\text{Nome, Città}}(\sigma_{\text{N-pezzi}>100}(\text{Cliente}\bowtie \text{Ordine}))
$$
![[Screenshot 2024-10-03 alle 13.15.23.png|300]]

### Esempio 2
![[Screenshot 2024-10-07 alle 10.52.54.png|440]]
Nomi e città dei clienti che hanno ordinato più di 100 pezzi per almeno un articolo con prezzo superiore a 2
$$
\pi_{\text{Nome, Città}}(\sigma_{\text{N-pezzi}>100\land \text{Prezzo}>2}((\text{Cliente}\bowtie \text{Ordine})\bowtie \text{Articolo}))
$$
![[Screenshot 2024-10-07 alle 11.00.19.png|200]]
#### Alternativa
In alternativa prima potremmo fare selezione sugli ordini con numero di pezzi maggiore di 100
$$
\sigma_{\text{N-pezzi}>100}(\text{Ordine})
$$
Quindi fare il join con $\text{Cliente}$
$$
\text{Cliente}\bowtie \sigma_{\text{N-pezzi}>100}(\text{Ordine})
$$
Per quanto riguarda il prezzo invece mi conviene fare una proiezione sul numero dell’articolo e sul prezzo per poi selezionare quelli con un prezzo maggiore di due
$$
\sigma_{\text{Prezzo}>2}(\pi_{\text{A\#, Prezzo}}(\text{Articolo}))
$$
E solo ora fare il join naturale con il resto:
$$
(\text{Cliente}\bowtie \sigma_{\text{N-pezzi}>100}(\text{Ordine}))\bowtie \sigma_{\text{Prezzo}>2}(\pi_{\text{A\#, Prezzo}}(\text{Articolo}))
$$
Infine fare la proiezione su $\text{Nome}$ e $\text{Città}$
$$
\pi_{\text{Nome, Città}}((\text{Cliente}\bowtie \sigma_{\text{N-pezzi}>100}(\text{Ordine}))\bowtie \sigma_{\text{Prezzo}>2}(\pi_{\text{A\#, Prezzo}}(\text{Articolo})))
$$

### Casi limite
Per quanto riguarda il join naturale sono presenti due casi limite:

**Caso limite 1**
Quando le relazioni contengono attributi con lo stesso nome ma non esistono ennuple con lo stesso valore per tali attributi in entrambe le relazioni, il risultato del join naturale è **vuoto**

Per esempio:
![[Screenshot 2024-10-07 alle 11.13.36.png|550]]
il risultato è vuoto

**Caso limite 2**
Quando le relazioni non contengono contengono attributi con lo stesso nome, il join naturale degenera nel **prodotto cartesiano**
![[Screenshot 2024-10-07 alle 11.17.05.png|440]]
$$
\text{Cliente}\bowtie \text{Ordine}
$$
![[Screenshot 2024-10-07 alle 11.19.53.png|500]]

### Possibili errori
Ovviamente perché il join naturale abbia senso gli attributi devono avere lo stesso significato
![[Screenshot 2024-10-03 alle 13.53.04.png|440]]
Perché il join tra queste due cose abbia senso, il join va effettuato tra $\text{Artista.C\#}$ e $\text{Quadro.Artista}$. Quindi posso utilizzare o il θ-join oppure la ridenominazione
Procedendo con la ridenominazione:
$$
\rho_{\text{CA\#}\leftarrow \text{C\#}}(\text{Artista})\bowtie \rho_{\text{CA\#}\leftarrow \text{Artista}}(\text{Quadro})
$$

---
## θ-join
Il **θ-join** consente di selezionare le tuple del prodotto cartesiano dei due operandi che soddisfano una condizione del tipo $\text{A}\theta \text{B}$
dove:
- θ è un operatore di confronto ($\theta \in \{<, =, >, \leq, \geq\}$)
- A è un attributo dello schema del primo operando
- B è un attributo dello schema del secondo operando
- $\text{dom(A)}=\text{dom(B)}$

$$
r_{1}\underset{\text{A}\theta \text{B}}{\bowtie}r_{2}=\sigma_{\text{A}\theta \text{B}}(r_{1}\times r_{2})
$$

---
## Condizioni negative
Alle query si possono anche applicare delle condizioni negative
![[Screenshot 2024-10-07 alle 11.28.28.png|440]]
$$
\sigma_{\neg(\text{Città}=\text{"Roma"})\land \text{Nome}=\text{"Rossi"}}(\text{Cliente})
$$

---
## Condizioni che implicano il quantificatore  universale
Fino ad ora abbiamo visto query che implicavano condizioni equivalenti al quantificatore esistenziale $\exists$ (esiste almeno un)
Il meccanismo di valutazione consente di rispondere facilmente a questo tipo di query infatt **in qualunque posizione** appaiono nell’espressione di algebra relazionale, la valutazione delle **condizioni** avviene in **sequenza**, tupla per tupla, e quando si incontra una tupla che soddisfa le condizioni, questa viene inserita nel risultato (eventualmente parziale)

La condizione potrebbe richiedere la valutazione di gruppi interi di tuple prima di decidere se inserirle tutte, qualcuna o nessuna nella risposta e le tuple non sono ordinate e la valutazione avviene in sequenza tupla per tupla, e una volta inserita una tupla nel risultato non possiamo più eliminarla. In questo caso la condizione equivale a valutare il quantificatore universale $\forall$ (per ogni) oppure $!\exists$ (non esiste nessun)
### Esempio 1
Nomi e città dei clienti che hanno **SEMPRE** ordinato più di 100 pezzi per articolo
![[Screenshot 2024-10-09 alle 21.59.37.png|440]]
Visto che risulterebbe complesso fare una selezione in cui ho solo i clienti che hanno ordinato sempre più di 100 pezzi, mi conviene piuttosto trovare quelli che hanno fatto ordini con numero di pezzi minori di 100 ed escluderli

$$
\sigma_{\text{N-pezzi}\leq 100}(\text{Cliente}\bowtie\text{Ordine})
$$
Facciamo quindi la proiezione sul nome e città
$$
\pi_{\text{Nome, Città}}(\sigma_{\text{N-pezzi}\leq 100}(\text{Cliente}\bowtie\text{Ordine}))
$$
Solo a questo punto posso fare la differenza con il totale
$$
\pi_{\text{Nome, Città}}(\text{Cliente}\bowtie\text{Ordine})-\pi_{\text{Nome, Città}}(\sigma_{\text{N-pezzi}\leq 100}(\text{Cliente}\bowtie\text{Ordine}))
$$

> [!warning] Attenzione
> Nel primo membro della sottrazione faccio il join naturale tra $\text{Cliente}$ e $\text{Ordine}$ in modo tale da poter togliere tutti i casi in cui sono presenti clienti che non hanno mai fatto ordini. Il join infatti assicura di effettuare la sottrazione non a partire da tutti i clienti, ma solo da quelli che hanno effettuato almeno un ordine

### Esempio 2
Nomi e città dei clienti che non hanno **MAI** ordinato più di 100 pezzi per un articolo
![[Screenshot 2024-10-09 alle 21.59.37.png|440]]
Applichiamo il ragionamento di prima e selezioniamo prima i nomi e città
di clienti che NON ci interessano
$$
\sigma_{\text{N-pezzi}>100}(\text{Cliente}\bowtie\text{Ordine})
$$
Facciamo la proiezione sul nome e città
$$
\pi_{\text{Nome, Città}}(\sigma_{\text{N-pezzi}>100}(\text{Cliente}\bowtie\text{Ordine}))
$$
Posso quindi fare la differenza
$$
\pi_{\text{Nome, Città}}(\text{Cliente}\bowtie\text{Ordine})-\pi_{\text{Nome, Città}}(\sigma_{\text{N-pezzi}>100}(\text{Cliente}\bowtie\text{Ordine}))
$$

---
## Condizioni che richiedono il prodotto di una relazione con sé stessa
Come negli esempi precedenti abbiamo visto casi in cui oggetti di relazioni diverse vengono associati, ci sono anche casi in cui sono in qualche modo associati oggetti della stessa relazione.

### Esempio 1
Nomi e codici degli impiegati che guadagnano quanto o più del loro capo
![[Screenshot 2024-10-10 alle 08.50.51.png|440]]
Per poter confrontare le informazioni sullo stipendio di un impiegato e su quello del suo capo che si trovano in tuple diverse questi devono trovarsi nella stessa tupla. Per farlo creiamo una copia della relazione  ed effettuiamo un prodotto in maniera da combinare le informazioni su di un impiegato con quelle del suo capo, che a questo punto possono essere confrontate. $\text{ImpiegatiC}$ sarà collegata in join ad impiegati combinando le tuple col valore di $\text{C\#}$ uguale a $\text{Capo\#}$. In questo modo accodiamo i dati del capo a quelli dell’impiegato.
Utilizziamo la ridenominazione e facciamo in modo che i nuovi nomi aiutino a distinguere il ruolo delle due parti nel join finale
$$
\text{ImpiegatiC} = \rho_{\text{Nome, C\#, Dipart, Stip, Capo\#}\leftarrow\text{CNome, CC\#, Cdipart, Cstip, Ccapo\#}}(\text{Impiegati})
$$
$$
\sigma_{\text{Capo\#}=\text{CC\#}}(\text{Impiegati}\times \text{ImpiegatiC})
$$
A questo punto basta confrontare lo stipendio dell’impiegato con quello del capo per selezionare gli impiegati che ci interessano e infine proiettare
$$
\text{r} =\sigma_{\text{Stip}\geq \text{CStip}}(\sigma_{\text{Capo\#}=\text{CC\#}}(\text{Impiegati}\times \text{ImpiegatiC}))
$$
$$
\pi_{\text{Nome, C\#}}(\text{r})
$$

### Esempio 2
**Query:** Nomi e codici dei capi che guadagnano più di tutti i loro impiegati
Ripriendiamo la query dell’esempio precedente che trova gli impiegati che guadagnano quanto o più del loro capo. I capi che compaiono anche una sola volta in questo risultato sono quelli che non ci interessano
$$
\text{r} =\sigma_{\text{Stip}\geq \text{CStip}}(\sigma_{\text{Capo\#}=\text{CC\#}}(\text{Impiegati}\times \text{ImpiegatiC}))
$$
$$
\pi_{\text{CNome, CC\#}}(\sigma_{\text{Capo\#}=\text{CC\#}}(\text{Impiegati}\times \text{ImpiegatiC}))-\pi_{\text{CNome,CC\#}}(r)
$$

#### Errori possibili
Lo stesso esercizio può essere svolto in altri modi altrettanto corretti, ma attenzione invece a quelli non corretti
$$
\pi_{\text{Nome, C\#}}(\text{Impiegati})-\pi_{\text{CNome, CC\#}}(\text{r})
$$
è sbagliato perché ci sono impiegati che non sono capi e non verrebbero sottratti alla prima proiezione

$$
\pi_{\text{Nome, Capo\#}}(\text{Impiegati})-\pi_{\text{CNome, CC\#}}(\text{r})
$$
è sbagliato perché nella prima proiezione il nome è dell’impiegato e il codice è del capo

Un’alternativa corretta è:
$$
\pi_{\text{Nome, C\#}}((\pi_{\text{Capo\#}}(\text{Impiegati})-\pi_{\text{CC\#}}(\text{r}))\bowtie \text{Impiegati})
$$
che estrae prima i codici, effettua un join per ottenere le altre informazioni (un capo è anche un impiegato) e poi effettua la proiezione. Vogliamo che i codici da cui sottraiamo siano sicuramente di capi