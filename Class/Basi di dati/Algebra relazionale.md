---
Created: 2024-09-29
Class: "[[Basi di dati]]"
Related: 
Completed:
---
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
Questa relazione però risulta essere sbagliata; ho infatti il $\text{Cliente}$ 