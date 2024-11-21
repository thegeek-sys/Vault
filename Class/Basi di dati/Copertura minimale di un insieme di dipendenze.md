---
Created: 2024-11-21
Class: "[[Basi di dati]]"
Related: 
Completed:
---
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