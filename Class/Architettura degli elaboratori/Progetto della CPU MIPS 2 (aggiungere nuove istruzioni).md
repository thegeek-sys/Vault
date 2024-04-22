---
Created: 2024-04-22
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
## Aggiungere una nuova istruzione
Supponiamo di voler aggiungere la nuova istruzione, J (jump), dobbiamo:
- definire la sua **codifica**
- definire **cosa faccia**
- individuare le **unità funzionali** necessarie (e se sono già presenti)
- individuare i **flussi delle informazioni** necessarie
- individuare i **segnali di controllo** necessari
- calcolare il **tempo necessario** per la nuova istruzione e se modifica il tempo totale

---
## Aggiungere il Jump
Supponiamo che abbia la codifica seguente:
![[Screenshot 2024-04-22 alle 16.44.12.png|center|500]]
Da questa codifica dobbiamo fare ulteriori supposizioni:
- il campo `indirizzo` rappresenta l’**istruzione di destinazione** del salto (va moltiplicato per 4 perché le istruzioni sono “allineate”)
- si tratta di un  **indirizzo assoluto** (invece che relativo come per i branch)
- i 4 bit “mancanti” verranno presi dal PC+4 (ovvero si rimane nello stesso blocco di 256M, per i salti tra blocchi diversi sarà necessario introdurre l’istruzione jr)

Dobbiamo quindi rispondere alle domande che ci siamo posti in precedenzz
- **Cosa fa**
	PC ← (shift left di 2 bit di Istruzione\[25-0]) OR (PC + 4)\[31-28]
- **Unità funzionali**
	PC + 4 → già presente
	shift left di 2 bit con input a 26 bit → da aggiungere
	OR dei 28 bit ottenuti con i 4 del PC+4 → si ottiene dalle connessioni
	MUX per selezionare  il nuovo PC → da aggiungere
- **Flussi dei dati**
	Istruzione\[25-0] → SL2 → (OR con i 4 MSBs dj PC+4) → MUX → PC
- **Segnali di controllo**
	Jump asserito per selezionare la nuova destinazione sul MUX
	`RegWrite=0` e `MemWrite=0` per evitare modifiche a registri e memoria
- **Tempo necessario**
	Fetch e in parallelo il tempo dell’adder che calcola PC+4 (quindi il massimo tra i due tempi)

>[!info]
>l’hardware necessario al calcolo della destinazione del salto è sempre presente e calcola la destinazione anche se l’istruzione non è un Jump. Solo se la CU riconosce che è un Jump il valore calcolato viene immesso nel PC per passare (al colpo di clock successivo) alla destinazione del salto.

