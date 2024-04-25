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
- i 4 bit “mancanti” verranno presi dal PC+4 (ovvero si rimane nello stesso blocco di 256M, per i salti tra blocchi diversi sarà necessario introdurre l’istruzione jr). Infatti i 4 MSBs del PC indicano in quale dei 16 blocchi (da 256M, 4G totali) della RAM ci si trova.

Dobbiamo quindi rispondere alle domande che ci siamo posti in precedenza
- **Cosa fa**
	- PC ← (shift left di 2 bit di Istruzione\[25-0]) OR (PC + 4)\[31-28]
- **Unità funzionali**
	- PC + 4 → presente
	- shift left di 2 bit con input a 26 bit → da aggiungere
	- OR dei 28 bit ottenuti con i 4 del PC+4 → si ottiene dalle connessioni
	- MUX per selezionare  il nuovo PC → da aggiungere
- **Flussi dei dati**
	- `Istruzione[25-0] → SL2 → (OR con i 4 MSBs di PC+4) → MUX → PC`
- **Segnali di controllo**
	- Jump asserito per selezionare la nuova destinazione sul MUX
	- `RegWrite=0` e `MemWrite=0` per evitare modifiche a registri e memoria
- **Tempo necessario**
	- Fetch e in parallelo il tempo dell’adder che calcola PC+4 (quindi il massimo tra i due tempi)

>[!info]
>l’hardware necessario al calcolo della destinazione del salto è sempre presente e calcola la destinazione anche se l’istruzione non è un Jump. Solo se la CU riconosce che è un Jump il valore calcolato viene immesso nel PC per passare (al colpo di clock successivo) alla destinazione del salto.

![[Screenshot 2024-04-22 alle 17.11.08.png]]

---
## Aggiungere il Jump and Link
Stessa codifica dell’istruzione Jump
- **Cosa fa**
	- PC ← (shift left di 2 bit di Istruzione\[25-0]) OR (PC + 4)\[31-28]
	- $ra ← PC+4
- **Unità funzionali**
	- le stesse del Jump
	- MUX per selezionare il valore di PC+4 come valore di destinazione
	- MUX per selezionare il numero del registro $ra come destinazione
- **Flussi dei dati**
	- lo stesso del Jump
	- `PC+4 → MUX → Registri (dato da memorizzare)`
	- `31 → MUX → Registri (registro destinazione)`
- **Segnali di controllo**
	- Jump asserito
	- la CU deve produrre un segnale Link per attivare i due nuovi MUX
- **Tempo necessario**
	- il WriteBack deve avvenire dopo che fono finiti sia il Fetch (per leggere l’istruzione) sia il calcolo di PC+4 (che va memorizzato in $ra) per cui possono presentarsi due casi. Bisogna quindi verificare quale tra le due istruzioni (PC+4 o fetch) impiega più tempo prima di poter fare il WriteBack

![[Screenshot 2024-04-22 alle 17.24.40.png]]

---
## Aggiungere addi/la
Codifica istruzione
![[Screenshot 2024-03-11 alle 19.19.07.png]]

- **Cosa fa**
	- Somma la parte immediata al registro `rs` e ne pone il risultato in `rt`
- **Unità funzionali**
	- ALU per la somma → presente
	- MUX che selezione la parte immediata come secondo argomento → presente
	- Estensione del segno della parte immediata → presente
- **Flussi dei dati**
	- `Registri[rs] → ALU`
	- `Costante → Estensione del segno → ALU`
	- `ALU → Registri[rt]`
- **Segnali di controllo**
	- Si comporta come una `lw` (nonostante questa memorizzi l’indirizzo invece che il dato) ovvero come l’istruzione `la` (load address)
	- `Reg Dst = 0`
	- `ALU Src = 1`
	- `Mem toReg = 0`
	- `Reg Write = 1`
	- `Mem Read = X`
	- `Mem Write = 0`
	- `Branch = 0`
	- `Jump = 0`
	- `ALU Op1 = 0`
	- `ALU Op2 = 0`
- **Tempo necessario**
	- come istruzione di tipo R

![[Screenshot 2024-04-22 alle 17.24.40.jpg]]

---
## Aggiungere jr
- **Cosa fa**
	- trasferisce nel PC il contenuto del registro `rs`
- **Unità funzionali**
	- MUX per selezionare il PC dall’uscita del blocco registri
- **Flussi dei dati**
	- `Registri[rs] → PC`
- **Segnali di controllo**
	- `JumpToReg` che abilita il MUX per inserire in PC il valore del registro

![[Screenshot 2024-04-25 alle 18.28.32.png]]

---
## Aggiungere jrr
- **Cosa fa**
	- salta all’indirizzo (relativo al PC) contenuto nel registro `rs`
- **Unità funzionali**
	- PC + 4 → presente
	- PC + 4 + contenuto registro → presente
	- OR per selezionare tra `Branch` e `JumpRelReg` → da aggiungere
- **Flussi dei dati**
	- `PC+4+Registri[rs]`
- **Segnali di controllo**
	- `JumpRelReg = 1`
	- ``
- **Tempo necessario**
	- Fetch e in parallelo il tempo dell’adder che calcola PC+4 (quindi il massimo tra i due tempi)

![[jrr.png]]