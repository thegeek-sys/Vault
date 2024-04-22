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