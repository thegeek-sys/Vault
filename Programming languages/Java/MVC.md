Pattern MVC è un pattern architetturale (Modello Vista Controllo)
In modello devo solo definire la struttura, nella vista devo avere tutti i modi per visualizzare la struttura (immagini, audio etc.), in Controllo devo chiedere al modello di aggiornare il suo stato. Dall’uml del modello (package) non deve uscire dipendenze ad altri package (non possono uscire frecce ma possono entrare)

in swing gli eventi dell’utente vanno gestiti nel Controllo