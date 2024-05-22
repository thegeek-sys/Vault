Pattern MVC è un pattern architetturale (Modello Vista Controllo)
In modello devo solo definire la struttura, nella vista devo avere tutti i modi per visualizzare la struttura (immagini, audio etc.), in Controllo devo chiedere al modello di aggiornare il suo stato. Dall’uml del modello (package) non deve uscire dipendenze ad altri package (non possono uscire frecce ma possono entrare)

in swing gli eventi dell’utente vanno gestiti nel Controllo

creare 3 packages model, view e controller

ci serve per fare in modo che ciò che è contenuto nel model è riutilizzabile al 100%, QUI DENTRO CI VA LA LOGICA (es. capire come si scontrano gli oggetti, come funzionano le bolle…). ciò significa che posso prendere questo package e metterlo in un altro progetto in quanto nel model non ci sono dipendenze da altre parti (view o controller). non ci sono assets grafici o audio. qui dentro ci stanno le cose che estenderanno Observable. qui ci sta il gameloop. quando il modello si aggiorna

il controller è riutilizzabile al 50%. qui ci sta un metodo main del videogioco che chiede un istanza del modello e un’istanza della view sulla quale aggiunge gli action listener (gli action listener devono essere forniti dal controller alla view). questo controlla dal modello e dalla view. qui dentro ci stanno le cose che estenderanno Observer
deve fare:
- crea modello
- crea view
- aggiunti action listener alla view
- view inizia ad osservare  il modello

la view non è riutilizzabile
