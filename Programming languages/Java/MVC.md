Pattern MVC è un pattern architetturale (Modello Vista Controllo)
In modello devo solo definire la struttura, nella vista devo avere tutti i modi per visualizzare la struttura (immagini, audio etc.), in Controllo devo chiedere al modello di aggiornare il suo stato. Dall’uml del modello (package) non deve uscire dipendenze ad altri package (non possono uscire frecce ma possono entrare)

in swing gli eventi dell’utente vanno gestiti nel Controllo

creare 3 packages model, view e controller

ci serve per fare in modo che ciò che è contenuto nel model è riutilizzabile al 100%, QUI DENTRO CI VA LA LOGICA (es. capire come si scontrano gli oggetti, come funzionano le bolle…). ciò significa che posso prendere questo package e metterlo in un altro progetto in quanto nel model non ci sono dipendenze da altre parti (view o controller). non ci sono assets grafici o audio. qui dentro ci stanno le cose che estenderanno Observable. quando il modello si aggiorna deve mandare una notifica tramite Observable che deve essere catturata dalla view che si deve aggiornare. dei nemici e dei power up ci deve essere solamente la logica che li fa funzionare


il controller è riutilizzabile al 50%. qui ci sta un metodo main del videogioco che chiede un istanza del modello e un’istanza della view sulla quale aggiunge gli action listener (gli action listener devono essere forniti dal controller alla view, NON E’ UN PROBLEMA SE VENGONO GESTITI DIRETTAMENTE QUI). questo controlla dal modello e dalla view. qui dentro ci stanno le cose che estenderanno Observer. gli Osberver ricevono oltre all’oggetto anche il riferimento all’osservabile che glielo ha mandato. qui ci sta il gameloop che chiede al modello di aggiornarsi che notifica al view che il modello è pronto e quindi di aggiornarsi
deve fare:
- crea modello
- crea view
- aggiunti action listener alla view
- view inizia ad osservare il modello


la view non è riutilizzabile
