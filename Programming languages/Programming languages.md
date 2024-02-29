# Cosa vuol dire programmare?
1. Dato un problema
2. Si analizza il problema e si capisce cosa può costituire una soluzione (e cosa no)
3. Si progetta una serie di passi che risolvono il problema (**algoritmo**)
4. Si traducono i passi in istruzioni per il calcolatore seguendo la sintassi
5. Si valuta l'implementazione sui casi di test per capire se le istruzioni risolvono il problema

# Come approcciare un problema?
1. **Fase di analisi**: Analisi del problema e comprensione approfondita
	- Controllare tutti i casi di test per vedere se ci sono casistiche particolari non descritte
	- Provare a darne una breve spiegazione in italiano
	- Capire cosa può costituire una soluzione al problema
2. **Fase di definizione strutture dati e algoritmo**꞉
	- Capire quali strutture dati sono appropriate per il problema
	- Capire quali sono le istruzioni per risolvere il problema (algoritmo)
3. **Implementazione in python**꞉
	- Rappresentazione dei dati
	- Scrivere l’algoritmo secondo sintassi e potenzialità di Python
4. **Testing e debug di eventuali casi incorretti**
	- Capire velocemente eventuali errori in maniera da isolare la parte di codice non corretta
## Approccio bottom-up
1. Parto da piccoli sottoproblemi che ho localizzato nell'analisi e provo a scrivere piccole funzioni che risolvono questi sotto problemi
2. Mi assicuro che le funzioni siano corrette (debug oppure prova nell'interprete ipython)
3. Compongo le funzionalità al fine di risolvere il problema principale
## Approccio top-down
1. Fornisco una descrizione del problema ad alto livello oppure pseudo‐codice (descrivo in italiano cosa devo fare)
2. Implemento l'infrastruttura globale (flusso del programma principale)
	- Creo funzioni placeholder che NON implementano la semantica ma sono dei segnaposti
	- Definisco l'interfaccia delle funzioni (parametri di ingresso e valore di ritorno) in maniera congruente al problema da risolvere
3. A questo punto ho il sistema che "esegue" da cima a fondo ma senza la logica corretta. Passo ad implementare la logica specifica di ogni funzione segnaposto

## Languages
[[Python]]
[[Verilog]]
[[Java]]
