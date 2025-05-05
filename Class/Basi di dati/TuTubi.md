---
Created: 2025-04-30
Class: "[[Basi di dati]]"
Related:
---
---
## Raffinamento dei requisiti
1. utenti
	1. nome
	2. data di iscrizione
	3. gli utenti si devono poter registrare
	4. gli utenti registrati possono pubblicare video, visualizzare quelli disponibili ed esprimere valutazioni e commenti testuali
2. video
	1. titolo
	2. durata
	3. descrizione
	4. nome del file di memorizzazione
	5. categoria (unica, stringa)
	6. tag (almeno una, stringa)
	7. un video può essere in risposta ad un video esistente (vd. 2)
		1. nessun utente può pubblicare un video in risposta ad un video pubblicato da sé stesso
	8. numero di visualizzazioni (vd. 3)
	9. censurato
3. servizio di cronologia
	1. data di visualizzazione
4. valutazione al video (vd. 2)
	1. valore (da 0 a 5)
	2. l’utente che ha pubblicato il video non può votarlo
	3. gli altri utenti possono votare un video al più una volta (il video deve essere visionato)
5. commento al video (vd. 2)
	1. commento
	2. data e ora
	3. ogni utente può commentare più volte uno stesso video (ma lo deve aver visionato)
6. ogni utente può creare delle playlist personali (collezioni ordinate di video)
	1. nome
	2. data di creazione
	3. privata o pubblica
		1. è possibile ottenere le playlist pubbliche degli altri utenti
7. deve essere la ricerca di un video
	1. data categoria, insieme di tag e un intero tra 0 e 5 (se un video non ha nessuna valutazione va comunque restituito)
	2. data una categoria restituire i video con più video in risposta

---
## Diagramma UML delle classi
![[Pasted image 20250504110126.png]]

---
## Specifica delle classi
### Video
Un’istanza di questa classe rappresenta un video
#### Vincoli esterni
`[V.Video.no_risposte_da_stesso_utente]`
Per ogni `v:Video` coinvolto nel link `(v:principale, v:risposta):video_risposta` deve essere che `(principlale, u1):pubblica` e `(risposta, u2):pubblica` tale che `u1!=u2`
#### Specifica delle operazioni di classe
`media_valutazioni():Reale 0..5`
- precondizioni → esiste `u:Utente` tale che esiste il link `(u,this):valutazione`
- postcondizioni →
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `V` l’insieme dei link `valutazione` a cui `this` partecipa
		- sia `S` la somma di `valutazione.valore` in `V`
		- `result=S/|V|`

`visualizzazioni():Intero>=0`
- precondizioni → nessuna
- postcondizioni →
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `v:Visualizza`
		- sia `V` l’insieme dei link `(v,this):vis_video`
		- `result=|V|`

### Utente
Un’istanza di questa classe rappresenta un utente
#### Specifica delle operazioni di classe
`cerca_playlist(u:Utente): Playlist [0..*]`
- precondizioni → nessuna
- postcondizioni →
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `P` l’insieme di `p:Playlist` tali che esiste il link `(u,p):utente_playlist`e che `p.visibilità=pubblico`
		- `result=P`

`cerca_video(c:Stringa, t:Stringa [0..*], v:Intero):Video [0..*]`
- precondizioni → nessuna
- postcondizioni →
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `V` l’insieme dei `vid:Video` tali che `vid.categoria=c` e `vid.tag=t` e `vid` non è istanza di `VideoCensurato`
		- per ogni `vid` in `V`, per ogni `u:Utente` se esiste il link `(vid,u):valutazione`, allora deve essere che `vid.media_valutazione()>=v`
		- `result=V`

`cerca_più_risposte(c:Stringa):Video [0..*]`
- precondizioni → nessuna
- postcondizioni →
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `V` l’insieme dei `v:Video` tali che `v.categoria=c`
		- per ogni `v` in `V` per ogni `v1:Video` coinvolto nei link `(v,v1):video_risposta` tale che `v` abbia il ruolo `principale`, sia `p` il numero di `v` che rispettano questa condizione
		- `result` è/sono gli `v` che hanno `p` massimo

#### Vincoli esterni
`[V.Utente.no_valutazioni_video_non_visto]`
Per ogni `u:Utente` e per ogni `v:Video` coinvolto in link `(u,v):valutazione` deve succedere che esiste `vis:Visualizza` tale ci siano sia `(u,vis):utente_vis` che `(vis,v):vis_video`

`[V.Utente.no_commenti_video_non_visto]`
Per ogni `c:Commento` tale che esiste `u:Utente` e `v:Video` tale che ci siano i link `(u,c):utente_comm` e `(c,v):comm_video` deve esistere `vis:Visualizza` tale che ci siano sia `(u,vis):utente_vis` che `(vis,v):vis_video`

`[V.Utente.no_interazioni_video_censurati]`
Per ogni `u:Utente`, per ogni `v:VideoCensurato` non esiste nessun link `(u,v):valutazione`, (`c:Commento` il link `(u,c):utente_comm` e `(c,v):comm_video`), (`vis:Visualizza` il link `(u,vis):utente_vis` e `(vis,v):vis_video`)

---
## Diagramma degli use case
![[Pasted image 20250502011240.png]]

---
## Specifica degli use case
### Pubblicazione
`pubblica(u:Utente, c:Stringa, d:Stringa, tag:Stringa, t:Stringa):Video`
- precondizioni → nessuna
- postcondizioni → 
	- viene creato e restituito un nuovo oggetto `result:Video` con i valori `c`, `d`, `tag`, `t` rispettivamente per gli attributi `categoria`, `descrizione`, `tag`, `titolo`
	- viene creato il link `(result,u):pubblica`

### CreazionePlaylist
`crea_playlist(nom:Stringa, vis:{pubblico, privato}):Playlist`
- precondizioni → nessuna
- postcondizioni →
	- viene creato e restituito un nuovo oggetto `result:Playlist` con i valori `nom`, `vis`, `adesso:Data` rispettivamente per gli attributi `nome`, `visibilità`, `creazione`

### Censuramento
`censura(v:Video)`
- precondizioni → esiste `u:Utente` tale che esiste il link `(v,u):pubblica`
- postcondizioni →
	- crea un’istanza di `VideoCensurato` a partire da `v`

### AggiuntaVideo
`aggiungi_video(p:Playlist, v:Video, u:Utente)`
- precondizioni → `v` non è istanza di `VideoCensurato` e esiste il link `(u,p):utente_playlist`
- postcondizioni → viene creato il link `(v,p):video_playlist`

### CommentaVideo
`commenta(c:Stringa, v:Video, u:Utente):Commento`
- precondizioni → `v` non è istanza di `VideoCensurato` e sia `vis:Visualizza` esistono i link `(v,vis):vis_video` e `(vis,u):utente_vis`
- postcondizioni →
	- viene creato e restituito un nuovo oggetto `result:Commento` con i valori `c`, `adesso:Data` rispettivamente per gli attributi `commento`, `pubblicazione`
	- viene creato il link `(result,u):utente_comm` e viene creato il link `(result,v):comm_video`

### ValutaVideo
`valutazione(val:Intero, v:Video, u:Utente):Commento`
- precondizioni → `v` non è istanza di `VideoCensurato` e sia `v:Visualizza` esistono i link `(v,vis):vis_video` e `(vis,u):utente_vis`
- postcondizioni →
	- viene creato il link `(u,v):valutazione` con valore per l’attributo `valore=val`