---
Created: 2024-05-13
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
Uno **stream** è un’astrazione derivata da dispositivi dispositivi di input o output sequenziale e i **file** possono essere interpretati come tali (in realtà vengono **bufferizzati** per questioni di efficienza)
Si differenziano in stream di:
- **input** → riceve uno stream di caratteri “uno alla volta”
- **output** → produce uno stream di caratteri

E’ importante ricordare inoltre che gli stream non si applicano solo ai file, ma anche a dispositivi di input/output, internet  ecc.

Per manipolarli in Java esistono 4 differenti classi
- per leggere/scrivere **caratteri** → `java.io.Reader/Writer`
- per leggere/scrivere **byte** → `java.io.StreamInput/StreamOutput`

>[!hint]
>Da Java 5 l’accesso ai file è stato semplificato mediante l’aggiunta della classe `java.util.Scanner`, che però risulta essere **più lenta** perché più potente

>[!warning] Non utilizzare `java.io.File`
>- Molti metodi di File non emettono eccezione quando falliscono
>- Nessun supporto per i collegamenti simbolici
>- Mancanza di metadati: permessi, proprietario, ecc.
>- Molti metodi non scalavano su cartelle piene di file

---
## File
Un **file** è una **collezione di dati salvata** su un supporto di memorizzazione di massa che può essere letto o modificato da programmi differenti. Il programma che lo esegue o modifica deve conoscere il formato dei dati nel file
I file si  distinguono in: file di **testo** e file **binari**

### File di testo
Un file di testo contiene linee di testo (es. in ASCII) ed ognuna di esse termina con un carattere di **nuova linea** (`\n`) o di **carriage return** (`\r`) concatenato con una nuova linea

### File binario
Un file binario può contenere qualsiasi informazione sotto forma di **concatenazione di byte** e diversi programmi potrebbero interpretare lo stesso file in modo diverso (es. immagine)

---
## FileReader e BufferedReader
`BufferedReader` permette una lettura bufferizzata dei caratteri forniti da `FileReader`

```java
BufferedReader br = null;
try {
	br = new BufferedReader(new FileReader(filename));
	
	while(br.ready()) {
		String line = br.readline();
		// ...
	}
}
catch (IOException e) {
	// gestisci l'eccezioe
}
finally {
	if (br != null) {
		try { br.close(); catch(IOException e) {/* gestisci */} }
	}
}
```

### Try with resources
E’ possibile specificare tra parentesi dopo `try` un elenco di istruzioni (separate da `;`) che definiscono **risorse da chiudere automaticamente**

```java
try(BufferedReader br = new BufferedReader(new FileReader(filename));) {
	
	while(br.ready()) {
		String line = br.readline();
		// ...
	}
}
catch (IOException e) {
	// gestisci l'eccezioe
}
```

Questo è possibile solo poiché le classi di questi oggetti implementano `java.lang.AutoCloseable` che è estesa dall’interfaccia `java.io.Closeable`

### Con classe Scanner
Possiamo eventualmente anche leggere file di testo con la classe `java.util.Scanner`
```java
File f = new FIle("mio_file.txt");

try {
	Scanner in = new Scanner(f);
	
	// fineché esiste una prossima riga
	while(in.hasNext())
		// stampa la riga
		System.out.println(in.nextLine())
	in.close();
}
catch(FileNotFoundException e) {
	e.printStackTrace();
}
```

---
## FileWriter e BufferedWriter
Usando il try with resources
```java
try(BufferedWriter bw = new BufferedWriter(new FileWriter(filename))) {
	bw.write("bla bla bla\n")
	// ...
}
catch(IOException e) {
	// gestisci l'eccezione
}
```

### Con classe PrintWriter
Si può altrimenti usare la classe `PrintWriter` e i metodi corrispondenti
```java
// costruisce l'oggetto File
File f = new File("mio_file.txt")

try {
	// costruisce il Writer di file
	PrintWriter out = new PrintWriter(f);
	
	// scrive due righe di testo
	out.println("Prima riga del file");
	out.println("Seconda riga: ")
	out.print(2);
	
	// chiude il file
	out.close();
}
catch(FileNotFoundException e) {
	e.printStackTrace();
}
```

---
## Testi formattati
### Scrivere
E’ possibile scrivere un file di testo formattato utilizzando la classe `java.util.Formatter`
```java
public class FormattaFile {
	private String nome;
	private int valore;
	
	public void scrivi(String filename) throws IOException {
		Formatter output = new Formatter(filename);
		output.format("%s\t%d, nome, valore");
		output.close();
	}
}
```

### Leggere
Essendo il testo nel file formattato usando dei separatori, lo `Scanner` è in grado di restituire la prossima stringa e il prossimo intero
```java
public class FormattaFile {
	private String nome;
	private int valore;
	
	public void scrivi(String filename) throws IOException {
		Formatter output = new Formatter(filename);
		output.format("%s\t%d, nome, valore");
		output.close();
	}
	
	public void leggi(String filename) throws IOException {
		Scanner input = new Scanner(new File(filename));
		nome = input.next();
		valore = input.nextInt();
	}
}
```

---
## java.nio.file.Path
La classe `java.io.File` è rimpiazzata dall'interfaccia `java.nio.file.Path` che rappresenta un percorso gerarchico

### Ottenere un Path
Per ottenere un Path è possibile utilizzare il metodo `Paths.get`
```java
Path p = Path.get("tmp", "foo");
Path p = Path.get("tmp" + File.separator + "foo");

// è un'abbreviazione per
Path p = FileSystem.getDefault().getPath("tmp/foo")
```


Le operazioni che prima si svolgevano nella classe File ora sono metodi statici della classe `java.nio.file.Files` inclusi metodi di comodo per la creazione di `BufferedReader` e `BufferedWriter`
```java
try(BufferedReader br=Files.newBufferedReader(Path.get("fr.txt"));
	BufferedWriter bw=Files.newBufferedWriter(Path.get("fw.txt"))) {
	// legge da fr.txt e scrive in fw.txt
}
```

---
## Serializzare un oggetto