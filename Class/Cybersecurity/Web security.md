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

