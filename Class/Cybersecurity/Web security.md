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

