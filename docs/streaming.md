# Streaming

## Réponses (NDJSON `/v1/generate/stream`, `/v1/chat/stream`, gros corps)

* Les octets amont sont relayés **dès réception** (`httpx.Response.aiter_raw`), sans
  analyse, sans réécriture, sans agrégation. Aucun événement n'est ajouté.
* Pas de SSE dans le contrat réel : la Gateway n'émet que du NDJSON (vérifié en live :
  `Content-Type: application/x-ndjson`, un seul `{"type":"done"}` final).
* **Rupture amont** au milieu du flux : la connexion aval est interrompue (pas de fin
  propre fabriquée, pas de `done` inventé) ; métrique `aigp_proxy_stream_interruptions_total`.
* **Déconnexion client** : le générateur de relais est annulé, la réponse amont est fermée,
  donc la connexion vers la Gateway aussi (qui libère alors le GPU).
* Ressources : `finally` ferme la réponse amont et décrémente `aigp_proxy_active_requests`
  dans tous les cas (succès, erreur, annulation, timeout).

Tests (`tests/integration/test_proxy_streaming.py`, vrais serveurs HTTP) :

| Exigence | Test | Résultat |
|---|---|---|
| FIRST_CHUNK_DELIVERED_EARLY | premier fragment < 0,5 s alors que la génération dure > 1 s | PASS |
| CHUNK_ORDER_PRESERVED / STREAM_END_DETECTED | ordre `t0..t3` puis un unique `done` | PASS |
| octets identiques au direct | `test_chat_stream_relayed_unchanged` | PASS |
| STREAM_INTERRUPTION / UPSTREAM_DISCONNECT | crash amont après 2 lignes → erreur côté client, pas de `done` | PASS |
| CLIENT_DISCONNECT_HANDLED | client parti après 2 lignes → générateur amont stoppé < 5 s (au lieu de 10 s) | PASS |
| NO_FULL_RESPONSE_BUFFERING | réponse de 256 Mio relayée, croissance RSS < 96 Mio | PASS |

## Requêtes (multipart ASR, gros JSON, chunked)

* Le corps est transmis à httpx comme itérateur asynchrone de `request.stream()` :
  jamais `await request.body()` hors inspection JSON bornée (≤ 16 Mio, pour lire `model`).
* `Content-Length` du client conservé → l'amont reçoit `Content-Length` **sans**
  `Transfer-Encoding` (la Gateway refuse les deux). Requête chunked → relayée chunked.
* **Pression de retour** : uvicorn suspend la lecture de la socket client quand son
  tampon dépasse son seuil haut, et le relais ne lit le fragment suivant qu'une fois le
  précédent écrit vers l'amont.
* Pas de base64, pas de Redis, pas de fichier temporaire, pas de re-segmentation audio.
* Aucune limite de taille propre au pool : la limite effective est celle de l'endpoint
  Gateway (ASR : 256 Mio par job). Les 600 Mo d'import RAGatm ne sont pas une limite ASR.

| Exigence | Test | Résultat |
|---|---|---|
| REQUEST_STREAMING | 300 Mio multipart : l'amont reçoit des octets avant la fin de l'envoi client | PASS |
| NO_FULL_BODY_BUFFERING | même test : croissance RSS < 96 Mio, fragments ≤ 4 Mio | PASS |
| BACKPRESSURE | amont lent (2 ms/fragment), 96 Mio : croissance RSS < 64 Mio | PASS |
| corps incomplet | `Content-Length: 1000` puis coupure → jamais reçu « complet » en amont | PASS |
| 600 000 000 octets (simulé) | banc `scripts/bench_proxy.py` : 548 Mo/s, +0,7 Mio RSS du processus pool | PASS |

Le scénario de 600 Mo est une **simulation** de transfert multipart, pas un job ASR réel.

## Cas particulier POST ASR et déconnexion

Si le client se déconnecte pendant l'envoi, la requête amont échoue (corps incomplet) et
aucune association n'est créée. S'il se déconnecte **après** l'admission par la Gateway,
la réponse de soumission est quand même lue par le pool et l'association est enregistrée :
le job reste consultable et supprimable. Le pool ne supprime jamais un job parce que son
client est parti (purge par durée de vie côté Gateway).
