#!/usr/bin/env python3
"""Suite de parité Direct / Pool — contre une AI Gateway Open RÉELLE.

Les deux chemins interrogent la MÊME Gateway Open et les MÊMES modèles, sans reconfiguration :
  DIRECT : http://127.0.0.1:18100 (port de qualification de la Gateway)
  POOL   : http://127.0.0.1:18090 (Pool Open devant cette Gateway)

Comparaisons : contrat de réponse (champs), identité des modèles (model / model_logical / digest /
pipeline_id / model_revision), vecteurs d'embeddings (égalité numérique), classement du reranker,
événements de flux, en-têtes (X-Request-ID, X-AIGP-Error, Retry-After), erreurs négatives. La
génération n'est pas déterministe : on compare le contrat et les invariants, jamais le texte.

Usage : parity_direct_pool.py --direct URL --pool URL --out results.json
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import uuid

import requests

R: list[dict] = []


def rec(op, verdict, **d):
    R.append({"operation": op, "verdict": verdict, **d})
    print(
        f"{verdict:12s} {op}  "
        + json.dumps(
            {k: v for k, v in d.items() if k not in ("direct_body", "pool_body")},
            ensure_ascii=False,
        )[:360],
        flush=True,
    )


def call(base, method, path, *, json_body=None, data=None, headers=None, stream=False, timeout=300):
    h = {"X-Request-ID": f"open5-{uuid.uuid4().hex[:12]}"}
    h.update(headers or {})
    r = requests.request(
        method, base + path, json=json_body, data=data, headers=h, stream=stream, timeout=timeout
    )
    return r


def body(r):
    try:
        return r.json()
    except Exception:
        return None


def detail(b):
    return b.get("detail") if isinstance(b, dict) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", default="http://127.0.0.1:18100")
    ap.add_argument("--pool", default="http://127.0.0.1:18090")
    ap.add_argument("--out", default="parity-results.json")
    a = ap.parse_args()
    D, P = a.direct, a.pool

    # ---------------------------------------------------------------- health / catalogue
    rd, rp = call(D, "GET", "/health/live"), call(P, "GET", "/health/live")
    rec(
        "GET /health/live",
        "PASS" if rd.status_code == 200 and rp.status_code == 200 else "FAIL",
        direct={"http": rd.status_code, "inferential": body(rd).get("inferential")},
        pool={"http": rp.status_code, "body": body(rp)},
        note=(
            "POOL_LOCAL: le Pool répond lui-même (liveness du Pool), "
            "la Gateway n'est pas sollicitée"
        ),
    )
    rd, rp = call(D, "GET", "/health/runtime"), call(P, "GET", "/health/runtime")
    bd, bp = body(rd), body(rp)
    same = (
        bd is not None
        and bp is not None
        and {k: v for k, v in bd.items() if k not in ("inflight", "queued", "queue_time")}
        == {k: v for k, v in bp.items() if k not in ("inflight", "queued", "queue_time")}
    )
    rec(
        "GET /health/runtime",
        "PASS" if rd.status_code == rp.status_code == 200 and same else "FAIL",
        http=(rd.status_code, rp.status_code),
        identical_minus_inflight=same,
        pool_request_id=rp.headers.get("X-Request-ID") is not None,
        pool_marked_error=rp.headers.get("X-AIGP-Error"),
    )
    rd, rp = call(D, "GET", "/v1/models"), call(P, "GET", "/v1/models")
    bd, bp = body(rd), body(rp)
    rec(
        "GET /v1/models",
        "PASS" if rd.status_code == rp.status_code == 200 and bd == bp else "FAIL",
        http=(rd.status_code, rp.status_code),
        identical=bd == bp,
        models=[
            (m.get("name"), m.get("model_logical"), m.get("registry_status"))
            for m in (bd or {}).get("models", [])
        ],
    )
    rd, rp = (
        call(D, "GET", "/health/embeddings", timeout=120),
        call(P, "GET", "/health/embeddings", timeout=120),
    )
    bd, bp = body(rd), body(rp)

    def ref(b):
        return [
            {
                k: m.get(k)
                for k in (
                    "model_logical",
                    "expected_digest",
                    "observed_digest",
                    "identity_verified",
                    "pipeline_id_query",
                    "pipeline_id_document",
                    "registry_status",
                )
            }
            for m in (b or {}).get("models", [])
        ]

    rec(
        "GET /health/embeddings (sonde explicite)",
        "PASS" if rd.status_code == rp.status_code == 200 and ref(bd) == ref(bp) else "FAIL",
        http=(rd.status_code, rp.status_code),
        reference_identical=ref(bd) == ref(bp),
        reference=ref(bd),
    )

    # ---------------------------------------------------------------- génération (agrégée)
    req = {
        "prompt": "Réponds par un seul mot : quelle est la couleur du ciel par beau temps ?",
        "model": "compact-generation-8",
        "options": {"num_predict": 8, "temperature": 0, "seed": 7},
    }
    rd, rp = (
        call(D, "POST", "/v1/generate", json_body=req),
        call(P, "POST", "/v1/generate", json_body=req),
    )
    bd, bp = body(rd), body(rp)

    def keys(b):
        return sorted(b.keys()) if isinstance(b, dict) else None

    def ident(b):
        return (
            {k: b.get(k) for k in ("model", "model_logical", "provider", "done_reason")}
            if isinstance(b, dict)
            else None
        )

    rec(
        "POST /v1/generate",
        "PASS"
        if rd.status_code == rp.status_code == 200
        and keys(bd) == keys(bp)
        and ident(bd) == ident(bp)
        and (bd.get("usage") or {}).get("prompt_tokens")
        == (bp.get("usage") or {}).get("prompt_tokens")
        else "FAIL",
        http=(rd.status_code, rp.status_code),
        contract_keys_identical=keys(bd) == keys(bp),
        identity_direct=ident(bd),
        identity_pool=ident(bp),
        prompt_tokens=((bd or {}).get("usage") or {}).get("prompt_tokens"),
        text_direct=(bd or {}).get("text", "")[:40],
        text_pool=(bp or {}).get("text", "")[:40],
        pool_request_id_echo=rp.headers.get("X-Request-ID"),
        pool_marked_error=rp.headers.get("X-AIGP-Error"),
    )
    # modèle implicite (sans `model`) : défaut Gateway = défaut déclaré au Pool
    req2 = {"prompt": "Dis bonjour.", "options": {"num_predict": 4, "temperature": 0}}
    rd, rp = (
        call(D, "POST", "/v1/generate", json_body=req2),
        call(P, "POST", "/v1/generate", json_body=req2),
    )
    rec(
        "POST /v1/generate (modèle implicite)",
        "PASS"
        if rd.status_code == rp.status_code == 200 and ident(body(rd)) == ident(body(rp))
        else "FAIL",
        http=(rd.status_code, rp.status_code),
        identity=ident(body(rd)),
    )

    # ---------------------------------------------------------------- flux NDJSON
    def stream(base):
        t0 = time.time()
        r = call(
            base,
            "POST",
            "/v1/generate/stream",
            json_body={
                "prompt": "Compte de un à cinq en lettres.",
                "model": "compact-generation-8",
                "options": {"num_predict": 16, "temperature": 0, "seed": 7},
            },
            stream=True,
            headers={"Accept": "application/x-ndjson"},
        )
        evs = []
        first = None
        for line in r.iter_lines():
            if not line:
                continue
            if first is None:
                first = round((time.time() - t0) * 1000)
            evs.append(json.loads(line))
        return r, evs, first

    rd, ed, fd = stream(D)
    rp, ep, fp = stream(P)
    td, tp = [e.get("type") for e in ed], [e.get("type") for e in ep]
    done_d, done_p = (ed[-1] if ed else {}), (ep[-1] if ep else {})
    rec(
        "POST /v1/generate/stream",
        "PASS"
        if rd.status_code == rp.status_code == 200
        and rd.headers.get("Content-Type") == rp.headers.get("Content-Type")
        and td.count("done") == tp.count("done") == 1
        and td[-1] == tp[-1] == "done"
        and "error" not in td + tp
        and sorted(done_d.keys()) == sorted(done_p.keys())
        and done_d.get("model") == done_p.get("model")
        else "FAIL",
        http=(rd.status_code, rp.status_code),
        content_type=(rd.headers.get("Content-Type"), rp.headers.get("Content-Type")),
        events_direct={e: td.count(e) for e in set(td)},
        events_pool={e: tp.count(e) for e in set(tp)},
        first_event_ms=(fd, fp),
        done_keys_identical=sorted(done_d.keys()) == sorted(done_p.keys()),
        done_model=(done_d.get("model"), done_p.get("model")),
        done_reason=(done_d.get("done_reason"), done_p.get("done_reason")),
        pool_request_id=rp.headers.get("X-Request-ID") is not None,
    )
    # annulation côté client : le Pool ne doit pas produire d'erreur ensuite (service nominal)
    r = call(
        P,
        "POST",
        "/v1/generate/stream",
        json_body={
            "prompt": "Écris un long texte sur la montagne.",
            "model": "compact-generation-8",
            "options": {"num_predict": 200},
        },
        stream=True,
        headers={"Accept": "application/x-ndjson"},
    )
    n = 0
    for line in r.iter_lines():
        if line:
            n += 1
        if n >= 2:
            break
    r.close()
    time.sleep(1)
    r2 = call(P, "GET", "/health/runtime")
    rec(
        "stream : annulation client via le Pool",
        "PASS" if r2.status_code == 200 and n >= 2 else "FAIL",
        events_before_abort=n,
        pool_healthy_after=r2.status_code,
        inflight_after=(body(r2) or {}).get("inflight"),
    )

    # ---------------------------------------------------------------- embeddings
    texts = [
        "Le lac de Vermeil a une profondeur maximale de 47 mètres.",
        "The turbidity sensor model is Hydra-TX-9.",
    ]
    for task in ("document", "query"):
        req = {"texts": texts, "model": "fr-en-retrieval", "task": task}
        rd, rp = (
            call(D, "POST", "/v1/embeddings", json_body=req),
            call(P, "POST", "/v1/embeddings", json_body=req),
        )
        bd, bp = body(rd), body(rp)

        def idf(b):
            return (
                {
                    k: b.get(k)
                    for k in (
                        "model",
                        "model_logical",
                        "dimension",
                        "pipeline_id",
                        "model_digest",
                        "expected_digest",
                        "identity_verified",
                        "normalized",
                        "task",
                        "truncated",
                    )
                }
                if isinstance(b, dict)
                else None
            )

        vec_equal = (
            isinstance(bd, dict)
            and isinstance(bp, dict)
            and bd.get("embeddings") == bp.get("embeddings")
        )
        max_diff = None
        if (
            isinstance(bd, dict)
            and isinstance(bp, dict)
            and bd.get("embeddings")
            and bp.get("embeddings")
        ):
            max_diff = max(
                abs(x - y)
                for vd, vp in zip(bd["embeddings"], bp["embeddings"], strict=False)
                for x, y in zip(vd, vp, strict=False)
            )
        rec(
            f"POST /v1/embeddings task={task}",
            "PASS"
            if rd.status_code == rp.status_code == 200 and idf(bd) == idf(bp) and vec_equal
            else "FAIL",
            http=(rd.status_code, rp.status_code),
            identity_identical=idf(bd) == idf(bp),
            identity=idf(bd),
            vectors_identical=vec_equal,
            max_abs_diff=max_diff,
            dim=len((bd or {}).get("embeddings", [[]])[0]) if bd else None,
        )

    # ---------------------------------------------------------------- rerank
    docs = [
        {"id": "d1", "text": "La cantine sert de la polenta le mardi."},
        {"id": "d2", "text": "La profondeur maximale du lac de Vermeil est de 47 mètres."},
        {"id": "d3", "text": "Le capteur de turbidité est le Hydra-TX-9."},
        {"id": "d4", "text": "Le lac Aurore gèle 182 jours par an."},
    ]
    req = {
        "query": "Quelle est la profondeur maximale du lac de Vermeil ?",
        "documents": docs,
        "model": "rerank-default",
        "top_n": 4,
    }
    rd, rp = (
        call(D, "POST", "/v1/rerank", json_body=req),
        call(P, "POST", "/v1/rerank", json_body=req),
    )
    bd, bp = body(rd), body(rp)

    def res(b):
        return [
            (x.get("id"), x.get("index"), round(x.get("relevance_score", x.get("score", 0)), 6))
            for x in (b or {}).get("results", [])
        ]

    def idr(b):
        return {k: (b or {}).get(k) for k in ("model", "model_logical", "model_revision")}

    rec(
        "POST /v1/rerank",
        "PASS"
        if rd.status_code == rp.status_code == 200
        and res(bd) == res(bp)
        and idr(bd) == idr(bp)
        and res(bd)
        and res(bd)[0][0] == "d2"
        else "FAIL",
        http=(rd.status_code, rp.status_code),
        ranking_identical=res(bd) == res(bp),
        identity=idr(bd),
        results_direct=res(bd),
        results_pool=res(bp),
    )

    # ---------------------------------------------------------------- erreurs négatives
    def neg(op, method, path, json_body=None, data=None, headers=None, expect_pool_marked=None):
        rd = (
            call(D, method, path, json_body=json_body, data=data, headers=headers, timeout=60)
            if expect_pool_marked is not True
            else None
        )
        rp = call(P, method, path, json_body=json_body, data=data, headers=headers, timeout=60)
        bd, bp = (body(rd) if rd is not None else None), body(rp)
        pool_marked = rp.headers.get("X-AIGP-Error")
        gw_code = (detail(bp) or {}).get("code") if isinstance(detail(bp), dict) else None
        d_code = (detail(bd) or {}).get("code") if isinstance(detail(bd), dict) else None
        if expect_pool_marked is True:
            ok = pool_marked is not None and isinstance(bp, dict) and "error" in bp
        elif expect_pool_marked is False:
            ok = (
                rd.status_code == rp.status_code
                and pool_marked is None
                and (detail(bd) == detail(bp) or (d_code and d_code == gw_code))
            )
        else:
            ok = rd.status_code == rp.status_code
        rec(
            op,
            "PASS" if ok else "FAIL",
            http=((rd.status_code if rd is not None else None), rp.status_code),
            direct_code=d_code,
            pool_relayed_code=gw_code,
            pool_marked=pool_marked,
            pool_error=(bp or {}).get("error") if isinstance(bp, dict) and "error" in bp else None,
            retry_after=(
                rd.headers.get("Retry-After") if rd is not None else None,
                rp.headers.get("Retry-After"),
            ),
            request_id=rp.headers.get("X-Request-ID") is not None,
        )

    neg(
        "modèle inconnu (llama3.1:8b) — Pool refuse avant la Gateway",
        "POST",
        "/v1/generate",
        {"prompt": "x", "model": "llama3.1:8b"},
        expect_pool_marked=True,
    )
    neg(
        "modèle enregistré non installé (compact-generation) — Pool refuse (non déclaré)",
        "POST",
        "/v1/generate",
        {"prompt": "x", "model": "compact-generation"},
        expect_pool_marked=True,
    )
    # Différence ATTENDUE et documentée : en Direct la Gateway refuse (400
    # model_not_registered) ; via le Pool, l'allowlist de modèles refuse AVANT la
    # Gateway (404 MODEL_NOT_AVAILABLE, erreur marquée Pool).
    rd = call(
        D, "POST", "/v1/generate", json_body={"prompt": "x", "model": "llama3.1:8b"}, timeout=60
    )
    rp = call(
        P, "POST", "/v1/generate", json_body={"prompt": "x", "model": "llama3.1:8b"}, timeout=60
    )
    dcode = (detail(body(rd)) or {}).get("code") if isinstance(detail(body(rd)), dict) else None
    rec(
        "modèle inconnu — Direct vs Pool (différence attendue, erreur Pool distinguable)",
        "EXPECTED_DIFFERENCE"
        if rd.status_code == 400
        and dcode == "model_not_registered"
        and rp.status_code == 404
        and rp.headers.get("X-AIGP-Error") == "MODEL_NOT_AVAILABLE"
        else "FAIL",
        http=(rd.status_code, rp.status_code),
        direct_code=dcode,
        pool_marked=rp.headers.get("X-AIGP-Error"),
    )
    neg(
        "embeddings texte > 1200 car. — erreur Gateway relayée",
        "POST",
        "/v1/embeddings",
        {"texts": ["a" * 1201], "model": "fr-en-retrieval", "task": "document"},
        expect_pool_marked=False,
    )
    neg(
        "embeddings lot > 64 — erreur Gateway relayée",
        "POST",
        "/v1/embeddings",
        {"texts": ["x"] * 65, "model": "fr-en-retrieval", "task": "document"},
        expect_pool_marked=False,
    )
    neg(
        "rerank document > 4000 car. — erreur Gateway relayée",
        "POST",
        "/v1/rerank",
        {"query": "q", "documents": [{"id": "big", "text": "b" * 4001}], "model": "rerank-default"},
        expect_pool_marked=False,
    )
    neg(
        "erreur de contrat (champ provider) — Gateway 400 relayée",
        "POST",
        "/v1/generate",
        {"prompt": "x", "model": "compact-generation-8", "provider": "ollama"},
        expect_pool_marked=False,
    )
    neg(
        "num_ctx hors borne — Gateway 400 relayée",
        "POST",
        "/v1/generate",
        {"prompt": "x", "model": "compact-generation-8", "options": {"num_ctx": 999999}},
        expect_pool_marked=False,
    )
    neg(
        "corps JSON invalide — Pool refuse (INVALID_REQUEST_BODY)",
        "POST",
        "/v1/generate",
        data=b"{not json",
        headers={"Content-Type": "application/json"},
        expect_pool_marked=True,
    )
    neg(
        "route hors contrat (/v1/vision/structured-extract) — Pool 404",
        "POST",
        "/v1/vision/structured-extract",
        {"x": 1},
        expect_pool_marked=True,
    )
    neg(
        "route hors contrat (/health/structured-extract) — Pool 404",
        "GET",
        "/health/structured-extract",
        expect_pool_marked=True,
    )
    neg(
        "en-tête de redirection (X-Upstream-Url) — Pool refuse",
        "GET",
        "/health/runtime",
        headers={"X-Upstream-Url": "http://evil.invalid:1"},
        expect_pool_marked=True,
    )
    neg(
        "en-tête forgé X-AIGP-Error par le client — neutralisé",
        "GET",
        "/health/runtime",
        headers={"X-AIGP-Error": "FORGED"},
        expect_pool_marked=None,
    )
    r = call(P, "GET", "/health/runtime", headers={"X-AIGP-Error": "FORGED"})
    rec(
        "X-AIGP-Error forgé non relayé",
        "PASS" if r.status_code == 200 and r.headers.get("X-AIGP-Error") is None else "FAIL",
        http=r.status_code,
        header_back=r.headers.get("X-AIGP-Error"),
    )
    r = call(P, "GET", "/health/runtime", headers={"X-AIGP-Target-Node": "compact-gateway"})
    rec(
        "X-AIGP-Target-Node sans jeton admin — refusé",
        "PASS"
        if r.status_code == 403 and r.headers.get("X-AIGP-Error") == "TARGET_NODE_FORBIDDEN"
        else "FAIL",
        http=r.status_code,
        marked=r.headers.get("X-AIGP-Error"),
    )
    r = call(P, "GET", "/internal/nodes")
    rec(
        "/internal/nodes sans jeton — refusé",
        "PASS" if r.status_code in (401, 403, 404) else "FAIL",
        http=r.status_code,
    )
    rid = "open5-parity-rid-0001"
    r = call(P, "GET", "/health/runtime", headers={"X-Request-ID": rid})
    rec(
        "X-Request-ID fourni — relayé tel quel",
        "PASS" if r.headers.get("X-Request-ID") == rid else "FAIL",
        back=r.headers.get("X-Request-ID"),
    )
    r = call(P, "GET", "/health/runtime", headers={"X-Request-ID": "bad id with spaces!"})
    rec(
        "X-Request-ID invalide — remplacé",
        "PASS"
        if r.headers.get("X-Request-ID") and r.headers.get("X-Request-ID") != "bad id with spaces!"
        else "FAIL",
        back=r.headers.get("X-Request-ID"),
    )
    r = call(
        P,
        "POST",
        "/v1/audio/transcriptions",
        data=b"x",
        headers={"Content-Type": "multipart/form-data; boundary=x"},
    )
    rec(
        "ASR via Pool (capacité non déclarée) — refus Pool",
        "PASS" if r.headers.get("X-AIGP-Error") else "FAIL",
        http=r.status_code,
        marked=r.headers.get("X-AIGP-Error"),
    )
    r = call(
        P,
        "POST",
        "/v1/vision/analyze",
        json_body={"schema_version": "1", "image_base64": "aGVsbG8=", "task": "describe"},
    )
    rec(
        "Vision via Pool (capacité non déclarée) — refus Pool",
        "PASS" if r.headers.get("X-AIGP-Error") else "FAIL",
        http=r.status_code,
        marked=r.headers.get("X-AIGP-Error"),
    )
    # saturation : 4 générations simultanées via le Pool (Gateway : concurrence 2)
    # => 503 generation_saturated relayé + Retry-After
    out = []

    def w(base):
        rr = call(
            base,
            "POST",
            "/v1/generate",
            json_body={
                "prompt": "Écris une phrase sur la montagne.",
                "model": "compact-generation-8",
                "options": {"num_predict": 40},
            },
            timeout=300,
        )
        out.append(
            (
                rr.status_code,
                (
                    (detail(body(rr)) or {}).get("code")
                    if isinstance(detail(body(rr)), dict)
                    else None
                ),
                rr.headers.get("Retry-After"),
                rr.headers.get("X-AIGP-Error"),
            )
        )

    th = [threading.Thread(target=w, args=(P,)) for _ in range(4)]
    [t.start() for t in th]
    [t.join() for t in th]
    sat = [o for o in out if o[0] == 503]
    rec(
        "saturation Gateway relayée par le Pool (4 requêtes, concurrence 2)",
        "PASS"
        if sat and all(o[1] == "generation_saturated" and o[2] and o[3] is None for o in sat)
        else "OBSERVED",
        outcomes=out,
    )
    time.sleep(2)
    r = call(
        P,
        "POST",
        "/v1/generate",
        json_body={
            "prompt": "Dis bonjour.",
            "model": "compact-generation-8",
            "options": {"num_predict": 4},
        },
    )
    rec(
        "après saturation : Pool sert à nouveau",
        "PASS" if r.status_code == 200 else "FAIL",
        http=r.status_code,
    )

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    fails = [x for x in R if x["verdict"] == "FAIL"]
    print(f"\n{len(R)} opérations, {len(fails)} FAIL -> {a.out}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
