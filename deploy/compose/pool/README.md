# Déploiement Compose du Pool Open

1. `cp ../../../config/compact.example.yaml pool.yaml` puis relire chaque valeur ; passer `status: approved`
   (impose `upstream_allowlist` et `job_affinity.backend` ∈ {redis, none}).
2. `cp .env.example .env && chmod 600 .env` ; renseigner `POOL_ADMIN_TOKEN`, `REDIS_PASSWORD`
   (`openssl rand -hex 32`) et `UPSTREAM_NETWORK` (réseau Docker du service `ai-gateway`).
3. `docker compose build && docker compose up -d` ; `curl -s http://127.0.0.1:18090/health/live`.

Le Pool ne publie que son port en boucle locale ; la Gateway, Ollama et le reranker restent sur leur
réseau interne. `pool.yaml` et `.env` sont ignorés par git (`.gitignore`).
