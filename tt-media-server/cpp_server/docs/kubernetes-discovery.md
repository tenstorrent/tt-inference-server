# Kubernetes discovery backend (`DYNAMO_DISCOVERY_BACKEND=kubernetes`)

The cpp_server Dynamo worker can register itself for discovery either over **etcd**
(default) or **Kubernetes-natively**. With the kubernetes backend the worker
server-side-applies a `DynamoWorkerMetadata` custom resource (group
`nvidia.com`, version `v1alpha1`) to the in-cluster API server, wire-compatible
with NVIDIA Dynamo's native kubernetes discovery — so the etcd StatefulSet can be
dropped.

The CR bundles the same per-instance JSON the etcd backend writes (`endpoints`
and `model_cards` maps under `spec.data`, keyed by
`<namespace>/<component>/<endpoint>/<instance_id_hex>`), plus a Pod
`ownerReference` for garbage collection. Everything that works with etcd today
(including `ROUTER_MODE=kv`) works unchanged — only the discovery transport
differs.

## Worker configuration (this repo)

Set on the cpp_server container:

| Env var | Default | Purpose |
|---|---|---|
| `DYNAMO_DISCOVERY_BACKEND` | `etcd` | Set to `kubernetes` to use this backend. |
| `DYNAMO_KUBE_API_SERVER` | `https://$KUBERNETES_SERVICE_HOST:$KUBERNETES_SERVICE_PORT` | API server URL. Auto-derived in-cluster. |
| `DYNAMO_KUBE_TOKEN_PATH` | `/var/run/secrets/kubernetes.io/serviceaccount/token` | ServiceAccount bearer token (re-read per request for rotation). |
| `DYNAMO_KUBE_VALIDATE_CERT` | `true` | Validate the API server TLS cert (see TLS note below). |
| `POD_NAME` | — | **Required.** CR name (pod mode) + owner reference. Downward API. |
| `POD_UID` | — | **Required.** Owner reference (enables GC). Downward API. |
| `POD_NAMESPACE` | SA namespace file → `DYNAMO_NAMESPACE` | Namespace the CR is created in. Downward API. |
| `POD_IP` | (interface auto-detect) | Advertised request-plane dial-back address. Downward API. |

`DYNAMO_NAMESPACE` / `DYNAMO_COMPONENT` / `DYNAMO_ENDPOINT_NAME` are shared with
the etcd backend and unchanged.

## Deployment requirements (tt-orchestration)

**Topology:** run the Dynamo **frontend via a `DynamoGraphDeployment` (DGD)**
(operator sets its `DYN_DISCOVERY_BACKEND=kubernetes` + reader RBAC), and keep
**tt-inference-server as a self-managed workload** (Deployment / JobSet). Because
the worker is *not* a DGD-managed pod, the operator injects none of the discovery
wiring — add it by hand on the worker pod:

### 1. Downward-API env (worker container)
```yaml
env:
  - name: DYNAMO_DISCOVERY_BACKEND
    value: "kubernetes"
  - name: DYN_KUBE_DISCOVERY_MODE      # pod mode
    value: "pod"
  - name: POD_NAME
    valueFrom: { fieldRef: { fieldPath: metadata.name } }
  - name: POD_UID
    valueFrom: { fieldRef: { fieldPath: metadata.uid } }
  - name: POD_NAMESPACE
    valueFrom: { fieldRef: { fieldPath: metadata.namespace } }
  - name: POD_IP
    valueFrom: { fieldRef: { fieldPath: status.podIP } }
```
Drop `DYNAMO_ETCD_ENDPOINTS`.

### 2. Pod labels (so the frontend's daemon discovers the pod)
```yaml
metadata:
  labels:
    nvidia.com/dynamo-discovery-backend: "kubernetes"
    nvidia.com/dynamo-discovery-enabled: "true"
```

### 3. RBAC (worker ServiceAccount)
```yaml
kind: Role
rules:
  - apiGroups: ["nvidia.com"]
    resources: ["dynamoworkermetadatas"]
    verbs: ["create", "get", "list", "watch", "update", "patch", "delete"]
```
Bind it to the worker's ServiceAccount and set `serviceAccountName` on the pod.

### 4. Service for readiness
Create a Service selecting the worker pod(s) by the discovery labels, targeting
the health port (8000). The daemon reads the pod's Ready condition + `targetRef`
(pod name) from the resulting EndpointSlice. The Service port is irrelevant to
Dynamo routing (the frontend dials the CR's `transport.tcp` at the pod IP
directly) — it only needs to produce an EndpointSlice with pod readiness.

### 5. readinessProbe (worker rank-0 pod)
```yaml
readinessProbe:
  httpGet: { path: /health, port: 8000 }
```
Without it, a hung-but-alive server stays Ready in the EndpointSlice and keeps
receiving traffic. The readinessProbe is what makes the k8s deregistration path
fire on failure rather than only on pod deletion.

### Frontend via DGD
Declare a **frontend-only** `DynamoGraphDeployment` (no backend/worker component
— the frontend learns the model from the discovered CR's Model card). Use the
custom `Dockerfile.frontend` image. Drop the standalone frontend Deployment, its
`ETCD_ENDPOINTS`, and the `wait-for-etcd` initContainer. Keep the event-plane
(ZMQ) config as-is — discovery migration doesn't touch kv-routing.

## Critical constraints

- **Same namespace:** the worker workload MUST run in the namespace the DGD
  frontend watches (the daemon watches only its own namespace).
- **Multi-node:** a plain `Deployment` implies single-node serving. For
  multi-node models keep a JobSet/LeaderWorkerSet, and apply the discovery
  labels + Service selector + CR **only to the rank-0 pod** (the one running the
  Dynamo endpoint) — the other ranks must not register.
- **Deregistration is Kubernetes-owned:** the worker does not depend on
  cooperative unregister. Removal happens via (1) EndpointSlice readiness and
  (2) owner-reference GC when the pod is deleted. `unregisterSelf()` (a
  best-effort CR delete on graceful shutdown) is only a fast-path.

## Request-plane port (`DYNAMO_BIND_PORT`)

The Dynamo TCP listener binds an OS-assigned port and advertises it in the CR.
For basic connectivity nothing extra is needed — the frontend dials the pod IP on
that port over the flat pod network. **If NetworkPolicies restrict pod-to-pod
traffic**, pin the port so it can be allow-listed: set `DYNAMO_BIND_PORT=<fixed>`,
declare a matching `containerPort`, and add a NetworkPolicy allowing
frontend→worker on that port. Otherwise the request-plane connect is silently
dropped while discovery still looks healthy (CR present + pod Ready), and requests
hang. (Note: `DYNAMO_BIND_PORT` is only wired on the branch where port pinning
was added; on branches without it the port is always OS-assigned.)

## TLS / CA validation

Validation works **out of the box** — no deployment config required. drogon
1.9.12's `HttpClient` has no per-client trusted-CA setter, so when
`DYNAMO_KUBE_VALIDATE_CERT=true` (default) the worker points OpenSSL's default
trust store at the mounted ServiceAccount CA by setting `SSL_CERT_FILE=`
`/var/run/secrets/kubernetes.io/serviceaccount/ca.crt` before creating the client
(`SSL_CTX_set_default_verify_paths`, which trantor calls for `validateCert=true`,
honors it). This is process-wide but safe: the worker's only TLS client is the
API-server call (the Dynamo request plane is plain TCP). It only fires when the CA
file exists and `SSL_CERT_FILE` is not already set.

Overrides:

- **Custom CA / nonstandard mount:** set `SSL_CERT_FILE` yourself in the pod env —
  the worker detects it's already set and honors it (does not overwrite).
- **Bake into the image trust store** (`update-ca-certificates`): also works;
  `SSL_CERT_FILE` simply isn't needed then.
- **Skip validation (in-cluster only):** `DYNAMO_KUBE_VALIDATE_CERT=0`. Acceptable
  because the API server is reached over the pod network; logged as a warning at
  startup.

## Verify

1. `kubectl get crd dynamoworkermetadatas.nvidia.com` — CRD installed (operator present).
2. Deploy the worker; then
   `kubectl get dynamoworkermetadatas -n <ns> -o yaml` — the CR exists with the
   Pod owner reference and populated `spec.data.endpoints` / `model_cards`.
3. `kubectl delete pod <worker>` — the CR is garbage-collected.
4. Frontend (DGD, `DYN_DISCOVERY_BACKEND=kubernetes`): `GET /v1/models` lists the
   model; a `/v1/chat/completions` request routes end-to-end. kv-routing works
   unchanged (it uses the worker-advertised id).
