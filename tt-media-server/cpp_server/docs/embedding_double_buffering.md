# Double buffering u embedding putanji C++ servera

Grana: `jzivanovic/bge-m3-double-buffer` (commit `853fcc82`, dopuna `d74380f6`).

Cilj: preklopiti host-side rad (tokenizacija, serijalizacija, mrežni roundtrip)
sa device forward pass-om, tako da uređaj između dva batcha ne stoji ~310 ms
koliko je izmereno na BGE-M3 N300 benchmarku (E2EL ≈ 1.29 s vs. ~0.98 s čistog
forward-a).

## 1. Stanje pre izmene

Putanja jednog batcha je bila potpuno serijska, na dva nivoa:

**Parent proces** (`embedding_service.cpp`, jedan dispatch thread po workeru):

```
collectBatch()  →  JSON encode  →  write(pipe)  →  [BLOKIRA na read(pipe)]  →  decode  →  onComplete×N
```

Dispatch thread je slao batch i **blokirao dok ne stigne odgovor**
(`dispatchBatchToWorker` je zvao `sendRequest` pa odmah `receiveResponse`).
Worker zato nikad nije mogao ni da dobije batch N+1 pre nego što vrati N.

**Worker child** (`embedding_worker_main.cpp`, jedna nit):

```
read(pipe)  →  parse JSON  →  tokenize  →  forward + sync  →  extract  →  write(pipe)
```

Sav host rad (tokenizacija 12×8192 tokena, parsiranje, ekstrakcija i
serijalizacija 12×1024 float embeddinga) sedeo je na kritičnoj putanji između
dva forward-a.

## 2. Preduslov: ponašanje GIL-a u ttnn

Runner drži embedded Python interpreter (pybind11) — i tokenizacija i forward su
Python pozivi, pa je ključno pitanje da li dve C++ niti uopšte mogu da napreduju
paralelno kroz Python.

Provereno u tt-metal source-u (nanobind bindings):

- `ttnn/cpp/ttnn-nanobind/bind_function.hpp` registruje `__call__` **svake ttnn
  operacije** sa `nb::call_guard<nb::gil_scoped_release>()` — GIL se pušta dok
  operacija radi u C++-u.
- `ttnn/cpp/ttnn-nanobind/device.cpp` isto radi za `synchronize_device`, uz
  eksplicitan komentar: *"Release GIL: sync can block a long time; other
  threads need it to run Python"*.

Zaključak: dok forward nit sedi u ttnn op-u ili u `synchronize_device` (tj.
većinu trajanja forward-a), druga nit može da drži GIL i tokenizuje sledeći
batch. Bez ovoga bi dve niti samo naizmenično čekale GIL i preklapanja ne bi
bilo — zato je ovo provereno **pre** pisanja koda.

HF fast tokenizer (Rust) dodatno pušta GIL tokom batch encode-a, pa tokenizacija
ne guši ni retke Python deonice forward niti.

## 3. Nova arhitektura

Pipeline dubine 2, kroz sva tri sloja:

```
             PARENT                                WORKER CHILD
┌─────────────────────────────┐        ┌────────────────────────────────┐
│ sender thread               │  pipe  │ prepare thread                 │
│  collectBatch → encode →────┼───────►│  read → parse → prepare()      │
│  push u inFlight (max 2)    │        │  (validacija + tokenizacija)   │
│                             │        │        │ single-slot handoff   │
│ receiver thread             │        │        ▼                       │
│  pop inFlight ← decode ←────┼◄───────┼─ forward thread (main)         │
│  onComplete×N               │  pipe  │   runPrepared() → write        │
└─────────────────────────────┘        └────────────────────────────────┘
```

U ustaljenom stanju: dok uređaj melje batch N, parent je već poslao batch N+1,
a prepare nit u workeru ga je već tokenizovala. Kad se forward N završi, forward
nit odmah uzima gotov (tokenizovan) N+1 — sa kritične putanje ostaje samo
ekstrakcija/serijalizacija odgovora za N.

## 4. Izmene po fajlu

### 4.1 `include/runtime/runners/i_embedding_runner.hpp` — novi interfejs

Dodata struktura koja nosi batch između dve faze:

```cpp
struct PreparedBatch {
  std::vector<domain::EmbeddingRequest> requests;

  // Ne-prazno kada je priprema već proizvela konačne odgovore
  // (validacija ili tokenizacija pala); runPrepared() tada vraća ovo.
  std::vector<domain::EmbeddingResponse> immediate;

  // Runner-specifično host stanje (tokenizovani tenzori), tip-obrisan.
  std::shared_ptr<void> payload;
};
```

I dve nove virtuelne metode sa default implementacijama:

```cpp
virtual PreparedBatch prepare(std::vector<domain::EmbeddingRequest> requests) {
  return PreparedBatch{std::move(requests), {}, nullptr};
}

virtual std::vector<domain::EmbeddingResponse> runPrepared(PreparedBatch& batch) {
  if (!batch.immediate.empty()) return std::move(batch.immediate);
  return run(batch.requests);
}
```

Zašto default implementacije: `MockEmbeddingRunner` (i bilo koji budući runner)
radi bez izmena — `prepare` mu je no-op, `runPrepared` delegira na postojeći
`run`. Serve petlja u workeru koristi isključivo novi API, a ostaje korektna za
svaki runner.

Zašto `shared_ptr<void>` payload: header ne sme da uvuče pybind11 tipove
(postojeće pravilo u kodu — `EmbeddingImpl` je iza forward deklaracije iz istog
razloga). Konkretni tip zna samo `embedding_runner.cpp`.

### 4.2 `embedding_runner.{hpp,cpp}` — podela `runInference` na dve faze

Stari `runInference` (validacija → tokenize → forward → extract, sve pod jednim
GIL acquire-om) podeljen je na:

**`prepareInference`** (zove se sa prepare niti):
1. `py::gil_scoped_acquire`.
2. Validacija model id-a po zahtevu — mismatch bilo gde failuje ceo batch
   (isti ugovor kao ranije): puni `prepared.immediate` error odgovorima i vraća
   se bez tokenizacije.
3. Tokenizacija: `tokenizer(texts, padding=true, truncation=true,
   max_length=max_seq_len, return_tensors="pt")`; rezultat ide u payload.
4. Ako tokenizacija baci Python izuzetak, prva linija greške ide u
   `immediate` odgovore za sve zahteve (isti stil kao stari error handling).

**`runPrepared`** (zove se sa forward/main niti):
1. Ako `immediate` nije prazno — vrati ga, bez dodira uređaja.
2. `py::gil_scoped_acquire`, pa `forwardAndSync(tokenized)` → `extractDense` →
   attention-mask token counts → sklapanje `EmbeddingResponse` po redu
   (`responses[i]` odgovara `requests[i]`, pozicija je ugovor).
3. Python izuzetak → error odgovori za ceo batch, kao ranije.

**GIL i životni vek Python objekata.** Tokenizovani tenzori nastaju na jednoj
niti a troše se i umiru na drugoj. Refcount operacije py::object-a smeju samo
pod GIL-om, a `PreparedBatch` se uništava u čistom C++ kontekstu (bez GIL-a).
Zato payload nije goli `py::object` nego:

```cpp
struct TokenizedPayload {
  py::object tokenized;
  ~TokenizedPayload() {
    if (!Py_IsInitialized()) return;
    py::gil_scoped_acquire gil;   // destruktor može stići sa bilo koje niti
    tokenized = py::object();
  }
};
```

`EmbeddingRunner::run()` je sada kompozicija `prepare` + `runPrepared`
(koristi ga samo warmup/legacy pozivalac), pa nema duplirane logike.
`EmbeddingRunner::prepare()` dodatno pokriva "runner nije inicijalizovan" —
vraća `immediate` greške umesto praznog rezultata.

### 4.3 `embedding_worker_main.cpp` — dvo-nitna serve petlja

Stara petlja (read→run→write u jednoj niti) zamenjena je sa dve niti spojene
**single-slot handoff-om** (`std::optional<PreparedBatch> slot` + mutex + cv):

- **prepare nit**: `pipeReadString` (blokira bez GIL-a) → `parseBatch` →
  `runner.prepare(...)` → čeka da se slot isprazni → `slot.emplace(...)` →
  notify. Na EOF pipe-a postavlja `eof = true` i notify.
- **forward nit** (originalna nit `serveLoop`-a): čeka `slot || eof` → uzme
  batch iz slota (notify da prepare može dalje) → `runner.runPrepared(...)` →
  `encodeResponses` → `pipeWrite`.

Zašto slot kapaciteta tačno 1: dubina pipeline-a je 1 batch na uređaju + 1
pripremljen. Dublji red ne pomaže (uređaj je usko grlo) a povećava latenciju
zahteva koji čame tokenizovani.

Redosled na gašenju: prepare nit vidi EOF tek pošto je poslednji batch gurnula
u slot, pa forward nit prvo obradi slot (predikat `slot || eof`, a `break` samo
kad je slot prazan), tek onda izađe; `prepareThread.join()` na kraju.

FIFO garancija: jedna prepare nit i jedna forward nit, slot je red dužine 1 —
redosled batcheva se ne može preokrenuti.

### 4.4 `embedding_service.cpp` + `embedding_worker_process.hpp` — parent

Bez ove izmene worker ne može ni dobiti batch N+1, pa je dispatch po workeru
podeljen na dve niti:

**Novo stanje po workeru:**

```cpp
static constexpr size_t kMaxBatchesInFlight = 2;

struct InFlightState {
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<std::vector<std::shared_ptr<PendingRequest>>> batches;  // send redosled
};
```

**Sender** (`workerDispatchLoop`, postojeći `dispatchThread`):
1. Čeka na cv dok `inflight.batches.size() < 2` (ili shutdown/worker mrtav).
2. `collectBatch` — linger semantika (`MAX_BATCH_DELAY_TIME_MS`) važi samo kad
   je worker besposlen; pri prefetch-u se parcijalni batchevi ne šalju
   (detalji u §10).
3. `checkAlive` + `sendRequest(encodeBatchJson(batch))`; na grešku failuje batch
   i nastavlja.
4. Push batcha u `inflight.batches` + notify. Veličina batcha se hvata **pre**
   move-a (pristup `batches.back()` posle push-a bio bi trka sa receiverom).
5. Na izlazu iz petlje: `inflight.cv.notify_all()` — budi receiver da vidi da je
   `isReady`/`running` palo (bez ovoga bi receiver visio do `stop()`).

**Receiver** (`workerReceiveLoop`, novi `receiveThread` u `WorkerProcess`):
1. Čeka na cv dok ne postoji batch u letu (ili shutdown) — **ne sme** da bloki
   na `receiveResponse` dok ništa nije poslato, inače bi na gašenju visio na
   read-u praznog pipe-a.
2. `receiveResponse()` (blokirajući read; na grešku čisti `isReady`).
3. Pop najstarijeg batcha iz deque-a + notify (oslobađa slot senderu).
4. Prazan odgovor → `failBatch("Failed to read response from worker")`; sledeća
   iteracija drenira eventualni drugi batch u letu (read vraća EOF jer je child
   mrtav) pa petlja izlazi kroz predikat. Neprazan → `decodeResponses` i
   `onComplete`/`completeWithError` po task_id-u.

Uparivanje batch↔odgovor je FIFO: pipe je serijski, worker obrađuje redom, pa
odgovori stižu redom slanja. Task_id lookup unutar batcha ostaje kao zaštita —
pomešani batchevi bi dali "Response not found for task_id", ne pogrešan vektor.

`dispatchBatchToWorker` (stari send+wait) je obrisan. `DispatchStats` sada meri
queue-wait i trajanje write-a (roundtrip više ne postoji na sender niti);
throughput formula je izbačena iz debug loga jer bi bila besmislena.

**`stop()` redosled** (bitno da se ništa ne zaglavi):
1. `running = false`; notify `queueCv` + svih `inflight.cv`.
2. Join svih sender niti (bude se kroz predikate u `collectBatch`/inflight wait).
3. Po workeru: `terminate()` (gasi child → EOF na response pipe-u budi receiver
   zaglavljen u read-u) pa join receiver niti.

### 4.5 Popravka pre-postojećeg shutdown deadlocka (commit `d74380f6`)

Smoke test je otkrio da se server (i pre ove grane!) ne gasi čisto:

- Fork-ovani worker **nasleđuje Drogon-ov SIGTERM handler** od parenta. U
  detetu taj handler samo "gurne" event petlju koja tamo ne postoji — dete
  efektivno ignoriše SIGTERM (potvrđeno preko `SigCgt` maske u
  `/proc/<child>/status` i wchan-a niti).
- `terminate()` je radio `kill(SIGTERM)` → **blokirajući `waitpid`** → tek onda
  `writeFd.reset()`. Dete čeka EOF na request pipe-u koji parent i dalje drži
  otvorenim; parent čeka smrt deteta koje ne reaguje na signal. Klasičan
  međusobni deadlock (parent u `do_wait`, dete u `pipe_wait`/`futex`).

Popravka na dva mesta (belt and braces):

```cpp
// embedding_worker_main.cpp, ulaz u workerProcessMain:
signal(SIGTERM, SIG_DFL);
signal(SIGINT, SIG_DFL);

// embedding_worker_process.cpp, terminate():
writeFd.reset();          // prvo EOF — worker blokiran na read-u izlazi sam
const pid_t p = pid.load();
if (p > 0) { kill(p, SIGTERM); waitpid(p, nullptr, 0); ... }
readFd.reset();
```

Sa vraćenom default dispozicijom SIGTERM ubija dete; sa pipe-om zatvorenim pre
signala postoji i graceful put (EOF → serve petlja izađe → `runner->close()` →
`_exit(0)`).

## 5. Prateća izmena benchmark parametara

`workflows/model_specs/dev/embedding.yaml`, BGE-M3 N300:

```yaml
BENCHMARK_MAX_CONCURRENCY: "24"   # bilo 12
BENCHMARK_NUM_PROMPTS: "120"      # bilo 60
BENCHMARK_NUM_WARMUPS: "24"       # bilo 12
```

Razlog: closed-loop test sa concurrency == batch size (12) ne može da napuni
pipeline — dok uređaj radi, svi klijenti čekaju odgovore i red je prazan, pa
prepare nit nema šta da tokenizuje. Sa 24 klijenta uvek postoji drugi pun batch
u redu. `MAX_BATCH_SIZE` ostaje 12 (to je device batch, fiksiran trace-om);
`MAX_BATCH_DELAY_TIME_MS=300` ostaje kao gornja granica za tail batcheve — u
punom režimu se ne troši jer se batch napuni odmah.

## 6. Sinhronizacioni invarianti (rezime)

| Invariant | Mehanizam |
|---|---|
| ≤ 2 batcha u letu po workeru | sender cv-wait na `batches.size() < 2` |
| Parcijalni batch samo kad je worker besposlen | prefetch gating u `collectBatch` (§10) |
| ≤ 1 pripremljen batch u workeru | single-slot handoff (`optional` + cv) |
| FIFO batch ↔ odgovor | serijski pipe + jedna nit po strani + deque |
| py::object refcount samo pod GIL-om | `TokenizedPayload` destruktor uzima GIL |
| GIL se ne drži tokom device rada | ttnn `gil_scoped_release` (provereno u tt-metal) |
| Receiver ne bloki na read bez batcha u letu | cv predikat pre `receiveResponse` |
| Gašenje ne visi | signal reset u detetu + `writeFd.reset()` pre `waitpid` |

## 7. Verifikacija

- Build: `tt_media_server_cpp`, `llm_service_test` (deli iste source fajlove) —
  čisto; `clang-format-20` čist.
- Smoke test sa mock runnerom (`MODEL_RUNNER_TYPE=embedding_mock`):
  - 40 konkurentnih zahteva → 40/40 ispravnih odgovora; determinističke mock
    vektore smo uporedili po ulazu (isti ulaz → identičan vektor, različit ulaz
    → različit), čime je potvrđeno da uparivanje batch↔odgovor ne meša zahteve.
  - Log potvrđuje preklapanje: `Preparing batch` za N+1 se dešava dok
    `Processing batch` za N još traje.
  - SIGTERM → receiver i dispatch niti izađu, `Worker 0 terminated`
    (waitpid prošao), `Stopped`, nula zaostalih procesa.
- Stvarni test preklapanja tokenizacije i forward-a moguć je tek na N300 u CI
  (mock nema uređaj, forward mu je trivijalan).

## 8. Očekivani efekat i šta ostaje serijsko

Sa c=24 i duplim baferom, od ~310 ms host overhead-a po ciklusu iza forward-a
se sakrivaju: klijentski roundtrip + HTTP, JSON encode/decode u parentu,
parsiranje i tokenizacija u workeru. Serijski ostaju samo readback rezultata sa
uređaja i sklapanje/serijalizacija odgovora (deo `runPrepared`). Realno
očekivanje za BGE-M3 N300: sa izmerenih 9.2 req/s ka ~11–11.5 req/s, uz
bare-metal plafon ~12.1 req/s.

## 9. Ograničenja / napomene

- Dubina 2 je konstanta (`kMaxBatchesInFlight`), nije konfigurisabilna — svesna
  odluka, dublji pipeline ne pomaže dok je uređaj usko grlo.
- Preklapanje zavisi od toga da ttnn pušta GIL; ako bi neka buduća verzija
  tt-metal-a to promenila, kod ostaje korektan ali degradira na serijsko
  ponašanje (nema trke, samo nema dobitka).
- Warmup retry (`EMBEDDING_WARMUP_MAX_RETRIES`, isti commit `d74380f6`) je
  logički nezavisna izmena za BGE-large PCC flakiness — respawn palih workera
  do 3 runde tokom startupa.

## 10. Dopuna: nalaz sa bge-large Galaxy benchmarka i gating parcijalnih batcheva

### 10.1 Šta je izmereno

Prvi Galaxy benchmark bge-large na ovoj grani (32 workera, `MAX_BATCH_SIZE=8`,
512 klijenata, 20000 zahteva) dao je **isti throughput kao pre duplog bafera**,
a batchevi su iz konstantnih osmica postali neravnomerni — osmice izmešane sa
parcijalama od 1 do 7 (`batch_timeout=2ms` u logu: Galaxy spec ne postavlja
`MAX_BATCH_DELAY_TIME_MS`, pa važi default od 2 ms).

### 10.2 Uzrok: apsorpcioni kapacitet je izjednačen sa konkurentnošću

Zašto su ranije batchevi bili konstantno puni: stari dispatch thread je
**blokirao na odgovoru** — dok batch od 8 radi forward, taj worker ne uzima
ništa iz reda. Apsorpcija servera je bila 32 × 8 = **256** zahteva; sa 512
klijenata u redu je stalno stajao backlog od ~256, pa je svaki `collectBatch`
zatekao pun red i uzeo tačno 8. Serijski dispatch je, nenamerno, bio mehanizam
koji puni batcheve.

Sa duplim baferom svaki worker drži do 2 batcha u letu: apsorpcija je 32 × 16 =
**512** — tačno jednako broju klijenata. Red je zato skoro uvek prazan; kada
sender krene da skupi *drugi* (prefetch) batch dok prvi radi forward, zatekne
šta god je slučajno stiglo, sačeka 2 ms lingera i pošalje parcijalu.

Zašto to poništava dobitak: kernel je fiksnog oblika — batch od 3 košta uređaj
isto koliko i batch od 8, a praznih 5 slotova se ne može naknadno popuniti.
Prosečna veličina batcha padne sa 8 na ~5–6, pa uređaj melje više (delimično
praznih) batcheva za isti broj zahteva. Dobitak od preklapanja i gubitak od
paddinga se približno ponište → isti TPUT.

### 10.3 Popravka: prefetch samo punih batcheva

`collectBatch` sada ima dva režima, po tome da li worker već ima batch u letu
(`inFlight[workerIdx].batches` neprazan):

- **Worker besposlen** (ništa u letu): batch ide pravo na forward, pa je
  parcijala bolja od praznog uređaja. Ostaje postojeća linger semantika:
  čekaj do `MAX_BATCH_DELAY_TIME_MS` od dolaska najstarijeg zahteva, pa šalji
  šta ima.
- **Prefetch** (≥1 batch u letu): **parcijala se ne šalje nikad**. Sender čeka
  dok se ili ne formira pun batch u redu, ili tekući batch ne završi (receiver
  ga skine iz `inFlight` → worker je besposlen → gornja pravila preuzimaju; ako
  je linger deadline najstarijeg zahteva u međuvremenu istekao, parcijala
  odlazi odmah).

Ključno svojstvo: čekanje u prefetch režimu **ne dodaje latenciju nijednom
zahtevu** u odnosu na serijski dispatch — prefetch batch ionako čeka iza
tekućeg forward-a, pa je "sačekaj da se napuni dok uređaj radi" strogo bolje.
Zato ovo nije konfigurabilno: ispravno je za svaki model i uređaj.

Sinhronizacija: receiver posle skidanja batcha notifikuje i `queueCv` (ne samo
`inflight.cv`), da sender parkiran u "čekam pun batch" odmah vidi prelazak u
besposleni režim. `prefetching()` predikat uzima `inflight.mutex` unutar čekanja
na `queueMutex` — redosled zaključavanja je bezbedan jer se obrnuto ugnježdenje
nigde ne dešava; a pošto samo sender puni `inFlight`, `false` rezultat ne može
da se promeni tokom jednog collect-a (nema trke ka parcijali).

### 10.4 Posledice po benchmark parametre

- **`MAX_BATCH_DELAY_TIME_MS` ne treba dizati.** Linger sada važi samo kad je
  uređaj besposlen, a tada je kratak linger ispravan izbor (latencija). U punom
  režimu batcheve puni gating, ne linger.
- **Konkurentnost:** minimum za pun pipeline je `workers × batch × 2` (za
  bge-large Galaxy tačno 512). Na tom minimumu red radi "na nuli", pa sender
  ponekad čeka da se prefetch batch napuni umesto da već bude spreman; malo
  headroom-a (npr. 768–1024) garantuje da prefetch uvek zatekne pun batch.
  Bitno: sa gatingom ni 512 ne može biti *gore* od starog serijskog ponašanja —
  najgori slučaj degradira na puni batch + kratka pauza uređaja, tj. staro
  stanje.

### 10.5 Verifikacija

Smoke test sa mock runnerom (1 worker, batch 8, linger 50 ms): 20 konkurentnih
zahteva → batchevi 8, 8, 4 (poslednja parcijala tek po isteku lingera, uređaj
besposlen), 20/20 ispravnih odgovora, SIGTERM gašenje čisto (obe niti izađu,
`Worker 0 terminated`, `Stopped`, nula zaostalih procesa).
