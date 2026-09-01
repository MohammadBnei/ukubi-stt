// Microphone -> 16 kHz mono s16 PCM, in exactly the chunks ukubi-stt's encoder
// wants. Transport-agnostic: it knows nothing about gRPC, tokens or origins, so
// a caller can post the chunks to its own backend, which is what every consumer
// actually does (no browser holds an STT credential — ADR-0046).
//
// This exists as one file because the chunk arithmetic below carried THREE
// separate latency bugs during 0.5.x, every one of them invisible in review:
//
//   - shipping the whole buffer sent 768ms chunks instead of 560ms, and left the
//     server holding a remainder until the next request — ~1.4s and irregular
//     instead of ~650ms
//   - a 2048-frame callback added up to 128ms of jitter waiting for the block
//     that crossed the boundary
//   - the tail flush sent zero samples when the buffer landed exactly empty
//     (1 callback in 35, and every Stop before speaking), which the server then
//     rejected, losing the last words and leaking the session
//
// Writing it twice means finding the fourth one twice.

const SAMPLE_RATE = 16000;
// 8960 samples = 560ms, the streaming encoder's granularity. Smaller chunks add
// requests without lowering latency; larger ones add latency for nothing.
const CHUNK_SAMPLES = 8960;

const WORKLET_SRC = `
class Chunker extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.size = options.processorOptions.chunkSamples;
    this.buf = new Float32Array(this.size);
    this.n = 0;
    this.quanta = 0;
    this.peak = 0;
    this.port.onmessage = (e) => {
      if (e.data === 'flush') {
        // Sent even when n === 0: an empty tail is the close signal, and the
        // server pads it with silence to flush the encoder's own buffer.
        this.port.postMessage({ type: 'tail', samples: this.buf.slice(0, this.n) });
        this.n = 0;
      }
    };
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      const v = ch[i];
      if (v > this.peak) this.peak = v; else if (-v > this.peak) this.peak = -v;
    }
    // A level reading every 8 quanta (~64ms). Per-quantum would be 125 messages
    // a second to move one float.
    if (++this.quanta >= 8) {
      this.port.postMessage({ type: 'peak', value: this.peak });
      this.quanta = 0; this.peak = 0;
    }
    let i = 0;
    while (i < ch.length) {
      const take = Math.min(ch.length - i, this.size - this.n);
      this.buf.set(ch.subarray(i, i + take), this.n);
      this.n += take; i += take;
      if (this.n === this.size) {
        this.port.postMessage({ type: 'chunk', samples: this.buf.slice(0) });
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('chunker', Chunker);
`;

/** Clamp before scaling: a sample above 1.0 wraps to a large negative int16,
 *  heard as a click and read by the model as noise. */
export function toPCM16(f32) {
  const out = new Uint8Array(f32.length * 2);
  const view = new DataView(out.buffer);
  for (let i = 0; i < f32.length; i++) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return out;
}

/**
 * @param {object} opts
 * @param {(pcm: Uint8Array, last: boolean) => Promise<void>} opts.send
 *   Called once per chunk, IN ORDER, never concurrently. May reject; see below.
 * @param {(err: Error) => void} [opts.onError]
 * @param {(level: number) => void} [opts.onLevel]  0..1, ~every 64ms
 */
// Everything that can be built BEFORE the user clicks: the AudioContext and the
// compiled worklet. Neither touches the microphone, so this asks for no
// permission and turns on no recording indicator — it is safe to call on hover
// or on mount, and that is the point.
//
// Without it the click path is: fetch this module, construct a context, compile
// a worklet, THEN open the device. The first words of the sentence land in that
// gap and are simply never captured. Reported from agent-fleet's composer as
// "the start of my phrase is not transcribed", 2026-09-01.
//
// Memoised, so hover-then-click warms once and repeated hovers are free.
let warmPromise = null;

export function prewarm() {
  if (!warmPromise) {
    warmPromise = (async () => {
      // Forcing the context rate is what removes the need for a resampler: the
      // graph resamples the microphone into 16 kHz, the one rate the model takes.
      const c = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      // Built from a Blob so this module stays a single file with no sibling
      // asset to deploy alongside it.
      const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" }));
      try {
        await c.audioWorklet.addModule(url);
      } finally {
        URL.revokeObjectURL(url);
      }
      return c;
    })();
    // A failed warm must not be cached, or the button is dead until reload.
    warmPromise.catch(() => { warmPromise = null; });
  }
  return warmPromise;
}

export function createDictation({ send, onError = () => {}, onLevel = () => {} }) {
  let ctx = null, stream = null, node = null;
  let chain = Promise.resolve();
  let abandoned = false;
  let onTail = null;

  // ORDERING IS A CONTRACT, NOT AN IMPLEMENTATION DETAIL. The encoder carries
  // cache forward, so a reordered or dropped chunk corrupts everything after
  // it. One request in flight at a time is what guarantees order — do not
  // "optimise" this into parallel sends.
  //
  // The catch is inside the chain deliberately. If a rejection escaped, `chain`
  // would stay rejected forever and every subsequent .then() would short-circuit
  // — all remaining chunks vanishing with no symptom but a short transcript.
  // That is precisely the failure this module exists to prevent, so a failed
  // send abandons the stream loudly instead.
  function enqueue(pcm, last) {
    chain = chain.then(async () => {
      if (abandoned) return;
      try {
        await send(pcm, last);
      } catch (err) {
        abandoned = true;
        onError(err instanceof Error ? err : new Error(String(err)));
      }
    });
    return chain;
  }

  async function start() {
    abandoned = false;

    // Kicked off BEFORE the context work rather than after it. These two are
    // independent, and getUserMedia is the slow one — opening the device with
    // echo cancellation and AGC is 100-300ms. Serialising them behind
    // addModule() put the whole context setup in front of the microphone for
    // no reason, and every millisecond here is speech the user has already
    // said into a graph that does not exist yet.
    const micPromise = navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    // Do not let an unhandled rejection escape if the context work throws first.
    micPromise.catch(() => {});

    ctx = await prewarm();
    // This dictation owns the context now; the next prewarm() builds a new one.
    warmPromise = null;

    stream = await micPromise;

    // A context built outside a user gesture starts suspended, and a suspended
    // context runs no worklet — it would look live and capture silence. resume()
    // is called from the click that reached start(), which is what makes it
    // allowed.
    if (ctx.state === "suspended") await ctx.resume();

    node = new AudioWorkletNode(ctx, "chunker", {
      numberOfInputs: 1, numberOfOutputs: 0, channelCount: 1,
      processorOptions: { chunkSamples: CHUNK_SAMPLES },
    });

    node.port.onmessage = (e) => {
      const m = e.data;
      if (m.type === "peak") onLevel(m.value);
      else if (m.type === "chunk") enqueue(toPCM16(m.samples), false);
      else if (m.type === "tail") {
        enqueue(toPCM16(m.samples), true);
        if (onTail) { onTail(); onTail = null; }
      }
    };

    // numberOfOutputs: 0 — a worklet runs from its input alone, so unlike a
    // ScriptProcessor there is no silent gain node needed to keep it alive.
    ctx.createMediaStreamSource(stream).connect(node);
  }

  /** Flush the tail, close the session, and wait for the last send to land. */
  async function stop() {
    if (node) {
      // Ask for the partial tail and wait for it before tearing the graph down,
      // or that audio dies with the worklet and the server never gets its close.
      // Bounded, so a worklet that never answers cannot wedge the caller.
      await new Promise((resolve) => {
        const timer = setTimeout(() => { onTail = null; resolve(); }, 1000);
        onTail = () => { clearTimeout(timer); resolve(); };
        node.port.postMessage("flush");
      });
      node.port.onmessage = null;
      node.disconnect();
      node = null;
    }
    if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
    if (ctx) { await ctx.close(); ctx = null; }
    // Await the queue, so a caller that re-enables its UI after stop() is not
    // doing so while text is still arriving.
    await chain;
  }

  return { start, stop, get abandoned() { return abandoned; }, CHUNK_SAMPLES, SAMPLE_RATE };
}
