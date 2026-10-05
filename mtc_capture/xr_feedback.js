/* mtc_capture: sound cues for the operator, injected into the capture page.
 * The capture pushes events over wss://<host>/mtc/events; each is a short WebAudio synth sound
 * (no audio files). Browsers only allow audio after a user action, so the AudioContext is created
 * or resumed on the first tap/click and when the XR session is requested ("Virtual Reality").
 */
(function () {
  "use strict";
  if (typeof window === "undefined" || window.__mtcFeedback) return;
  window.__mtcFeedback = true;
  var ctx = null;
  function audio() {
    if (!ctx) {
      var AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      ctx = new AC();
    }
    if (ctx.state === "suspended") ctx.resume();
    return ctx;
  }
  ["pointerdown", "click", "touchstart", "keydown"].forEach(function (ev) {
    window.addEventListener(ev, audio, { capture: true, passive: true });
  });
  if (navigator.xr && navigator.xr.requestSession) {
    var req = navigator.xr.requestSession.bind(navigator.xr);
    navigator.xr.requestSession = function () { audio(); return req.apply(null, arguments); };
  }

  // tone(freq Hz, start offset s, duration s, type, gain)
  function tone(f, at, dur, type, gain) {
    var c = audio();
    if (!c) return;
    var t = c.currentTime + (at || 0), o = c.createOscillator(), g = c.createGain();
    o.type = type || "sine";
    o.frequency.setValueAtTime(f, t);
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(gain || 0.25, t + 0.01);
    g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
    o.connect(g).connect(c.destination);
    o.start(t);
    o.stop(t + dur + 0.02);
  }
  var SOUNDS = {
    calibrated: function () { tone(523, 0, .15); tone(659, .12, .15); tone(784, .24, .25); },
    refused:    function () { tone(220, 0, .25, "square", .15); tone(185, .28, .35, "square", .15); },
    new_segment:function () { tone(660, 0, .12); tone(660, .16, .12); },
    arming:     function () { tone(880, 0, .08, "sine", .18); },
    tick:       function () { tone(880, 0, .08, "sine", .18); },
    start:      function () { tone(660, 0, .12); tone(990, .12, .3); },
    touch:      function () { tone(150, 0, .35, "sawtooth", .3); },
    warn:       function () { tone(1200, 0, .04, "triangle", .08); },
    hand_lost:  function () { tone(330, 0, .12, "square", .15); tone(330, .2, .12, "square", .15); },
    success:    function () { tone(523, 0, .12); tone(659, .1, .12); tone(784, .2, .12); tone(1047, .3, .35); },
    unsafe:     function () { tone(523, 0, .15); tone(392, .16, .35, "triangle"); },
    abort:      function () { tone(440, 0, .15, "square", .15); tone(330, .16, .15, "square", .15); tone(220, .32, .3, "square", .15); },
  };

  function connect() {
    var ws;
    try {
      ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/mtc/events");
    } catch (e) { return; }
    ws.onmessage = function (m) {
      try {
        var ev = JSON.parse(m.data).event;
        if (SOUNDS[ev]) SOUNDS[ev]();
      } catch (e) {}
    };
    ws.onclose = function () { setTimeout(connect, 1000); };
  }
  connect();
})();
