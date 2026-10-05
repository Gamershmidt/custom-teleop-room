/* mtc_capture: body / leg tracking through WebXR, injected into the capture page.
 *
 * 1. Asks for WebXR body tracking as an OPTIONAL feature (the session still starts if the
 *    browser does not know it).
 * 2. Every XR frame, reads whatever pose data the browser offers beyond head and hands:
 *      - XRFrame.body (WebXR Body Tracking: named joints -> XRSpace), when present;
 *      - extra XRInputSources that are neither hands nor the two controllers (trackers).
 *    and streams them to the Mac over wss://<host>/mtc/body, poses in the same reference space
 *    as the head and hands (column-major 4x4, like Vuer's HAND_MOVE).
 * 3. Once a second posts a capability report to /mtc/probe: enabled features, input sources,
 *    XR-related prototype members, body joint names, stream rate.
 * Observes only; nothing Vuer does is changed apart from the optional feature.
 */
(function () {
  "use strict";
  if (typeof window === "undefined" || window.__mtcBody) return;
  window.__mtcBody = true;
  var WANT = ["body-tracking"];
  var state = { features: null, sources: [], body: null, sent: 0, frames: 0, error: null, requested: null };

  if (navigator.xr && navigator.xr.requestSession) {
    var original = navigator.xr.requestSession.bind(navigator.xr);
    navigator.xr.requestSession = function (mode, init) {
      init = Object.assign({}, init || {});
      var opt = (init.optionalFeatures || []).slice();
      WANT.forEach(function (f) { if (opt.indexOf(f) < 0) opt.push(f); });
      var augmented = Object.assign({}, init, { optionalFeatures: opt });
      state.requested = { mode: mode, required: init.requiredFeatures || [], optional: opt };
      return original(mode, augmented).catch(function (err) {
        state.error = "with body-tracking: " + err;
        return original(mode, init);          // browser refused the extra feature: start as before
      });
    };
  }

  var ws = null, wsOpen = false;
  function connect() {
    try {
      ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/mtc/body");
      ws.onopen = function () { wsOpen = true; };
      ws.onclose = function () { wsOpen = false; setTimeout(connect, 1000); };
      ws.onerror = function () { wsOpen = false; };
    } catch (e) { state.error = "ws: " + e; }
  }
  connect();

  function mat(pose) { return pose ? Array.from(pose.transform.matrix) : null; }

  if (typeof XRSession !== "undefined" && XRSession.prototype.requestAnimationFrame) {
    var raf = XRSession.prototype.requestAnimationFrame;
    var refs = new WeakMap();
    XRSession.prototype.requestAnimationFrame = function (cb) {
      var session = this;
      if (!refs.has(session)) {
        refs.set(session, null);
        session.requestReferenceSpace("local-floor").then(function (s) { refs.set(session, s); })
          .catch(function () { session.requestReferenceSpace("local").then(function (s) { refs.set(session, s); }); });
        state.features = session.enabledFeatures ? Array.from(session.enabledFeatures) : "n/a";
      }
      return raf.call(session, function (t, frame) {
        try {
          var ref = refs.get(session);
          if (ref && frame) {
            state.frames++;
            var joints = {};
            if (frame.body && typeof frame.body.forEach === "function") {
              var names = [];
              frame.body.forEach(function (space, name) {
                names.push(name);
                var m = mat(frame.getPose(space, ref));
                if (m) joints["body:" + name] = m;
              });
              state.body = names;
            }
            var srcs = [];
            Array.from(session.inputSources || []).forEach(function (src, i) {
              srcs.push({ i: i, handedness: src.handedness, mode: src.targetRayMode, profiles: src.profiles,
                          hand: !!src.hand, grip: !!src.gripSpace });
              var isHand = !!src.hand;
              var isController = !isHand && (src.handedness === "left" || src.handedness === "right")
                                 && src.targetRayMode === "tracked-pointer";
              if (!isHand && !isController) {   // anything else: candidate tracker
                var sp = src.gripSpace || src.targetRaySpace;
                var m = sp && mat(frame.getPose(sp, ref));
                if (m) joints["input:" + i + ":" + (src.profiles && src.profiles[0] || src.handedness || "?")] = m;
              }
            });
            state.sources = srcs;
            var n = Object.keys(joints).length;
            if (n && wsOpen && ws.bufferedAmount < 65536) {
              ws.send(JSON.stringify({ t: performance.now(), joints: joints }));
              state.sent++;
            }
          }
        } catch (e) { state.error = "frame: " + e; }
        return cb.call(this, t, frame);
      });
    };
  }

  function members(proto) {
    try { return Object.getOwnPropertyNames(proto).filter(function (k) { return k !== "constructor"; }); }
    catch (e) { return []; }
  }
  setInterval(function () {
    var report = {
      userAgent: navigator.userAgent, requested: state.requested, enabledFeatures: state.features,
      inputSources: state.sources, bodyJoints: state.body, framesSeen: state.frames, bodyMessagesSent: state.sent,
      error: state.error,
      xrFrameMembers: typeof XRFrame !== "undefined" ? members(XRFrame.prototype) : [],
      xrSessionMembers: typeof XRSession !== "undefined" ? members(XRSession.prototype) : [],
      bodyApi: { XRBody: typeof XRBody !== "undefined", XRBodySpace: typeof XRBodySpace !== "undefined" },
    };
    try { fetch("/mtc/probe", { method: "POST", body: JSON.stringify(report), headers: { "Content-Type": "application/json" } }); }
    catch (e) {}
  }, 1000);
})();
