// MTC capture layer for the XRoboToolkit Unity client (PICO 4 Ultra).
//
// Starts by itself after the first scene loads (no scene edits). Every frame it streams head,
// controllers and PICO body tracking (24 joints) to the capture computer over ws://<host>:8013/mtc,
// and draws what the capture sends back: the room, the status panel, sound cues.
//
// Frame on the wire: WebXR / OpenXR basis (x right, y up, z back), metres, floor-level origin,
// poses as x,y,z,qx,qy,qz,qw. Unity is left-handed (z forward), so poses are converted once here
// (p -> x,y,-z; q -> -qx,-qy,qz,qw) and room matrices once in MtcScene.
//
// Host address: mtc_host.txt in Application.persistentDataPath ("192.168.1.20" or "192.168.1.20:8013"),
// otherwise the capture's UDP beacon ("MTC_CAPTURE <port>" on port 8014).
// Protocol: mtc_capture/native_app.py.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using LitJson;
using UnityEngine;
using UnityEngine.XR;

namespace MtcCapture
{
    public class MtcLink : MonoBehaviour
    {
        public const int DefaultPort = 8013;
        const int BeaconPort = 8014;
        const string Version = "1";

        static MtcLink _instance;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Boot()
        {
            if (_instance != null) return;
            var go = new GameObject("MtcCapture");
            DontDestroyOnLoad(go);
            _instance = go.AddComponent<MtcLink>();
        }

        // connection (background threads)
        string _fileHost;
        int _filePort = DefaultPort;
        volatile string _beaconHost;
        volatile int _beaconPort = DefaultPort;
        volatile string _status = "looking for the capture computer";
        volatile bool _connected;
        CancellationTokenSource _cts;
        UdpClient _udp;
        AndroidJavaObject _multicastLock;
        readonly ConcurrentQueue<string> _inbox = new ConcurrentQueue<string>();
        readonly ConcurrentQueue<string> _outbox = new ConcurrentQueue<string>();
        readonly ConcurrentQueue<byte[]> _povOut = new ConcurrentQueue<byte[]>();   // POV frames (MtcPov)
        string _latestTrack;
        readonly SemaphoreSlim _wake = new SemaphoreSlim(0);

        // main thread
        MtcScene _scene;
        MtcHud _hud;
        MtcSounds _sounds;
        MtcBody _body;
        MtcThirdPerson _third;
        string _origin = "unknown";
        bool _uiHidden;
        readonly List<Canvas> _hiddenCanvases = new List<Canvas>();
        readonly StringBuilder _sb = new StringBuilder(8192);
        float _nextOriginCheck;

        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
        static readonly long UnixEpochTicks = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc).Ticks;

        void Start()
        {
            ReadHostFile();
            _scene = new MtcScene(transform);
            _hud = new MtcHud(transform);
            _sounds = gameObject.AddComponent<MtcSounds>();
            _body = new MtcBody();
            gameObject.AddComponent<MtcPov>().Init(this);
            _third = gameObject.AddComponent<MtcThirdPerson>();
            _third.Init(this);
            SetFloorOrigin();
            StartBeaconListener();
            _cts = new CancellationTokenSource();
            Task.Run(() => RunAsync(_cts.Token));
        }

        void OnDestroy()
        {
            _cts?.Cancel();
            try { _udp?.Close(); } catch (Exception) { }
            try { _multicastLock?.Call("release"); } catch (Exception) { }
        }

        void Update()
        {
            var cam = XrCamera();
            var space = cam != null ? cam.transform.parent : null;
            _scene.SetParent(space);
            _hud.SetParent(space);

            for (int n = 0; n < 50 && _inbox.TryDequeue(out var text); n++)
                Handle(text);

            if (Time.unscaledTime > _nextOriginCheck)
            {
                _nextOriginCheck = Time.unscaledTime + 5f;
                SetFloorOrigin();
            }
            _body.Tick();

            bool headOk = HeadPose(cam, out var headPos, out var headRot);
            _hud.Place(headPos, headRot, Time.unscaledDeltaTime);
            _hud.SetStatus(_connected ? null : "MTC capture\n" + _status + "\n\n" + _body.Text);
            SetUiHidden(_connected);

            if (_connected)
            {
                Interlocked.Exchange(ref _latestTrack, TrackJson(headPos, headRot, headOk));
                _wake.Release();
            }
        }

        public bool Connected => _connected;
        public Camera XrCam => _xrCam;
        public static long NowNs() => (DateTime.UtcNow.Ticks - UnixEpochTicks) * 100L;

        /// Any thread: queue a POV frame ('P' + headset time ns + JPEG); old frames are dropped if
        /// the network falls behind, so tracking frames are never delayed by video.
        public void SendPov(long stampNs, byte[] jpg)
        {
            if (!_connected) return;
            var msg = new byte[9 + jpg.Length];
            msg[0] = (byte)'P';
            BitConverter.GetBytes(stampNs).CopyTo(msg, 1);   // little-endian on Android
            Buffer.BlockCopy(jpg, 0, msg, 9, jpg.Length);
            while (_povOut.Count >= 3 && _povOut.TryDequeue(out _)) { }
            _povOut.Enqueue(msg);
            _wake.Release();
        }

        // ---------------------------------------------------------------- incoming

        void Handle(string text)
        {
            if (text == "_connected")
            {
                _outbox.Enqueue(HelloJson());
                _wake.Release();
                return;
            }
            if (text == "_disconnected")
            {
                _third.SetOn(false);
                _scene.Clear();
                _hud.ClearPanel();
                return;
            }
            JsonData d;
            try { d = JsonMapper.ToObject(text); }
            catch (Exception e) { Debug.LogWarning("MTC: bad message " + e.Message); return; }
            if (!d.IsObject || !d.ContainsKey("type")) return;
            switch ((string)d["type"])
            {
                case "scene":
                    _scene.Apply("scene", d["els"]);
                    break;
                case "xr":
                    _scene.Apply("xr", d["els"]);
                    break;
                case "hud":
                    _hud.SetPanel(Convert.FromBase64String((string)d["png"]), d.ContainsKey("layout") ? d["layout"] : null);
                    break;
                case "event":
                    _sounds.Play((string)d["name"]);
                    break;
                case "cmd":
                    Command(d);
                    break;
            }
        }

        void Command(JsonData d)
        {
            switch ((string)d["name"])
            {
                case "body_calibrate": _body.Calibrate(); break;
                case "body_start": _body.Restart(); break;
                case "third_person": _third.SetOn(d.ContainsKey("on") && (bool)d["on"]); break;
            }
        }

        // ---------------------------------------------------------------- outgoing

        string HelloJson()
        {
            var sb = new StringBuilder();
            sb.Append("{\"type\":\"hello\",\"app\":\"xrobotoolkit-mtc\",\"version\":\"").Append(Version)
              .Append("\",\"device\":\"pico\",\"model\":\"").Append(Escape(SystemInfo.deviceModel))
              .Append("\",\"os\":\"").Append(Escape(SystemInfo.operatingSystem))
              .Append("\",\"origin\":\"").Append(_origin).Append("\"}");
            return sb.ToString();
        }

        string TrackJson(Vector3 headPos, Quaternion headRot, bool headOk)
        {
            var sb = _sb;
            sb.Length = 0;
            sb.Append("{\"type\":\"track\",\"t\":").Append((DateTime.UtcNow.Ticks - UnixEpochTicks) * 100L)   // unix ns
              .Append(",\"origin\":\"").Append(_origin).Append("\",\"cam\":\"").Append(Escape(_xrCamName))
              .Append("\",\"head_ok\":").Append(headOk ? "true" : "false").Append(",\"head\":");
            Pose(sb, headPos, headRot);
            sb.Append(",\"ctrl\":{\"left\":");
            Controller(sb, XRNode.LeftHand);
            sb.Append(",\"right\":");
            Controller(sb, XRNode.RightHand);
            sb.Append("},\"body\":");
            _body.Append(sb);
            sb.Append('}');
            return sb.ToString();
        }

        static void Controller(StringBuilder sb, XRNode node)
        {
            var dev = InputDevices.GetDeviceAtXRNode(node);
            bool ok = dev.isValid;
            ok &= dev.TryGetFeatureValue(CommonUsages.isTracked, out bool tracked) && tracked;
            if (dev.TryGetFeatureValue(CommonUsages.trackingState, out InputTrackingState st))
                ok &= (st & InputTrackingState.Position) != 0;
            dev.TryGetFeatureValue(CommonUsages.devicePosition, out Vector3 p);
            dev.TryGetFeatureValue(CommonUsages.deviceRotation, out Quaternion q);
            dev.TryGetFeatureValue(CommonUsages.trigger, out float trigger);
            dev.TryGetFeatureValue(CommonUsages.grip, out float grip);
            dev.TryGetFeatureValue(CommonUsages.primaryButton, out bool primary);
            dev.TryGetFeatureValue(CommonUsages.secondaryButton, out bool secondary);
            dev.TryGetFeatureValue(CommonUsages.menuButton, out bool menu);
            dev.TryGetFeatureValue(CommonUsages.primary2DAxis, out Vector2 axis);
            sb.Append("{\"ok\":").Append(ok ? "true" : "false").Append(",\"pose\":");
            if (dev.isValid) Pose(sb, p, q); else sb.Append("null");
            sb.Append(",\"trigger\":").Append(trigger.ToString("R", Inv))
              .Append(",\"grip\":").Append(grip.ToString("R", Inv))
              .Append(",\"primary\":").Append(primary ? 1 : 0)
              .Append(",\"secondary\":").Append(secondary ? 1 : 0)
              .Append(",\"menu\":").Append(menu ? 1 : 0)
              .Append(",\"axis\":[").Append(axis.x.ToString("R", Inv)).Append(',').Append(axis.y.ToString("R", Inv)).Append("]}");
        }

        /// Unity pose (left-handed, z forward) -> OpenXR basis x,y,z,qx,qy,qz,qw.
        public static void Pose(StringBuilder sb, Vector3 p, Quaternion q)
        {
            sb.Append('[').Append(p.x.ToString("R", Inv)).Append(',').Append(p.y.ToString("R", Inv)).Append(',')
              .Append((-p.z).ToString("R", Inv)).Append(',').Append((-q.x).ToString("R", Inv)).Append(',')
              .Append((-q.y).ToString("R", Inv)).Append(',').Append(q.z.ToString("R", Inv)).Append(',')
              .Append(q.w.ToString("R", Inv)).Append(']');
        }

        internal static string Escape(string s) =>
            (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", " ").Replace("\r", " ");

        // ---------------------------------------------------------------- tracking space

        Camera _xrCam;
        string _xrCamName = "";
        float _nextCamCheck;

        /// The camera that renders the headset view. Camera.main is not it in this client (it stays at
        /// the origin), so: the stereo camera whose local pose is closest to the tracked head. The room
        /// is drawn under its parent, the rig's tracking space, where the head and controller poses live.
        Camera XrCamera()
        {
            if (_xrCam != null && _xrCam.isActiveAndEnabled && Time.unscaledTime < _nextCamCheck) return _xrCam;
            _nextCamCheck = Time.unscaledTime + 2f;
            var head = InputDevices.GetDeviceAtXRNode(XRNode.CenterEye);
            Vector3 hp = Vector3.zero;
            bool hasHead = head.isValid && head.TryGetFeatureValue(CommonUsages.devicePosition, out hp);
            Camera best = null;
            float bestD = float.MaxValue;
            foreach (var c in Camera.allCameras)
            {
                if (c.stereoTargetEye == StereoTargetEyeMask.None) continue;
                float d = hasHead ? (c.transform.localPosition - hp).sqrMagnitude : 0f;
                if (d < bestD) { best = c; bestD = d; }
            }
            _xrCam = best != null ? best : Camera.main;
            _xrCamName = _xrCam != null ? _xrCam.name : "none";
            return _xrCam;
        }

        /// Head pose in the tracking space: the XR head device, else the XR camera's local pose.
        static bool HeadPose(Camera cam, out Vector3 p, out Quaternion q)
        {
            var head = InputDevices.GetDeviceAtXRNode(XRNode.CenterEye);
            if (head.isValid && head.TryGetFeatureValue(CommonUsages.devicePosition, out p) &&
                head.TryGetFeatureValue(CommonUsages.deviceRotation, out q))
                return true;
            if (cam != null)
            {
                p = cam.transform.localPosition;
                q = cam.transform.localRotation;
                return true;
            }
            p = Vector3.zero;
            q = Quaternion.identity;
            return false;
        }

        /// The capture needs a floor-level origin (eye height is checked at calibration).
        void SetFloorOrigin()
        {
            var subsystems = new List<XRInputSubsystem>();
            SubsystemManager.GetInstances(subsystems);
            foreach (var s in subsystems)
            {
                if ((s.GetSupportedTrackingOriginModes() & TrackingOriginModeFlags.Floor) != 0 &&
                    s.GetTrackingOriginMode() != TrackingOriginModeFlags.Floor)
                    s.TrySetTrackingOriginMode(TrackingOriginModeFlags.Floor);
                _origin = s.GetTrackingOriginMode().ToString().ToLowerInvariant();
            }
            // an XR Origin would put its own mode back: ask it for Floor too (by reflection, no package dependency)
            var t = Type.GetType("Unity.XR.CoreUtils.XROrigin, Unity.XR.CoreUtils");
            if (t == null) return;
            foreach (var o in FindObjectsOfType(t))
            {
                var prop = t.GetProperty("RequestedTrackingOriginMode");
                if (prop != null && prop.PropertyType.IsEnum)
                    prop.SetValue(o, Enum.Parse(prop.PropertyType, "Floor"));
            }
        }

        /// XRoboToolkit's own panels would float in the operator's view during capture.
        void SetUiHidden(bool hide)
        {
            if (hide == _uiHidden) return;
            _uiHidden = hide;
            if (hide)
            {
                _hiddenCanvases.Clear();
                foreach (var c in FindObjectsOfType<Canvas>())
                    if (c.enabled && c.isRootCanvas)
                    {
                        c.enabled = false;
                        _hiddenCanvases.Add(c);
                    }
            }
            else
            {
                foreach (var c in _hiddenCanvases)
                    if (c != null) c.enabled = true;
                _hiddenCanvases.Clear();
            }
        }

        // ---------------------------------------------------------------- host address

        void ReadHostFile()
        {
            try
            {
                var path = Path.Combine(Application.persistentDataPath, "mtc_host.txt");
                if (!File.Exists(path)) return;
                var s = File.ReadAllText(path).Trim();
                if (s.Length == 0) return;
                var parts = s.Split(':');
                _fileHost = parts[0];
                if (parts.Length > 1 && int.TryParse(parts[1], out int port)) _filePort = port;
            }
            catch (Exception e) { Debug.LogWarning("MTC: mtc_host.txt: " + e.Message); }
        }

        void StartBeaconListener()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            try   // some Android builds drop broadcasts without a multicast lock
            {
                using (var up = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                {
                    var activity = up.GetStatic<AndroidJavaObject>("currentActivity");
                    var wifi = activity.Call<AndroidJavaObject>("getSystemService", "wifi");
                    _multicastLock = wifi.Call<AndroidJavaObject>("createMulticastLock", "mtc_capture");
                    _multicastLock.Call("acquire");
                }
            }
            catch (Exception e) { Debug.Log("MTC: no multicast lock (" + e.Message + "); the beacon may not arrive"); }
#endif
            try
            {
                _udp = new UdpClient();
                _udp.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
                _udp.Client.Bind(new IPEndPoint(IPAddress.Any, BeaconPort));
                Task.Run(async () =>
                {
                    while (true)
                    {
                        UdpReceiveResult r;
                        try { r = await _udp.ReceiveAsync(); }
                        catch (Exception) { return; }
                        var msg = Encoding.ASCII.GetString(r.Buffer).Trim().Split(' ');
                        if (msg.Length == 2 && msg[0] == "MTC_CAPTURE" && int.TryParse(msg[1], out int port))
                        {
                            _beaconPort = port;
                            _beaconHost = r.RemoteEndPoint.Address.ToString();
                        }
                    }
                });
            }
            catch (Exception e) { Debug.LogWarning("MTC: beacon listener: " + e.Message); }
        }

        // ---------------------------------------------------------------- WebSocket

        async Task RunAsync(CancellationToken ct)
        {
            while (!ct.IsCancellationRequested)
            {
                string host = _fileHost ?? _beaconHost;
                int port = _fileHost != null ? _filePort : _beaconPort;
                if (host == null)
                {
                    _status = "looking for the capture computer (UDP beacon on port " + BeaconPort + ")\n" +
                              "or put its address in " + Path.Combine(Application.persistentDataPath, "mtc_host.txt");
                    if (!await Pause(500, ct)) return;
                    continue;
                }
                var ws = new ClientWebSocket();
                ws.Options.KeepAliveInterval = TimeSpan.FromSeconds(5);
                try
                {
                    _status = "connecting to " + host + ":" + port;
                    using (var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct))
                    {
                        timeout.CancelAfter(3000);
                        await ws.ConnectAsync(new Uri("ws://" + host + ":" + port + "/mtc"), timeout.Token);
                    }
                    _status = "connected to " + host + ":" + port;
                    _inbox.Enqueue("_connected");
                    _connected = true;
                    var sender = SendLoop(ws, ct);
                    await ReceiveLoop(ws, ct);
                    await sender;
                }
                catch (Exception e)
                {
                    _status = "no connection to " + host + ":" + port + " (" + e.Message + "), retrying";
                }
                finally
                {
                    _connected = false;
                    _inbox.Enqueue("_disconnected");
                    try { ws.Abort(); } catch (Exception) { }
                    ws.Dispose();
                }
                if (!await Pause(1000, ct)) return;
            }
        }

        static async Task<bool> Pause(int ms, CancellationToken ct)
        {
            try { await Task.Delay(ms, ct); return true; }
            catch (OperationCanceledException) { return false; }
        }

        async Task ReceiveLoop(ClientWebSocket ws, CancellationToken ct)
        {
            var buf = new byte[1 << 16];
            var msg = new MemoryStream();
            while (ws.State == WebSocketState.Open && !ct.IsCancellationRequested)
            {
                var r = await ws.ReceiveAsync(new ArraySegment<byte>(buf), ct);
                if (r.MessageType == WebSocketMessageType.Close) break;
                msg.Write(buf, 0, r.Count);
                if (!r.EndOfMessage) continue;
                if (r.MessageType == WebSocketMessageType.Text)
                    _inbox.Enqueue(Encoding.UTF8.GetString(msg.GetBuffer(), 0, (int)msg.Length));
                msg.SetLength(0);
            }
        }

        /// One sender: queued messages first, then only the newest track frame (stale ones are dropped).
        async Task SendLoop(ClientWebSocket ws, CancellationToken ct)
        {
            while (ws.State == WebSocketState.Open && !ct.IsCancellationRequested)
            {
                try { await _wake.WaitAsync(100, ct); }
                catch (OperationCanceledException) { return; }
                string text;
                while (_outbox.TryDequeue(out text))
                    await Send(ws, text, ct);
                text = Interlocked.Exchange(ref _latestTrack, null);
                if (text != null)
                    await Send(ws, text, ct);
                if (_povOut.TryDequeue(out var pov))
                    await ws.SendAsync(new ArraySegment<byte>(pov), WebSocketMessageType.Binary, true, ct);
            }
        }

        static async Task Send(ClientWebSocket ws, string text, CancellationToken ct)
        {
            var bytes = Encoding.UTF8.GetBytes(text);
            await ws.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, ct);
        }
    }
}
