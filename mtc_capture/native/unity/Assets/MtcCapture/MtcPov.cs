// The operator's point of view as a video stream for analysing takes: a second, off-screen camera
// follows the headset camera and renders the room, the route, the status panel and the tracked
// skeleton (not the passthrough image, which the app never sees) at 640x480, 15 fps. Each frame is
// read back from the GPU asynchronously, JPEG-encoded off the main thread, and sent to the capture
// as a binary message 'P' + headset time (unix ns, the clock of the track frames) + JPEG.

using System;
using System.Threading.Tasks;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;

namespace MtcCapture
{
    public class MtcPov : MonoBehaviour
    {
        public const int Width = 640, Height = 480;
        const float Fps = 15f;
        const float VerticalFov = 75f;   // ~90 deg horizontally at 4:3, close to the headset's view

        MtcLink _link;
        Camera _cam;
        RenderTexture _rt;
        float _next;
        bool _rendering, _inFlight;
        long _stamp;

        public void Init(MtcLink link) => _link = link;

        void Start()
        {
            if (!SystemInfo.supportsAsyncGPUReadback)
            {
                Debug.LogWarning("MTC: no async GPU readback, POV video off");
                enabled = false;
                return;
            }
            _rt = new RenderTexture(Width, Height, 24, RenderTextureFormat.ARGB32);
            _rt.Create();
            var go = new GameObject("MtcPovCamera");
            go.transform.SetParent(transform, false);
            _cam = go.AddComponent<Camera>();
            _cam.stereoTargetEye = StereoTargetEyeMask.None;   // mono, off-screen; MtcLink.XrCamera skips it
            _cam.targetTexture = _rt;
            _cam.fieldOfView = VerticalFov;
            _cam.nearClipPlane = 0.05f;
            _cam.farClipPlane = 50f;
            _cam.clearFlags = CameraClearFlags.SolidColor;
            _cam.backgroundColor = new Color(0.16f, 0.18f, 0.2f);
            _cam.enabled = false;
        }

        void LateUpdate()
        {
            if (_cam == null || _link == null)
                return;
            if (_rendering)   // rendered at the end of the previous frame: read it back
            {
                _rendering = false;
                _cam.enabled = false;
                _inFlight = true;
                long stamp = _stamp;
                AsyncGPUReadback.Request(_rt, 0, TextureFormat.RGBA32, r => OnReadback(r, stamp));
                return;
            }
            var xr = _link.XrCam;
            if (!_link.Connected || xr == null || _inFlight || Time.unscaledTime < _next)
                return;
            _next = Time.unscaledTime + 1f / Fps;
            _cam.transform.SetPositionAndRotation(xr.transform.position, xr.transform.rotation);
            _stamp = MtcLink.NowNs();
            _cam.enabled = true;   // renders once at the end of this frame
            _rendering = true;
        }

        void OnReadback(AsyncGPUReadbackRequest r, long stamp)
        {
            _inFlight = false;
            if (r.hasError || _link == null)
                return;
            byte[] pixels = r.GetData<byte>().ToArray();
            Task.Run(() =>
            {
                try
                {
                    byte[] jpg = ImageConversion.EncodeArrayToJPG(pixels, GraphicsFormat.R8G8B8A8_SRGB,
                        (uint)Width, (uint)Height, 0, 70);
                    _link.SendPov(stamp, jpg);
                }
                catch (Exception e)
                {
                    Debug.LogWarning("MTC: POV encode " + e.Message);
                }
            });
        }

        void OnDestroy()
        {
            if (_rt != null) _rt.Release();
        }
    }
}
