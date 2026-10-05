// Third-person view for the operator: a camera 1.8 m behind and 0.6 m above the head (turning with
// the gaze) renders the room and the self view the capture draws (skeleton / robot collision shapes)
// at 480x360, 20 fps, onto a small panel at the upper left of the view. Toggled by the capture
// (cmd third_person, from the left controller's Y button or the t key).

using UnityEngine;

namespace MtcCapture
{
    public class MtcThirdPerson : MonoBehaviour
    {
        const int Width = 480, Height = 360;
        const float Fps = 20f;
        const int PanelLayer = 30;            // the panel itself; the third-person camera never sees it
        const float Back = 1.8f, Up = 0.6f, LookDown = 0.5f;
        const float PanelDistance = 1.4f, PanelLeft = 0.55f, PanelUp = 0.12f, PanelHeight = 0.36f;

        MtcLink _link;
        Camera _cam;
        RenderTexture _rt;
        GameObject _panel;
        bool _on;
        float _next;
        Vector3 _pos;
        float _yaw;
        bool _placed;

        public void Init(MtcLink link) => _link = link;

        public void SetOn(bool on)
        {
            _on = on;
            if (_panel != null) _panel.SetActive(on);
            if (!on && _cam != null) _cam.enabled = false;
        }

        void Start()
        {
            _rt = new RenderTexture(Width, Height, 24, RenderTextureFormat.ARGB32);
            _rt.Create();
            var go = new GameObject("MtcThirdPersonCamera");
            go.transform.SetParent(transform, false);
            _cam = go.AddComponent<Camera>();
            _cam.stereoTargetEye = StereoTargetEyeMask.None;   // mono, off-screen; MtcLink.XrCamera skips it
            _cam.targetTexture = _rt;
            _cam.fieldOfView = 60f;
            _cam.nearClipPlane = 0.05f;
            _cam.farClipPlane = 50f;
            _cam.clearFlags = CameraClearFlags.SolidColor;
            _cam.backgroundColor = new Color(0.16f, 0.18f, 0.2f);
            _cam.cullingMask &= ~(1 << PanelLayer);
            _cam.enabled = false;

            _panel = GameObject.CreatePrimitive(PrimitiveType.Quad);
            _panel.name = "MtcThirdPersonPanel";
            _panel.layer = PanelLayer;
            Destroy(_panel.GetComponent<Collider>());
            _panel.transform.SetParent(transform, false);
            _panel.transform.localScale = new Vector3(PanelHeight * Width / Height, PanelHeight, 1f);
            var mat = new Material(Resources.Load<Shader>("MtcSimple"));
            mat.SetTexture("_MainTex", _rt);
            mat.SetColor("_Color", Color.white);
            mat.SetFloat("_Lit", 0f);
            mat.SetFloat("_ZWrite", 0f);
            mat.SetFloat("_ZTest", (float)UnityEngine.Rendering.CompareFunction.Always);
            mat.renderQueue = 4001;
            _panel.GetComponent<Renderer>().sharedMaterial = mat;
            _panel.SetActive(_on);
        }

        void LateUpdate()
        {
            if (_cam == null) return;
            var xr = _link != null ? _link.XrCam : null;
            if (!_on || xr == null || !_link.Connected)
            {
                if (_panel.activeSelf) _panel.SetActive(false);
                _cam.enabled = false;
                return;
            }
            if (!_panel.activeSelf) _panel.SetActive(true);
            var head = xr.transform.position;
            var fwd = xr.transform.forward;
            fwd.y = 0f;
            if (fwd.sqrMagnitude < 1e-6f) fwd = Vector3.forward;
            fwd.Normalize();
            float yaw = Mathf.Atan2(fwd.x, fwd.z) * Mathf.Rad2Deg;

            // the camera behind and above, looking at the body
            _cam.transform.position = head - fwd * Back + Vector3.up * Up;
            _cam.transform.LookAt(head + Vector3.down * LookDown);
            _cam.enabled = Time.unscaledTime >= _next;   // ~20 renders per second
            if (_cam.enabled) _next = Time.unscaledTime + 1f / Fps;

            // the panel: upper left of the view, lazily following the gaze heading
            var rot = Quaternion.Euler(0f, yaw, 0f);
            var target = head + rot * new Vector3(-PanelLeft, PanelUp, PanelDistance);
            float k = 1f - Mathf.Exp(-Time.unscaledDeltaTime / 0.25f);
            if (!_placed) { _pos = target; _yaw = yaw; _placed = true; }
            _pos = Vector3.Lerp(_pos, target, k);
            _yaw += k * Mathf.DeltaAngle(_yaw, yaw);
            _panel.transform.SetPositionAndRotation(_pos, Quaternion.Euler(0f, _yaw, 0f));
        }

        void OnDestroy()
        {
            if (_rt != null) _rt.Release();
        }
    }
}
