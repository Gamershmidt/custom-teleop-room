// The capture's status panel (a PNG rendered by hud.py), placed like vr_app._place_hud: ahead of the
// operator at layout.distance, layout.below under eye level, lazily following the gaze heading.
// Before the capture connects, a text panel in the same place shows the connection status.

using LitJson;
using TMPro;
using UnityEngine;

namespace MtcCapture
{
    public class MtcHud
    {
        readonly Transform _root;
        readonly GameObject _panel;
        readonly Material _mat;
        readonly TextMeshPro _text;
        Texture2D _tex;
        float _distance = 1.4f, _height = 0.17f, _aspect = 4f, _below = 0.34f;
        Vector3 _pos;
        float _yaw;
        bool _placed, _hasPanel;

        public MtcHud(Transform owner)
        {
            _root = new GameObject("MtcHud").transform;
            _root.SetParent(owner, false);
            _panel = GameObject.CreatePrimitive(PrimitiveType.Quad);
            _panel.name = "MtcHudPanel";
            Object.Destroy(_panel.GetComponent<Collider>());
            _panel.transform.SetParent(_root, false);
            _mat = new Material(Resources.Load<Shader>("MtcSimple"));
            _mat.SetFloat("_Lit", 0f);
            _mat.SetFloat("_ZWrite", 0f);
            _mat.SetFloat("_ZTest", (float)UnityEngine.Rendering.CompareFunction.Always);   // never hidden by furniture
            _mat.renderQueue = 4000;
            _panel.GetComponent<Renderer>().sharedMaterial = _mat;
            _panel.SetActive(false);

            var textGo = new GameObject("MtcStatusText");
            textGo.transform.SetParent(_root, false);
            _text = textGo.AddComponent<TextMeshPro>();
            _text.rectTransform.sizeDelta = new Vector2(1.0f, 0.3f);
            _text.enableAutoSizing = true;
            _text.fontSizeMin = 0.05f;
            _text.fontSizeMax = 0.6f;
            _text.alignment = TextAlignmentOptions.Center;
            _text.color = Color.white;
        }

        public void SetParent(Transform space)
        {
            if (_root.parent == space) return;
            _root.SetParent(space, false);
            _root.localPosition = Vector3.zero;
            _root.localRotation = Quaternion.identity;
        }

        public void SetPanel(byte[] png, JsonData layout)
        {
            if (_tex == null) _tex = new Texture2D(2, 2, TextureFormat.RGBA32, false);
            _tex.LoadImage(png);
            _mat.SetTexture("_MainTex", _tex);
            _mat.SetColor("_Color", Color.white);
            if (layout != null && layout.IsObject)
            {
                if (layout.ContainsKey("distance")) _distance = (float)MtcScene.Num(layout["distance"]);
                if (layout.ContainsKey("height")) _height = (float)MtcScene.Num(layout["height"]);
                if (layout.ContainsKey("aspect")) _aspect = (float)MtcScene.Num(layout["aspect"]);
                if (layout.ContainsKey("below")) _below = (float)MtcScene.Num(layout["below"]);
            }
            _panel.transform.localScale = new Vector3(_height * _aspect, _height, 1f);
            _hasPanel = true;
        }

        public void ClearPanel() => _hasPanel = false;

        /// Status text while the capture is not connected (null = connected: show the capture's panel).
        public void SetStatus(string text)
        {
            bool showText = text != null || !_hasPanel;
            _text.gameObject.SetActive(showText);
            if (showText) _text.text = text ?? "MTC capture: connected, waiting for the status panel";
            _panel.SetActive(!showText);
        }

        /// head: camera pose in the tracking space (Unity basis).
        public void Place(Vector3 head, Quaternion rot, float dt)
        {
            var fwd = rot * Vector3.forward;
            fwd.y = 0f;
            if (fwd.sqrMagnitude < 1e-6f) fwd = Vector3.forward;
            fwd.Normalize();
            var target = head + _distance * fwd + Vector3.down * _below;
            float yaw = Mathf.Atan2(fwd.x, fwd.z) * Mathf.Rad2Deg;
            if (!_placed)
            {
                _pos = target;
                _yaw = yaw;
                _placed = true;
            }
            float k = 1f - Mathf.Exp(-dt / 0.25f);   // ~0.25 s lag, like the browser panel
            _pos = Vector3.Lerp(_pos, target, k);
            _yaw += k * Mathf.DeltaAngle(_yaw, yaw);
            var r = Quaternion.Euler(0f, _yaw, 0f);   // the quad's visible face (-z) towards the operator
            _panel.transform.localPosition = _pos;
            _panel.transform.localRotation = r;
            _text.transform.localPosition = _pos + Vector3.up * 0.1f;
            _text.transform.localRotation = r;
        }
    }
}
