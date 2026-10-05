// The room and markers sent by the capture. Each element arrives as a row-major 4x4 matrix in the
// OpenXR basis (the capture has already applied its anchor), plus shape, dims, colour and opacity.
// OpenXR -> Unity is the conjugation M_u = S M S with S = diag(1, 1, -1).

using System.Collections.Generic;
using LitJson;
using UnityEngine;

namespace MtcCapture
{
    public class MtcScene
    {
        class Item
        {
            public GameObject Go;
            public Material Mat;
            public string Shape, Color;
            public float Opacity = -1f;
            public bool Emissive;
        }

        readonly Transform _root;
        readonly Shader _shader;
        readonly Dictionary<string, Dictionary<string, Item>> _groups = new Dictionary<string, Dictionary<string, Item>>();

        public MtcScene(Transform owner)
        {
            _root = new GameObject("MtcRoom").transform;
            _root.SetParent(owner, false);
            _shader = Resources.Load<Shader>("MtcSimple");
        }

        public void SetParent(Transform space)
        {
            if (_root.parent == space) return;
            _root.SetParent(space, false);
            _root.localPosition = Vector3.zero;
            _root.localRotation = Quaternion.identity;
            _root.localScale = Vector3.one;
        }

        public void Clear()
        {
            foreach (var g in _groups.Values)
                foreach (var it in g.Values)
                    Destroy(it);
            _groups.Clear();
        }

        /// Replaces a group ("scene" or "xr") with the given elements, reusing objects by key.
        public void Apply(string group, JsonData els)
        {
            if (!_groups.TryGetValue(group, out var items))
                _groups[group] = items = new Dictionary<string, Item>();
            var seen = new HashSet<string>();
            if (els != null && els.IsArray)
            {
                for (int i = 0; i < els.Count; i++)
                {
                    var e = els[i];
                    var key = (string)e["k"];
                    seen.Add(key);
                    var shape = (string)e["s"];
                    if (items.TryGetValue(key, out var it) && it.Shape != shape)
                    {
                        Destroy(it);
                        it = null;
                    }
                    if (it == null)
                        items[key] = it = Create(group + "/" + key, shape);
                    Place(it, e["m"], e["d"]);
                    Style(it, (string)e["c"], (float)Num(e["o"]), (bool)e["e"]);
                }
            }
            var gone = new List<string>();
            foreach (var k in items.Keys)
                if (!seen.Contains(k)) gone.Add(k);
            foreach (var k in gone)
            {
                Destroy(items[k]);
                items.Remove(k);
            }
        }

        Item Create(string name, string shape)
        {
            var type = shape == "cylinder" ? PrimitiveType.Cylinder :
                       shape == "sphere" ? PrimitiveType.Sphere :
                       shape == "plane" ? PrimitiveType.Quad : PrimitiveType.Cube;
            var go = GameObject.CreatePrimitive(type);
            go.name = name;
            Object.Destroy(go.GetComponent<Collider>());
            go.transform.SetParent(_root, false);
            var mat = new Material(_shader);
            var r = go.GetComponent<Renderer>();
            r.sharedMaterial = mat;
            r.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            r.receiveShadows = false;
            return new Item { Go = go, Mat = mat, Shape = shape };
        }

        static void Place(Item it, JsonData m, JsonData d)
        {
            // M_u[i][j] = s_i s_j M[i][j], s = (1, 1, -1)
            var a = new float[16];
            for (int i = 0; i < 16; i++) a[i] = (float)Num(m[i]);
            float S(int r) => r == 2 ? -1f : 1f;
            float U(int r, int c) => S(r) * S(c) * a[4 * r + c];
            var c0 = new Vector3(U(0, 0), U(1, 0), U(2, 0));
            var c1 = new Vector3(U(0, 1), U(1, 1), U(2, 1));
            var c2 = new Vector3(U(0, 2), U(1, 2), U(2, 2));
            var t = it.Go.transform;
            t.localPosition = new Vector3(U(0, 3), U(1, 3), U(2, 3));
            if (c2.sqrMagnitude > 1e-12f && c1.sqrMagnitude > 1e-12f)
                t.localRotation = Quaternion.LookRotation(c2.normalized, c1.normalized);
            // primitive geometry -> element size (three.js: box w,h,d; cylinder r,h along y; sphere r; plane w,h)
            Vector3 g;
            switch (it.Shape)
            {
                case "cylinder": g = new Vector3(2f * F(d, 0), F(d, 1) / 2f, 2f * F(d, 0)); break;   // Unity: radius 0.5, height 2
                case "sphere": g = Vector3.one * 2f * F(d, 0); break;                               // Unity: diameter 1
                case "plane": g = new Vector3(F(d, 0), F(d, 1), 1f); break;
                default: g = new Vector3(F(d, 0), F(d, 1), F(d, 2)); break;
            }
            t.localScale = new Vector3(c0.magnitude * g.x, c1.magnitude * g.y, c2.magnitude * g.z);
        }

        static void Style(Item it, string color, float opacity, bool emissive)
        {
            if (it.Color == color && Mathf.Approximately(it.Opacity, opacity) && it.Emissive == emissive) return;
            it.Color = color;
            it.Opacity = opacity;
            it.Emissive = emissive;
            ColorUtility.TryParseHtmlString(color, out var c);
            c.a = opacity;
            var m = it.Mat;
            m.SetColor("_Color", c);
            m.SetFloat("_Lit", 1f);
            m.SetFloat("_Emit", emissive ? 0.6f : 0f);
            bool transparent = opacity < 0.999f;
            m.SetFloat("_ZWrite", transparent ? 0f : 1f);
            m.renderQueue = transparent ? 3000 : 2000;
        }

        static void Destroy(Item it)
        {
            if (it == null) return;
            Object.Destroy(it.Go);
            Object.Destroy(it.Mat);
        }

        static float F(JsonData d, int i) => d != null && d.IsArray && i < d.Count ? (float)Num(d[i]) : 1f;

        public static double Num(JsonData j)
        {
            if (j == null) return 0;
            if (j.IsDouble) return (double)j;
            if (j.IsInt) return (int)j;
            if (j.IsLong) return (long)j;
            return 0;
        }
    }
}
