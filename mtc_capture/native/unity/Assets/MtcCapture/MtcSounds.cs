// Sound cues pushed by the capture (the same synth tones as xr_feedback.js in the browser):
// tone(freq Hz, start s, duration s, wave, gain) with a 10 ms attack and exponential decay.

using System;
using System.Collections.Generic;
using UnityEngine;

namespace MtcCapture
{
    public class MtcSounds : MonoBehaviour
    {
        const int Rate = 44100;
        struct Tone
        {
            public float F, At, Dur, Gain;
            public char Wave;   // s(ine) q(square) w(saw) t(riangle)
            public Tone(float f, float at, float dur, char wave = 's', float gain = 0.25f)
            { F = f; At = at; Dur = dur; Wave = wave; Gain = gain; }
        }

        static readonly Dictionary<string, Tone[]> Cues = new Dictionary<string, Tone[]>
        {
            ["calibrated"] = new[] { new Tone(523, 0, .15f), new Tone(659, .12f, .15f), new Tone(784, .24f, .25f) },
            ["refused"] = new[] { new Tone(220, 0, .25f, 'q', .15f), new Tone(185, .28f, .35f, 'q', .15f) },
            ["new_segment"] = new[] { new Tone(660, 0, .12f), new Tone(660, .16f, .12f) },
            ["arming"] = new[] { new Tone(880, 0, .08f, 's', .18f) },
            ["tick"] = new[] { new Tone(880, 0, .08f, 's', .18f) },
            ["start"] = new[] { new Tone(660, 0, .12f), new Tone(990, .12f, .3f) },
            ["touch"] = new[] { new Tone(150, 0, .35f, 'w', .3f) },
            ["warn"] = new[] { new Tone(1200, 0, .04f, 't', .08f) },
            ["hand_lost"] = new[] { new Tone(330, 0, .12f, 'q', .15f), new Tone(330, .2f, .12f, 'q', .15f) },
            ["success"] = new[] { new Tone(523, 0, .12f), new Tone(659, .1f, .12f), new Tone(784, .2f, .12f), new Tone(1047, .3f, .35f) },
            ["unsafe"] = new[] { new Tone(523, 0, .15f), new Tone(392, .16f, .35f, 't') },
            ["abort"] = new[] { new Tone(440, 0, .15f, 'q', .15f), new Tone(330, .16f, .15f, 'q', .15f), new Tone(220, .32f, .3f, 'q', .15f) },
        };

        readonly Dictionary<string, AudioClip> _clips = new Dictionary<string, AudioClip>();
        AudioSource _src;

        void Awake()
        {
            _src = gameObject.AddComponent<AudioSource>();
            _src.spatialBlend = 0f;
            _src.playOnAwake = false;
            foreach (var kv in Cues)
                _clips[kv.Key] = Synth(kv.Key, kv.Value);
        }

        public void Play(string name)
        {
            if (name != null && _clips.TryGetValue(name, out var clip))
                _src.PlayOneShot(clip);
        }

        static AudioClip Synth(string name, Tone[] tones)
        {
            float len = 0f;
            foreach (var t in tones) len = Mathf.Max(len, t.At + t.Dur + 0.02f);
            var data = new float[Mathf.CeilToInt(len * Rate)];
            foreach (var t in tones)
            {
                int i0 = (int)(t.At * Rate), n = (int)(t.Dur * Rate);
                float decay = Mathf.Log(0.0001f / t.Gain);
                for (int i = 0; i < n && i0 + i < data.Length; i++)
                {
                    float s = (float)i / Rate;
                    float env = s < 0.01f ? t.Gain * s / 0.01f : t.Gain * Mathf.Exp(decay * (s - 0.01f) / Mathf.Max(t.Dur - 0.01f, 1e-3f));
                    float ph = (t.F * s) % 1f;
                    float w = t.Wave == 'q' ? (ph < 0.5f ? 1f : -1f) :
                              t.Wave == 'w' ? 2f * ph - 1f :
                              t.Wave == 't' ? 1f - 4f * Mathf.Abs(ph - 0.5f) :
                              Mathf.Sin(2f * Mathf.PI * ph);
                    data[i0 + i] += env * w;
                }
            }
            var clip = AudioClip.Create("mtc_" + name, data.Length, 1, Rate, false);
            clip.SetData(data, 0);
            return clip;
        }
    }
}
