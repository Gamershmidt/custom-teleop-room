// PICO body tracking: 24 joints (BodyTrackerRole, SMPL order) fused from the headset, both
// controllers and two Motion Trackers on the lower legs. Started here once the trackers are
// calibrated (the PICO Motion Tracker app, or the capture's `b` key -> body_calibrate).
//
// The SDK's localPose is in Unity's left-handed basis, like the camera and controllers (measured on a
// PICO 4 Ultra: z mirrored against the controllers), so it gets the same conversion (MtcLink.Pose).
// tracker_trial --native checks it against the head pose.

using System;
using System.Globalization;
using System.Text;
using Unity.XR.PXR;
using UnityEngine;

namespace MtcCapture
{
    public class MtcBody
    {
        public string Text { get; private set; } = "body tracking: not started";

        bool _started, _calibrated, _supported, _tracking, _bodyMode;
        int _trackers = -1, _startResult = -1;
        float _nextModeRequest;
        string _stateText = "";
        float _nextCheck, _lostSince = -1f;
        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        public void Tick()
        {
            if (Time.unscaledTime < _nextCheck) return;
            _nextCheck = Time.unscaledTime + 1f;
#if UNITY_ANDROID && !UNITY_EDITOR
            try
            {
                PXR_MotionTracking.GetBodyTrackingSupported(ref _supported);
                // body-tracking mode first (as XRoboToolkit's UI does): in object-tracking mode the
                // trackers never report a body calibration
                _bodyMode = PXR_MotionTracking.GetMotionTrackerMode() == MotionTrackerMode.BodyTracking;
                if (!_bodyMode && Time.unscaledTime > _nextModeRequest)
                {
                    _nextModeRequest = Time.unscaledTime + 5f;
                    PXR_MotionTracking.CheckMotionTrackerModeAndNumber(MotionTrackerMode.BodyTracking, MotionTrackerNum.TWO);
                }
                var conn = new MotionTrackerConnectState();
                if (PXR_MotionTracking.GetMotionTrackerConnectStateWithSN(ref conn) == 0) _trackers = conn.trackerSum;
                int calib = 0;
                PXR_Input.GetMotionTrackerCalibState(ref calib);
                _calibrated = calib == 1;
                if (_calibrated && !_started)
                {
                    _startResult = PXR_MotionTracking.StartBodyTracking(BodyTrackingMode.BTM_FULL_BODY_HIGH,
                        new BodyTrackingBoneLength());
                    _started = _startResult == 0;
                }
                bool tracking = false;
                var st = new BodyTrackingState();
                PXR_MotionTracking.GetBodyTrackingState(ref tracking, ref st);
                _tracking = tracking && st.stateCode != BodyTrackingStatusCode.BT_INVALID;
                _stateText = _tracking ? (st.stateCode == BodyTrackingStatusCode.BT_LIMITED ? "limited" : "ok")
                                       : ErrorText(st.errorCode);
                // tracking lost for a while (strap moved, user changed): start again
                if (_started && !_tracking)
                {
                    if (_lostSince < 0f) _lostSince = Time.unscaledTime;
                    else if (Time.unscaledTime - _lostSince > 5f) { _started = false; _lostSince = -1f; }
                }
                else _lostSince = -1f;
            }
            catch (Exception e)
            {
                _stateText = "error: " + e.Message;
            }
#else
            _stateText = "editor: no PICO body tracking";
#endif
            Text = !_supported ? "body tracking: not supported on this device" :
                   _trackers == 0 ? "body tracking: no Motion Trackers connected (pair them in the PICO settings)" :
                   !_bodyMode ? "body tracking: switching the trackers to body mode" :
                   !_calibrated ? "body tracking: Motion Trackers not calibrated (key b on the laptop)" :
                   "body tracking: " + _stateText;
        }

        public void Calibrate()
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            try { PXR_MotionTracking.StartMotionTrackerCalibApp(); } catch (Exception e) { Debug.LogWarning("MTC: " + e.Message); }
#endif
            _started = false;
        }

        public void Restart() => _started = false;

        /// Appends {"state": {...}, "joints": [[x,y,z,qx,qy,qz,qw] x 24] or null}.
        public void Append(StringBuilder sb)
        {
            sb.Append("{\"state\":{\"supported\":").Append(_supported ? "true" : "false")
              .Append(",\"calibrated\":").Append(_calibrated ? "true" : "false")
              .Append(",\"started\":").Append(_started ? "true" : "false")
              .Append(",\"tracking\":").Append(_tracking ? "true" : "false")
              .Append(",\"body_mode\":").Append(_bodyMode ? "true" : "false")
              .Append(",\"trackers\":").Append(_trackers)
              .Append(",\"start_result\":").Append(_startResult)
              .Append(",\"text\":\"").Append(MtcLink.Escape(_stateText)).Append("\"},\"joints\":");
            if (!AppendJoints(sb)) sb.Append("null");
            sb.Append('}');
        }

        bool AppendJoints(StringBuilder sb)
        {
#if UNITY_ANDROID && !UNITY_EDITOR
            if (!_started || !_tracking) return false;
            try
            {
                var info = new BodyTrackingGetDataInfo();
                var data = new BodyTrackingData();
                if (PXR_MotionTracking.GetBodyTrackingData(ref info, ref data) != 0 || data.roleDatas == null)
                    return false;
                int n = (int)BodyTrackerRole.ROLE_NUM;
                if (data.roleDatas.Length < n) return false;
                sb.Append('[');
                for (int i = 0; i < n; i++)
                {
                    var p = data.roleDatas[i].localPose;
                    if (i > 0) sb.Append(',');
                    // localPose is in Unity's left-handed basis (measured: mirrored in z against the
                    // controllers, left hip on the right), so it gets the same conversion as every pose
                    MtcLink.Pose(sb, new Vector3((float)p.PosX, (float)p.PosY, (float)p.PosZ),
                                 new Quaternion((float)p.RotQx, (float)p.RotQy, (float)p.RotQz, (float)p.RotQw));
                }
                sb.Append(']');
                return true;
            }
            catch (Exception) { return false; }
#else
            return false;
#endif
        }

        static string ErrorText(BodyTrackingErrorCode e)
        {
            switch (e)
            {
                case BodyTrackingErrorCode.BT_ERROR_TRACKER_NOT_CALIBRATED: return "trackers not calibrated";
                case BodyTrackingErrorCode.BT_ERROR_TRACKER_NUM_NOT_ENOUGH: return "fewer than 2 trackers connected";
                case BodyTrackingErrorCode.BT_ERROR_TRACKER_STATE_NOT_SATISFIED: return "trackers not ready";
                case BodyTrackingErrorCode.BT_ERROR_TRACKER_PERSISTENT_INVISIBILITY: return "trackers not seen by the headset";
                case BodyTrackingErrorCode.BT_ERROR_TRACKER_DATA_ERROR: return "tracker data error";
                case BodyTrackingErrorCode.BT_ERROR_USER_CHANGE: return "user changed: recalibrate";
                case BodyTrackingErrorCode.BT_ERROR_TRACKING_POSE_ERROR: return "pose error";
                default: return "not tracking";
            }
        }
    }
}
