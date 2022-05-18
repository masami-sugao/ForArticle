#nullable enable
using UniRx;
using UniRx.Triggers;
using UnityEngine;

namespace TimerExperiment
{
    public class Rotator : MonoBehaviour
    {
        [SerializeField]
        private bool usesDeltaTime;

        private Quaternion initialRotation;

        public static float ElapsedTime { get; private set; }

        private void Start()
        {
            this.initialRotation = this.transform.rotation;

            if (this.usesDeltaTime)
                this.RotateByDeltaTime();
            else
                this.RotateByTimeSinceLevelLoad();
        }

        private void RotateByDeltaTime()
        {
            this.UpdateAsObservable().Subscribe(_ =>
            {
                ElapsedTime += Time.deltaTime;
                this.Rotate(ElapsedTime);
            });
        }

        private void RotateByTimeSinceLevelLoad()
        {
            this.UpdateAsObservable().Subscribe(_ => this.Rotate(Time.timeSinceLevelLoadAsDouble));
        }

        private void Rotate(double elapsedTime)
        {
            this.transform.rotation = this.initialRotation * Quaternion.Euler(0.0f, (float)(elapsedTime % 1.0) * -360.0f, 0.0f);
        }
    }
}
