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

        private static float elapsedTime;

        public static float ElapsedTime => elapsedTime;

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
                elapsedTime += Time.deltaTime;
                this.Rotate(elapsedTime);
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
