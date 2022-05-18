#nullable enable
using System;
using Cysharp.Threading.Tasks;
using Cysharp.Threading.Tasks.Linq;
using UnityEngine;
using UnityEngine.UI;

namespace TimerExperiment
{
    [RequireComponent(typeof(Text))]
    public class ElapsedTime : MonoBehaviour
    {
        private enum TimerPattern {
            Double,
            Float,
            DeltaTimeTotal,
            DeltaTime
        }

        [SerializeField]
        private TimerPattern pattern;

        private Text text = null!;

        private void Start()
        {
            this.text = this.GetComponent<Text>();
            UniTaskAsyncEnumerable.EveryUpdate()
                .Subscribe(_ =>
                    this.text.text = (this.pattern switch
                    {
                        TimerPattern.Double => Time.timeSinceLevelLoadAsDouble,
                        TimerPattern.Float => Time.timeSinceLevelLoad,
                        TimerPattern.DeltaTimeTotal => Rotator.ElapsedTime,
                        TimerPattern.DeltaTime => Time.deltaTime,
                        _ => throw new NotImplementedException()
                    }).ToString("G9"),
                    this.GetCancellationTokenOnDestroy());
        }
    }
}
