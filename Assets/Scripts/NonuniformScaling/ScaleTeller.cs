#nullable enable
using UnityEngine;

namespace NonuniformScaling
{
    public class ScaleTeller : MonoBehaviour
    {
        private void Start()
        {
            Debug.Log($"[{this.name}] LossyScale: {this.transform.lossyScale} LocalScale: {this.transform.localScale}");
        }
    }
}
