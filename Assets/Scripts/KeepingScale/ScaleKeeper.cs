#nullable enable
using UnityEngine;

namespace KeepingScale
{
    public class ScaleKeeper : MonoBehaviour
    {
        private void Start()
        {
            // localScaleを表すベクトルをワールドの向きに変換
            var worldDirectedLocalScale
                = this.transform.parent.TransformDirection(this.transform.localScale);
            Debug.Log($"worldDirectedLocalScal: {worldDirectedLocalScale}");
            Debug.Log($"cos75: {Mathf.Cos(75 / 360.0f * 2 * Mathf.PI) * Mathf.Sqrt(2) * 0.75} sin75: {Mathf.Sin(75 / 360.0f * 2 * Mathf.PI) * Mathf.Sqrt(2) * 0.75}");

            // ワールド上で上記ベクトルを示す場合のローカル上での表現方法を算出
            // これがlocalScale設定値の大きさをワールド上で実現するために必要なlocalScaleとなる
            var scaleUnaffectedByParents = this.transform.parent.InverseTransformVector(worldDirectedLocalScale);
            this.transform.localScale = scaleUnaffectedByParents;
            Debug.Log($"localScale/lossyScale {this.transform.localScale.x / this.transform.lossyScale.x}");
            Debug.Log($"localScale/ParentLossyScale {this.transform.localScale.x / this.transform.parent.lossyScale.x}");
        }
    }
}
