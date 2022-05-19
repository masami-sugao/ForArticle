#nullable enable
using UnityEngine;

namespace KeepingScale
{
    public class ScaleKeeper : MonoBehaviour
    {
        private void Start()
        {
            Debug.Log($"localScale/lossyScale {this.transform.localScale.x / this.transform.lossyScale.x}");
            Debug.Log($"localScale/ParentLossyScale {this.transform.localScale.x / this.transform.parent.lossyScale.x}");

            this.DoByMatrix();
            // this.DoByTransformMethods();
        }
        
        private void DoByTransformMethods()
        {
            // localScaleを表すベクトルをワールドの向きに変換
            var worldDirectedLocalScale
                = this.transform.parent.TransformDirection(this.transform.localScale);
            Debug.Log($"worldDirectedLocalScal: {worldDirectedLocalScale}");

            // ワールド上で上記ベクトルを示す場合のローカル上での表現方法を算出
            // これがlocalScale設定値の大きさをワールド上で実現するために必要なlocalScaleとなる
            var scaleUnaffectedByParents = this.transform.parent.InverseTransformVector(worldDirectedLocalScale);
            this.transform.localScale = scaleUnaffectedByParents;
        }
        
        private void DoByMatrix()
        {
            var parentScale = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, this.transform.parent.lossyScale);
            this.transform.localScale = parentScale.inverse * this.transform.lossyScale;
        }
    }
}
