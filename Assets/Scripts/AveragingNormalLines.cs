using System.Linq;
using UniRx;
using UniRx.Triggers;
using UnityEngine;

namespace Scripts
{
    /// <summary>
    /// Meshの法線を重なる頂点の平均の法線に書き替えるコンポーネント
    /// </summary>
    public class AveragingNormalLines : MonoBehaviour
    {
        private void Awake()
        {
            this.AverageNormals(this.GetComponent<MeshFilter>());
            //this.CopyMesh(this.GetComponent<MeshFilter>());
        }

        private void AverageNormals(MeshFilter meshFilter)
        {
            var baseMesh = meshFilter.mesh;

            var normals = baseMesh.normals;
            baseMesh.vertices
                .Select((x, idx) => (Pos: x, Index: idx, Normal: baseMesh.normals[idx]))
                .GroupBy(x => x.Pos)
                .ToList()
                .ForEach(x =>
                {
                    var avg = x.Aggregate(Vector3.zero, (sum, next) => sum + next.Normal).normalized;
                    x.ToList().ForEach(y => normals[y.Index] = avg);
                });

            var newMesh = new Mesh();
            newMesh.SetVertices(baseMesh.vertices);
            newMesh.SetNormals(normals);
            newMesh.SetUVs(0, baseMesh.uv);
            newMesh.SetIndices(baseMesh.GetIndices(0), MeshTopology.Triangles, 0);
            meshFilter.mesh = newMesh;

            this.OnDestroyAsObservable().Subscribe(_ => Destroy(newMesh));
        }

        private void CopyMesh(MeshFilter meshFilter)
        { 
            var baseMesh = meshFilter.mesh;
            var newMesh = new Mesh();
            newMesh.SetVertices(baseMesh.vertices);
            newMesh.SetNormals(baseMesh.normals);
            newMesh.SetUVs(0, baseMesh.uv);
            newMesh.SetIndices(baseMesh.GetIndices(0), MeshTopology.Triangles, 0);
            meshFilter.mesh = newMesh;

            this.OnDestroyAsObservable().Subscribe(_ => Destroy(newMesh));
        }

    }
}
