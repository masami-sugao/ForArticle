using UnityEditor;
using UnityEngine;

/// <summary>
/// UnpackしたGameObjectと同じPosition,Rotation,Scaleのものを再現したプレハブをヒエラルキーに配置する拡張機能
/// </summary>
/// <remarks>対象GameObjectを内包するプレハブをResources/<see cref="FolderName"/>フォルダに移動してから実行する必要があります。</remarks>
public static class UnpackedGameObjectRecreation
{
    private const string FolderName = "RecreationTarget";

    [MenuItem("GameObject/UnpackしたGameObjectをプレハブで再作成", false, 1)]
    public static void RecreateUnpackedGameObjectUsingPrefab()
    {
        var prefab = Resources.LoadAll<GameObject>(FolderName)[0];
        if (prefab == null)
        {
            Debug.LogError($"再現するプレハブをResources/{FolderName}フォルダに移動してから実行して下さい。");
            return;
        }

        foreach (var srcModel in Selection.transforms)
        {
            var createdRoot = (PrefabUtility.InstantiatePrefab(prefab) as GameObject).transform;
            var createdModel = createdRoot.GetChild(0);

            // 選択中のGameObjectと同じPosition, Rotation, Scaleとなるように、プレハブのルートGameObjectのTransformを変更
            var couldAdjust = AdjustScale(srcModel, createdModel);
            if (!couldAdjust)
            {
                UnityEngine.GameObject.DestroyImmediate(createdRoot.gameObject);
                continue;
            }

            AdjustRotation(srcModel, createdModel);
            AdjustPosition(srcModel, createdModel);
        }
    }

    private static bool AdjustScale(Transform src, Transform dest)
    {
        if (src.lossyScale == dest.lossyScale) return true;

        var tolerance = 0.0001f;
        var multiplier = src.lossyScale.x / dest.localScale.x;
        if (Mathf.Abs(multiplier - src.lossyScale.y / dest.localScale.y) > tolerance || Mathf.Abs(multiplier - src.lossyScale.z / dest.localScale.z) > tolerance)
        {
            Debug.LogError("選択したGameObjectとプレハブの子階層のGameObjectでScaleのXYZの比率が異なるため、再現できません。");
            return false;
        }

        dest.parent.localScale = Vector3.one * multiplier;
        return true;
    }

    private static void AdjustRotation(Transform src, Transform dest)
    {
        if (src.rotation == dest.rotation) return;

        dest.parent.rotation = src.rotation * Quaternion.Inverse(dest.localRotation);
    }

    private static void AdjustPosition(Transform src, Transform dest)
    {
        if (src.position == dest.position) return;

        dest.parent.position += src.position - dest.position;
    }
}
