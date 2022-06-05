Shader "Custom/Crack_Tessellation" {
    Properties {
        [Header(Albedo)]
        [MainColor] _BaseColor("Base Color", Color) = (1.0, 1.0, 1.0, 1.0)
        [MainTexture] _BaseMap("Base Map", 2D) = "white" {}

        [Header(Metallic and Smoothness)]
        _Smoothness("Smoothness", Range(0.0, 1.0)) = 0.0
        _Metallic("Metallic", Range(0.0, 1.0)) = 0.0

        [Header(Crack)]
        _CrackProgress("クラック進行具合", Range(0.0, 1.0)) = 0.0
        [HDR] _CrackColor("クラック色", Color) = (0.0, 0.0, 0.0, 1.0)
        _CrackDetailedness("クラック模様の細かさ", Range(0.0, 8.0)) = 3.0
        _CrackDepth("クラックの深さ", Range(0.0, 1.0)) = 0.5
        _AdditionalCrackDepthForLighting("ライティング計算時に実際の値に追加するクラック深さ ", Float) = 1.0
        _CrackWidth("クラックの幅", Range(0.01, 0.1)) = 0.05
        _CrackWallWidth("クラックの壁部分の幅", Range(0.001, 0.2)) = 0.08
        // フラグメントシェーダーでクラック対象かどうかの再計算を行うかどうか
        [Toggle] _DrawsCrackWithPixelUnit("ピクセル単位でクラック模様の再計算を行うか", Int) = 0

        [Space]
        _RandomSeed("クラック模様のランダムシード(非負整数のみ可)", Int) = 0

        [Header(SubdividingPolygon)]
        _SubdividingCount("細分化時に辺をいくつに分割するか(1以下は分割無し)", Int) = 1
        _SubdividingInsideScaleFactor("細分化時のポリゴン内部への新ポリゴン生成度合い", Range(0.0, 1.0)) = 1.0
    }

    SubShader {
        Tags {
            "RenderType" = "Opaque"
            "RenderPipeline" = "UniversalPipeline"
            "UniversalMaterialType" = "Lit"
        }
        LOD 300

        HLSLINCLUDE
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        ENDHLSL

        Pass {
            Name "Crack"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM

            // -------------------------------------
            // Material Keywords
            #pragma shader_feature_local_fragment _ALPHATEST_ON
            #pragma shader_feature_local_fragment _ALPHAPREMULTIPLY_ON
            #pragma shader_feature_local_fragment _SPECULARHIGHLIGHTS_OFF
            #pragma shader_feature_local_fragment _ENVIRONMENTREFLECTIONS_OFF
            #pragma shader_feature_local_fragment _SPECULAR_SETUP
            #pragma shader_feature_local _RECEIVE_SHADOWS_OFF

            // -------------------------------------
            // Universal Pipeline keywords
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS_CASCADE
            #pragma multi_compile _ _ADDITIONAL_LIGHTS_VERTEX _ADDITIONAL_LIGHTS
            #pragma multi_compile_fragment _ _ADDITIONAL_LIGHT_SHADOWS
            #pragma multi_compile_fragment _ _SHADOWS_SOFT

            //--------------------------------------
            // GPU Instancing
            #pragma multi_compile_instancing

            // -------------------------------------
            // Local Keywords
            #pragma shader_feature_local _ _WIREFRAME_ON


            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitForwardPass.hlsl"

            #pragma vertex Vert
            #pragma hull Hull
            #pragma domain Domain
            #pragma fragment Frag

            #pragma require tessellation tessHW


            // ---------------------------------------------------------------------------------------
            // 変数宣言
            // ---------------------------------------------------------------------------------------
            float _CrackProgress;
            half4 _CrackColor;
            float _CrackDetailedness;
            float _CrackDepth;
            float _AdditionalCrackDepthForLighting;
            float _CrackWidth;
            float _CrackWallWidth;
            bool _DrawsCrackWithPixelUnit;
            uint _RandomSeed;

            int  _SubdividingCount;
            float _SubdividingInsideScaleFactor;


            // ---------------------------------------------------------------------------------------
            // 構造体
            // ---------------------------------------------------------------------------------------
            struct v2d {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
                float3 normalOS : NORMAL;
                float3 normalWS : TEXCOORD1;
            };

            struct patchConstParam {
                float edgeTessFactors[3] : SV_TessFactor;
                float insideTessFactor : SV_InsideTessFactor;
            };

            struct d2f {
                float4 positionCS : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 initNormalOS : NORMAL;
                float3 initNormalWS : TEXCOORD1;
                float3 positionOS : TEXCOORD2;
                float3 initPositionOS : TEXCOORD3;
                float crackLevel : TEXCOORD4;
            };


            // ---------------------------------------------------------------------------------------
            // メソッド
            // ---------------------------------------------------------------------------------------
            /**
             * Xorshift32を用いて32bitの擬似乱数を生成する
             */
            uint Xorshift32(uint value) {
                value = value ^ (value << 13);
                value = value ^ (value >> 17);
                value = value ^ (value << 5);
                return value;
            }

            /**
             * 整数の値を1未満の小数にマッピングする
             */
            float MapToFloat(uint value) {
                const float precion = 100000000.0;
                return (value % precion) * rcp(precion);
            }

            /**
             * 3次元のランダムな値を算出する
             */
            float3 Random3(uint3 src, int seed) {
                uint3 random;
                random.x = Xorshift32(mad(src.x, src.y, src.z));
                random.y = Xorshift32(mad(random.x, src.z, src.x) + seed);
                random.z = Xorshift32(mad(random.y, src.x, src.y) + seed);
                random.x = Xorshift32(mad(random.z, src.y, src.z) + seed);

                return float3(MapToFloat(random.x), MapToFloat(random.y), MapToFloat(random.z));
            }

            /**
             * 指定した座標に対して、ボロノイパターンの最も近いランダム点と、2番目に近いランダム点を取得する
             */
            void CreateVoronoi(float3 pos, out float3 closest, out float3 secondClosest, out float secondDistance) {
                // セル番号が負の値とならないようにオフセット加算
                const uint offset = 100;
                uint3 cellIdx;
                float3 reminders = modf(pos + offset, cellIdx);

                // 対象地点が所属するセルと隣接するセル全てに対してランダム点との距離をチェックし
                // 1番近い点と2番目に近い点を見付ける
                float2 closestDistances = 8.0;

                [unroll]
                for(int i = -1; i <= 1; i++)
                [unroll]
                for(int j = -1; j <= 1; j++)
                [unroll]
                for(int k = -1; k <= 1; k++) {
                    int3 neighborIdx = int3(i, j, k);

                    // そのセル内でのランダム点の相対位置を取得
                    float3 randomPos = Random3(cellIdx + neighborIdx, _RandomSeed);
                    // 対象地点からランダム点に向かうベクトル
                    float3 vec = randomPos + float3(neighborIdx) - reminders;
                    // 距離は全て二乗で比較
                    float distance = dot(vec, vec);

                    if (distance < closestDistances.x) {
                        closestDistances.y = closestDistances.x;
                        closestDistances.x = distance;
                        secondClosest = closest;
                        closest = vec;
                    } else if (distance < closestDistances.y) {
                        closestDistances.y = distance;
                        secondClosest = vec;
                    }
                }

                secondDistance = closestDistances.y;
            }

            /**
             * 指定した座標がボロノイ図の境界線となるかどうかを0～1で返す
             */
            float GetVoronoiBorder(float3 pos, out float secondDistance) {
                float3 a, b;
                CreateVoronoi(pos, a, b, secondDistance);

                /*
                 * 以下のベクトルの内積が境界線までの距離となる
                 * ・対象地点から、1番近いランダム点と2番目に近い点の中点に向かうベクトル
                 * ・1番近い点と2番目に近い点を結ぶ線の単位ベクトル
                 */
                float distance = dot(0.5 * (a + b), normalize(b - a));

                return 1.0 - smoothstep(_CrackWidth, _CrackWidth + _CrackWallWidth, distance);
            }

            /**
             * 指定した座標のひび度合いを0～1で返す
             */
            float GetCrackLevel(float3 pos) {
                // ボロノイ図の境界線で擬似的なひび模様を表現
                float secondDistance;
                float level = GetVoronoiBorder(pos * _CrackDetailedness, secondDistance);

                /*
                 * 部分的にひびを消すためにノイズを追加
                 * 計算量が少なくて済むようにボロノイのF2(2番目に近い点との距離)を利用する
                 * 距離が一定値以下の場合はひび対象から外す
                 */
                float f2Factor = 1.0 - sin(_CrackProgress * PI * 0.5);
                float minTh = (2.9 * f2Factor);
                float maxTh = (3.5 * f2Factor);
                float factor = smoothstep(minTh, maxTh, secondDistance * 2.0);
                level *= factor;

                return level;
            }

            /**
             * ひびが入った後の座標を計算する
             */
            float3 CalcCrackedPos(float3 localPos, float3 localNormal, float3 worldNormal, out float crackLevel) {
                crackLevel = (_CrackProgress == 0 || dot(worldNormal, GetViewForwardDir()) > 0.5) ? 0.0 : GetCrackLevel(localPos);

                // ひび対象の場合は法線と逆方向に凹ませる
                float depth = crackLevel * _CrackDepth;
                localPos -= localNormal * depth;

                return localPos;
            }

            /**
             * [ドメインシェーダー用]
             * OutputTopology:triangle_cwで生成された重心座標系の座標をsrcの空間の座標に換算する
             */
            float3 CalcSubdividedPos(float3 src[3], float3 baryCentricCoords) {
                return src[0] * baryCentricCoords.x + src[1] * baryCentricCoords.y + src[2] * baryCentricCoords.z;
            }

            /**
             * [フラグメントシェーダー用]
             * CrackLevelに応じたOcclusionを算出する
             */
            half CalcOcclusion(float crackLevel) {
                // ひびの深さに応じて影を濃くする
                half occlusion = pow(lerp(1.0, 0.90, crackLevel), 2.0);
                // ひびが深い部分で、隣接ピクセルの高低差が大きい場合は影を濃くする
                occlusion *= (crackLevel > 0.95 ? lerp(0.9, 1.0, 1.0 - smoothstep(0.0, 0.1, max(abs(ddy(crackLevel)), abs(ddx(crackLevel))))) : 1.0);

                return occlusion;
            }


            // ---------------------------------------------------------------------------------------
            // シェーダー関数
            // ---------------------------------------------------------------------------------------
            /**
             * 頂点シェーダー
             */
            v2d Vert(Attributes input) {
                v2d output;

                output.positionOS = input.positionOS;
                output.normalOS = input.normalOS;

                Varyings varyings = LitPassVertex(input);
                output.uv = varyings.uv;
                output.normalWS = varyings.normalWS;

                return output;
            }

            /**
             * メインハルシェーダー
             */
            [domain("tri")]
            [partitioning("integer")]
            [outputtopology("triangle_cw")]
            [outputcontrolpoints(3)]
            [patchconstantfunc("PatchConstantFunc")]
            v2d Hull(InputPatch<v2d, 3> inputs, uint id:SV_OutputControlPointID) {
                v2d output = inputs[id];
                return output;
            }

            /**
             * パッチ定数関数
             */
            patchConstParam PatchConstantFunc(InputPatch<v2d, 3> inputs) {
                patchConstParam output;

                int subdividingCount = (_CrackProgress == 0.0 || _SubdividingCount <= 1) ? 0 : _SubdividingCount;

                [unroll]
                for (uint i = 0; i < 3; i++) {
                    // カメラを向いていない面は分割しない
                    subdividingCount = subdividingCount > 0 && dot(inputs[i].normalWS, GetViewForwardDir()) <= 0.5 ? subdividingCount : 0;
                }

                // プロパティ設定に合う分割数算出
                float3 rawEdgeFactors = subdividingCount;
                float3 roundedEdgeTessFactors;
                float roundedInsideTessFactor;
                float unroundedInsideTessFactor;
                ProcessTriTessFactorsAvg(rawEdgeFactors, _SubdividingInsideScaleFactor, roundedEdgeTessFactors, roundedInsideTessFactor, unroundedInsideTessFactor);

                // 辺側、内側それぞれの分割数を指定
                output.edgeTessFactors[0] = roundedEdgeTessFactors.x;
                output.edgeTessFactors[1] = roundedEdgeTessFactors.y;
                output.edgeTessFactors[2] = roundedEdgeTessFactors.z;
                output.insideTessFactor = roundedInsideTessFactor;

                return output;
            }

            /**
             * ドメインシェーダー
             */
            [domain("tri")]
            d2f Domain(patchConstParam param, const OutputPatch<v2d, 3> inputs, float3 baryCentricCoords:SV_DomainLocation) {
                d2f output;

                // 算出された座標を重心座標系からローカル座標等に換算
                float3 srcLocalPositions[3];
                float3 srcUVs[3];
                float3 srcLocalNormals[3];
                [unroll]
                for (uint i = 0; i < 3; i++) {
                    srcLocalPositions[i] = inputs[i].positionOS.xyz;
                    srcUVs[i] = float3(inputs[i].uv, 0.0);
                    srcLocalNormals[i] = inputs[i].normalOS;
                }

                output.initPositionOS = CalcSubdividedPos(srcLocalPositions, baryCentricCoords);
                output.uv = CalcSubdividedPos(srcUVs, baryCentricCoords).xy;

                output.initNormalOS = CalcSubdividedPos(srcLocalNormals, baryCentricCoords);
                output.initNormalWS = TransformObjectToWorldNormal(output.initNormalOS);

                // 頂点がひび模様に重なる場合は凹ませる
                output.positionOS = CalcCrackedPos(output.initPositionOS, output.initNormalOS, output.initNormalWS, output.crackLevel);
                output.positionCS = TransformObjectToHClip(output.positionOS);

                return output;
            }

            /**
             * フラグメントシェーダー
             */
            half4 Frag(d2f input) : SV_Target {
                float crackLevel = input.crackLevel;
                float3 positionOS = _DrawsCrackWithPixelUnit ? CalcCrackedPos(input.initPositionOS, input.initNormalOS, input.initNormalWS, crackLevel) : input.positionOS;

                float3 positionWS = TransformObjectToWorld(positionOS);

                // 隣接のピクセルとのワールド座標の差分を取得後に外積を求めて法線算出
                float3 normalWS = crackLevel > 0.0 ? normalize(cross(ddy(positionWS), ddx(positionWS))) : input.initNormalWS;

                Varyings varyings = (Varyings)0;
                varyings.positionCS = input.positionCS;
                varyings.uv = input.uv;
                varyings.positionWS = positionWS;
                varyings.normalWS = normalWS;

                SurfaceData surfaceData;
                InitializeStandardLitSurfaceData(input.uv, surfaceData);

                OUTPUT_SH(normalWS, varyings.vertexSH);

                InputData inputData;
                InitializeInputData(varyings, surfaceData.normalTS, inputData);
                inputData.normalWS = crackLevel > 0.0 ? normalWS : inputData.normalWS;
                inputData.vertexLighting = VertexLighting(positionWS, inputData.normalWS);


                /* ひび模様 */
                // ひび対象の場合はクラックカラーを追加
                surfaceData.albedo = lerp(surfaceData.albedo, _CrackColor.rgb, crackLevel);

                // ひび部分はAO設定
                surfaceData.occlusion = min(surfaceData.occlusion, CalcOcclusion(crackLevel));

                half4 color = UniversalFragmentPBR(inputData, surfaceData);

                clip(color.a <= 0 ? -1 : 1);

                return color;
            }
            ENDHLSL
        }
    }

    FallBack "Universal Render Pipeline/Lit"
}
