Shader "Custom/Crack_Full" {
    Properties {
        [Header(Albedo)]
        [MainColor] _BaseColor("Base Color", Color) = (1.0, 1.0, 1.0, 1.0)
        [MainTexture] _BaseMap("Base Map", 2D) = "white" {}

        [Header(NormalMap)]
        [Toggle(_NORMALMAP)] _NORMALMAP("Normal Map使用有無", Int) = 0
        [NoScaleOffset] _BumpMap("Normal Map", 2D) = "bump" {}
        [HideInInspector] _BumpScale("Bump Scale", Float) = 1.0

        [Header(Occlution)]
        [Toggle(_OCCLUSIONMAP)] _OCCLUSIONMAP("Occlusion Map使用有無", Int) = 0
        [NoScaleOffset] _OcclusionMap("Occlusion Map", 2D) = "white" {}
        [HideInInspector] _OcclusionStrength("Strength", Range(0.0, 1.0)) = 1.0

        [Header(Metallic and Smoothness)]
        _Smoothness("Smoothness(Map使用時はAlpha=1の箇所の値)", Range(0.0, 1.0)) = 0.0
        [Toggle(_METALLICSPECGLOSSMAP)] _METALLICSPECGLOSSMAP("Metallic and Smoothness Map使用有無", Int) = 0
        _Metallic("Metallic(Map不使用時のみ)", Range(0.0, 1.0)) = 0.0
        [NoScaleOffset] _MetallicGlossMap("Metallic and Smoothnes Map", 2D) = "white" {}

        [Header(Emission)]
        [Toggle(_EMISSION)] _EMISSION("Emission使用有無", Int) = 0
        [HDR] _EmissionColor("Emission Color", Color) = (0.0 ,0.0, 0.0)
        [NoScaleOffset] _EmissionMap("Emission Map", 2D) = "white" {}

        [Header(Crack)]
        [Toggle] _CRACK("クラック利用有無", Int) = 0
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
        [Toggle] _PN_TRIANGLES("PN-Triangles適用有無", Int) = 0
        _PnTriFactor("PN-Triangles適用係数", Range(0.0, 1.0)) = 1.0
        [Toggle] _AdaptsPolygonEdgeToPnTri("PN-Trianglesを辺にも適用するかどうか", Int) = 1

        [Header(Wireframe(ForDebug))]
        // ワイヤーフレーム表示（デバッグ用）
        [Toggle] _WIREFRAME("ワイヤーフレーム表示有無(デバッグ用)", Int) = 0
        _WireframeWidth("ワイヤーフレーム幅", Range(1, 5)) = 1
        _WireframeColor("ワイヤーフレーム色", Color) = (0.0, 0.0, 1.0)
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
            #pragma shader_feature_local _NORMALMAP
            #pragma shader_feature_local_fragment _ALPHATEST_ON
            #pragma shader_feature_local_fragment _ALPHAPREMULTIPLY_ON
            #pragma shader_feature_local_fragment _EMISSION
            #pragma shader_feature_local_fragment _METALLICSPECGLOSSMAP
            #pragma shader_feature_local_fragment _SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A
            #pragma shader_feature_local_fragment _OCCLUSIONMAP
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
            #pragma shader_feature_local _ _CRACK_ON
            #pragma shader_feature_local _ _PN_TRIANGLES_ON
            #pragma shader_feature_local _ _WIREFRAME_ON


            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitInput.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/Shaders/LitForwardPass.hlsl"

            #pragma vertex Vert
            #pragma hull Hull
            #pragma domain Domain
            #pragma geometry Geom
            #pragma fragment Frag

            #pragma require geometry
            #pragma require tessellation tessHW


            // ---------------------------------------------------------------------------------------
            // 変数宣言
            // ---------------------------------------------------------------------------------------
            float _CrackProgress;
#ifdef _CRACK_ON
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
#ifdef _PN_TRIANGLES_ON
            float _PnTriFactor;
            bool _AdaptsPolygonEdgeToPnTri;
#endif
#endif // _CRACK_ON

#ifdef _WIREFRAME_ON
            float _WireframeWidth;
            half3 _WireframeColor;
#endif

#ifdef _CRACK_ON
            static float OneThird = rcp(3.0);
            static float OneSixth = rcp(6.0);
#endif


            // ---------------------------------------------------------------------------------------
            // 構造体
            // ---------------------------------------------------------------------------------------
            struct v2h {
                float4 localPos : POSITION;
                float2 uv : TEXCOORD0;
                float3 localNormal : NORMAL;
#ifdef _CRACK_ON
                float3 worldNormal : TEXCOORD1;
#endif
#ifdef _NORMALMAP
                half4 worldTangent : TEXCOORD2;
#endif
            };

            struct h2d {
                float4 localPos : POSITION;
                float2 uv : TEXCOORD0;
                float3 localNormal : NORMAL;
#ifdef _CRACK_ON
                float3 worldNormal : TEXCOORD1;
#endif
#ifdef _NORMALMAP
                half4 worldTangent : TEXCOORD2;
#endif
            };

            struct patchConstParam {
                float edgeTessFactors[3] : SV_TessFactor;
                float insideTessFactor : SV_InsideTessFactor;

#if defined(_CRACK_ON) && defined(_PN_TRIANGLES_ON)
                // PN-Triangles計算用のコントロールポイント
                float3 b111 : TEXCOORD0;
                float3 localPositions[3][3] : TEXCOORD1;
#endif
            };

            struct d2g {
                float4 localPos : POSITION;
                float2 uv : TEXCOORD0;
                float3 worldNormal : TEXCOORD1;
#ifdef _CRACK_ON
                float3 localNormal : NORMAL;
#endif
#ifdef _NORMALMAP
                half4 worldTangent : TEXCOORD2;
#endif
            };

            struct g2f {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 initWorldNormal : TEXCOORD1;
                float3 localPos : TEXCOORD2;
#ifdef _NORMALMAP
                half4 worldTangent : TEXCOORD3;
#endif
#ifdef _CRACK_ON
                float3 initLocalPos : TEXCOORD4;
                float3 initLocalNormal : NORMAL;
                float crackLevel : TEXCOORD5;
#endif
#ifdef _WIREFRAME_ON
                float3 baryCentricCoords : TEXCOORD6;
#endif
            };

            /**
             * ジオメトリシェーダー内で利用するパラメータ格納用の構造体
             */
            struct geoParam {
                d2g input;
                g2f output;
            };


            // ---------------------------------------------------------------------------------------
            // メソッド
            // ---------------------------------------------------------------------------------------
#ifdef _CRACK_ON
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
             * 五次曲線による補完を行う
             */
            float CreateQuinticInterpolation(float value) {
                return value * value * value * (value * (value * 6.0 - 15.0) + 10.0);
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
#endif // _CRACK_ON

            /**
             * ガンマ補正関数
             */
            float gammaCorrect(float gamma, float x) {
                return pow(abs(x), 1.0 / gamma);
            }

            /**
             * ガンマ補正関数の派生バージョンの一つであるbias補正関数を実行する。
             * bias(b, 0.5) = bとなるように定義されたパラメータbでガンマ補正を行う。
             *
             * 参考: https://qiita.com/oishihiroaki/items/9d899cdcb9bee682531a
             */
            float bias(float b, float x) {
                return gammaCorrect(log(0.5) / log(b), x);
            }

            /**
             * ガンマ補正関数の派生バージョンの一つであるgain補正関数を実行する。
             * gの値によらずxが0.5のとき0.5を返す、bias曲線で形成されたS字曲線を返す。
             *
             * 参考: https://qiita.com/oishihiroaki/items/9d899cdcb9bee682531a
             */
            float gain(float g, float x) {
                if (x < 0.5) {
                    return bias(1.0 - g, 2.0 * x) / 2.0;
                } else {
                    return 1.0 - bias(1.0 - g, 2.0 - 2.0 * x) / 2.0;
                }
            }

#if defined(_CRACK_ON) && defined(_PN_TRIANGLES_ON)
            /**
             * [テッセレーションシェーダー用]
             * PN-Triangles用のコントロールポイントを算出する
             */
            float3 CalcControlPointForPnTri(float3 posA, float3 posB, float3 normalA) {
                // PosAとPosBを結ぶ線分を1:2に分けた地点をPosAの接平面上に移動した座標を算出
                return (2.0 * posA + posB - (dot((posB - posA), normalA) * normalA)) * OneThird;
            }
#endif

            /**
             * [ドメインシェーダー用]
             * OutputTopology:triangle_cwで生成された重心座標系の座標をsrcの空間の座標に換算する
             */
            float3 CalcSubdividedPos(float3 src[3], float3 baryCentricCoords) {
                return src[0] * baryCentricCoords.x + src[1] * baryCentricCoords.y + src[2] * baryCentricCoords.z;
            }

#if defined(_CRACK_ON) && defined(_PN_TRIANGLES_ON)
            /**
             * [ドメインシェーダー用]
             * PN-TrianglesのB111の位置を計算する
             *
             * 参考：https://alex.vlachos.com/graphics/CurvedPNTriangles.pdf
             */
            float3 CalcPnTriB111Pos(float3 controlPoints[3][3]) {
                float3 b300 = controlPoints[0][0];
                float3 b210 = controlPoints[0][1];
                float3 b120 = controlPoints[0][2];

                float3 b030 = controlPoints[1][0];
                float3 b021 = controlPoints[1][1];
                float3 b012 = controlPoints[1][2];

                float3 b003 = controlPoints[2][0];
                float3 b102 = controlPoints[2][1];
                float3 b201 = controlPoints[2][2];

                float3 e = (b210 + b120 + b021 + b012 + b102 + b201) * OneSixth;
                float3 v = (b003 + b030 + b300) * OneThird;

                return e + ((e - v) * 0.5);
            }

            /**
             * [ドメインシェーダー用]
             * PN-Trianglesを用いてカーブ上になるように変位させた座標を算出する
             *
             * 以下を参考にした
             * - PN-Trianglesの理論
             *   https://alex.vlachos.com/graphics/CurvedPNTriangles.pdf
             * - 実装(PN-Triangles-AENの実装ではあるが、PN-Trianglesとの共通部分も多い)
             *   https://developer.download.nvidia.com/whitepapers/2010/PN-AEN-Triangles-Whitepaper.pdf
             */
            float3 CalcPnTriPosition(float3 controlPoints[3][3], float3 b111, float3 baryCentricCoords) {
                float u = baryCentricCoords.x;
                float v = baryCentricCoords.y;
                float w = baryCentricCoords.z;
                float uu = u * u;
                float vv = v * v;
                float ww = w * w;
                float uu3 = 3.0 * uu;
                float vv3 = 3.0 * vv;
                float ww3 = 3.0 * ww;

                return controlPoints[0][0] * u * uu
                    + controlPoints[1][0] * v * vv
                    + controlPoints[2][0] * w * ww
                    + controlPoints[0][1] * uu3 * v
                    + controlPoints[0][2] * vv3 * u
                    + controlPoints[1][1] * vv3 * w
                    + controlPoints[1][2] * ww3 * v
                    + controlPoints[2][1] * ww3 * u
                    + controlPoints[2][2] * uu3 * w
                    + b111 * 6.0 * w * u * v;
            }
#endif

            /**
             * [ジオメトリシェーダー用]
             * パラメータをジオメトリックシェーダーの返却値となるTriangleStreamに追加する
             */
            void SetTriVerticesToStream(geoParam param[3], inout TriangleStream<g2f> outStream) {
                [unroll]
                for (int i = 0; i < 3; i++) {
#ifdef _WIREFRAME_ON
                    // ワイヤーフレーム描画用
                    param[i].output.baryCentricCoords = float3(i == 0, i == 1, i == 2);
#endif

                    outStream.Append(param[i].output);
                }

                outStream.RestartStrip();
            }

#ifdef _CRACK_ON
            /**
             * [フラグメントシェーダー用]
             * CrackLevelに応じたOcclusionを算出する
             */
            half CalcOcclusion(float crackLevel) {
                // ひびの深さに応じて影を濃くする
                half occlusion = pow(lerp(1.0, 0.9, crackLevel), 2.0);
                // ひびが深い部分で、隣接ピクセルの高低差が大きい場合は影を濃くする
                occlusion *= (crackLevel > 0.95 ? lerp(0.9, 1.0, 1.0 - smoothstep(0.0, 0.1, max(abs(ddy(crackLevel)), abs(ddx(crackLevel))))) : 1.0);

                return occlusion;
            }

            /**
             * [フラグメントシェーダー用]
             * Lighting.hlslのLightingPhysicallyBased()のランバート法をハーフランバートに変更し、crackLevelに応じたocclusion設定
             */
            half3 LightConsideringCrack(BRDFData brdfData, BRDFData brdfDataClearCoat,
                half3 lightColor, half3 lightDirectionWS, half lightAttenuation,
                half3 normalWS, half3 viewDirectionWS,
                half clearCoatMask, bool specularHighlightsOff, float crackLevel)
            {
                // ひび部分のみlighting.hlslのLightingPhysicallyBased()からハーフランバート法に変更
                //half NdotL = saturate(dot(normalWS, lightDirectionWS));
                //half3 radiance = lightColor * (lightAttenuation * NdotL);
                half lightingRatio = crackLevel > 0.0 ? pow(saturate(dot(normalWS, lightDirectionWS)) * 0.5 + 0.5, 2) : saturate(dot(normalWS, lightDirectionWS));

                // Occlusion
                // ひびの深さに応じて影を濃くする
                half occlusion = CalcOcclusion(crackLevel);
                half3 radiance = lightColor * (lightAttenuation * lightingRatio * occlusion);

                half3 brdf = brdfData.diffuse;
#ifndef _SPECULARHIGHLIGHTS_OFF
                [branch] if (!specularHighlightsOff) {
                    brdf += brdfData.specular * DirectBRDFSpecular(brdfData, normalWS, lightDirectionWS, viewDirectionWS);

#if defined(_CLEARCOAT) || defined(_CLEARCOATMAP)
                    // Clear coat evaluates the specular a second timw and has some common terms with the base specular.
                    // We rely on the compiler to merge these and compute them only once.
                    half brdfCoat = kDielectricSpec.r * DirectBRDFSpecular(brdfDataClearCoat, normalWS, lightDirectionWS, viewDirectionWS);

                    // Mix clear coat and base layer using khronos glTF recommended formula
                    // https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_clearcoat/README.md
                    // Use NoV for direct too instead of LoH as an optimization (NoV is light invariant).
                    half NoV = saturate(dot(normalWS, viewDirectionWS));
                    // Use slightly simpler fresnelTerm (Pow4 vs Pow5) as a small optimization.
                    // It is matching fresnel used in the GI/Env, so should produce a consistent clear coat blend (env vs. direct)
                    half coatFresnel = kDielectricSpec.x + kDielectricSpec.a * Pow4(1.0 - NoV);

                    brdf = brdf * (1.0 - clearCoatMask * coatFresnel) + brdfCoat * clearCoatMask;
#endif // _CLEARCOAT
                }
#endif // _SPECULARHIGHLIGHTS_OFF

                return brdf * radiance;
            }

            /**
             * [フラグメントシェーダー用]
             * Lighting.hlslのLightingPhysicallyBased()のランバート法をハーフランバートに変更し、crackLevelに応じたocclusion設定
             */
            half3 LightConsideringCrack(BRDFData brdfData, BRDFData brdfDataClearCoat, Light light, half3 normalWS, half3 viewDirectionWS, half clearCoatMask, bool specularHighlightsOff, float crackLevel) {
                return LightConsideringCrack(brdfData, brdfDataClearCoat, light.color, light.direction, light.distanceAttenuation * light.shadowAttenuation, normalWS, viewDirectionWS, clearCoatMask, specularHighlightsOff, crackLevel);
            }

            /**
             * [フラグメントシェーダー用]
             * Lighting.hlslのUniversalFragmentPBR()のライト設定を自作関数に変更
             */
            half4 UniversalFragmentRenderingWithCrack(InputData inputData, SurfaceData surfaceData, float crackLevel) {
#if defined(_SPECULARHIGHLIGHTS_OFF)
                bool specularHighlightsOff = true;
#else
                bool specularHighlightsOff = false;
#endif
                BRDFData brdfData;

                // NOTE: can modify "surfaceData"...
                InitializeBRDFData(surfaceData, brdfData);

#if defined(DEBUG_DISPLAY)
                half4 debugColor;

                if (CanDebugOverrideOutputColor(inputData, surfaceData, brdfData, debugColor)) {
                    return debugColor;
                }
 #endif

                // Clear-coat calculation...
                BRDFData brdfDataClearCoat = CreateClearCoatBRDFData(surfaceData, brdfData);
                half4 shadowMask = CalculateShadowMask(inputData);
                AmbientOcclusionFactor aoFactor = CreateAmbientOcclusionFactor(inputData, surfaceData);
                uint meshRenderingLayers = GetMeshRenderingLightLayer();
                Light mainLight = GetMainLight(inputData, shadowMask, aoFactor);

                // NOTE: We don't apply AO to the GI here because it's done in the lighting calculation below...
                MixRealtimeAndBakedGI(mainLight, inputData.normalWS, inputData.bakedGI);

                LightingData lightingData = CreateLightingData(inputData, surfaceData);

                lightingData.giColor = GlobalIllumination(brdfData, brdfDataClearCoat, surfaceData.clearCoatMask,
                                                          inputData.bakedGI, aoFactor.indirectAmbientOcclusion, inputData.positionWS,
                                                          inputData.normalWS, inputData.viewDirectionWS);

                if (IsMatchingLightLayer(mainLight.layerMask, meshRenderingLayers)) {
                    // Lighting.hlslのLightingPhysicallyBased()から自作関数に変更
                    lightingData.mainLightColor = LightConsideringCrack(brdfData, brdfDataClearCoat,
                                                                          mainLight,
                                                                          inputData.normalWS, inputData.viewDirectionWS,
                                                                          surfaceData.clearCoatMask, specularHighlightsOff, crackLevel);
                    //lightingData.mainLightColor = LightingPhysicallyBased(brdfData, brdfDataClearCoat,
                    //                                                      mainLight,
                    //                                                      inputData.normalWS, inputData.viewDirectionWS,
                    //                                                      surfaceData.clearCoatMask, specularHighlightsOff);
                }

#if defined(_ADDITIONAL_LIGHTS)
                uint pixelLightCount = GetAdditionalLightsCount();

#if USE_CLUSTERED_LIGHTING
                for (uint lightIndex = 0; lightIndex < min(_AdditionalLightsDirectionalCount, MAX_VISIBLE_LIGHTS); lightIndex++) {
                    Light light = GetAdditionalLight(lightIndex, inputData, shadowMask, aoFactor);

                    if (IsMatchingLightLayer(light.layerMask, meshRenderingLayers)) {
                        // Lighting.hlslのLightingPhysicallyBased()から自作関数に変更
                        lightingData.additionalLightsColor += LightConsideringCrack(brdfData, brdfDataClearCoat, light,
                                                                                      inputData.normalWS, inputData.viewDirectionWS,
                                                                                      surfaceData.clearCoatMask, specularHighlightsOff, crackLevel);
                        //lightingData.additionalLightsColor += LightingPhysicallyBased(brdfData, brdfDataClearCoat, light,
                        //                                                              inputData.normalWS, inputData.viewDirectionWS,
                        //                                                              surfaceData.clearCoatMask, specularHighlightsOff);
                    }
                }
#endif

                LIGHT_LOOP_BEGIN(pixelLightCount)
                    Light light = GetAdditionalLight(lightIndex, inputData, shadowMask, aoFactor);

                    if (IsMatchingLightLayer(light.layerMask, meshRenderingLayers))
                    {
                        // Lighting.hlslのLightingPhysicallyBased()から自作関数に変更
                        lightingData.additionalLightsColor += LightConsideringCrack(brdfData, brdfDataClearCoat, light,
                                                                                      inputData.normalWS, inputData.viewDirectionWS,
                                                                                      surfaceData.clearCoatMask, specularHighlightsOff, crackLevel);
                        //lightingData.additionalLightsColor += LightingPhysicallyBased(brdfData, brdfDataClearCoat, light,
                        //                                                              inputData.normalWS, inputData.viewDirectionWS,
                        //                                                              surfaceData.clearCoatMask, specularHighlightsOff);
                    }
                LIGHT_LOOP_END
#endif

#if defined(_ADDITIONAL_LIGHTS_VERTEX)
                lightingData.vertexLightingColor += inputData.vertexLighting * brdfData.diffuse;
#endif

                return CalculateFinalColor(lightingData, surfaceData.alpha);
            }

#endif // _CRACK_ON


            // ---------------------------------------------------------------------------------------
            // シェーダー関数
            // ---------------------------------------------------------------------------------------
            /**
             * 頂点シェーダー
             */
            v2h Vert(Attributes input) {
                v2h output;

                output.localPos = input.positionOS;

                output.localNormal = input.normalOS;
                Varyings varyings = LitPassVertex(input);
                output.uv = varyings.uv;
#ifdef _CRACK_ON
                output.worldNormal = varyings.normalWS;
#endif

#ifdef _NORMALMAP
                VertexNormalInputs normalInput = GetVertexNormalInputs(input.normalOS, input.tangentOS);
                real sign = input.tangentOS.w * GetOddNegativeScale();
                output.worldTangent = half4(normalInput.tangentWS.xyz, sign);
#endif

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
            h2d Hull(InputPatch<v2h, 3> inputs, uint id:SV_OutputControlPointID) {
                h2d output;

                v2h input = inputs[id];
                output.localPos = input.localPos;
                output.uv = input.uv;
                output.localNormal = input.localNormal;
#ifdef _CRACK_ON
                output.worldNormal = input.worldNormal;
#endif
#ifdef _NORMALMAP
                output.worldTangent = input.worldTangent;
#endif

                return output;
            }

            /**
             * パッチ定数関数
             */
            patchConstParam PatchConstantFunc(InputPatch<v2h, 3> inputs) {
                patchConstParam output;

#ifdef _CRACK_ON
                int subdividingCount = (_CrackProgress == 0.0 || _SubdividingCount <= 1) ? 0 : _SubdividingCount;

                [unroll]
                for (uint i = 0; i < 3; i++) {
                    // カメラを向いていない面は分割しない
                    subdividingCount = subdividingCount > 0 && dot(inputs[i].worldNormal, GetViewForwardDir()) <= 0.5 ? subdividingCount : 0;
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

#ifdef _PN_TRIANGLES_ON
                // PN-Triangles計算用のコントロールポイント算出
                bool usesPnTriangles = _PnTriFactor > 0.0 && subdividingCount > 1;

                [unroll]
                for (i = 0; i < 3; i++) {
                    uint nextId = (i + 1) % 3;
                    output.localPositions[i][0] = inputs[i].localPos.xyz;

                    if (usesPnTriangles) {
                        output.localPositions[i][1]
                            = CalcControlPointForPnTri(inputs[i].localPos.xyz, inputs[nextId].localPos.xyz, inputs[i].localNormal);
                        output.localPositions[i][2]
                            = CalcControlPointForPnTri(inputs[nextId].localPos.xyz, inputs[i].localPos.xyz, inputs[nextId].localNormal);
                    } else {
                        output.localPositions[i][1] = 0.0;
                        output.localPositions[i][2] = 0.0;
                    }
                }

                output.b111 = usesPnTriangles ? CalcPnTriB111Pos(output.localPositions) : 0.0;
#endif

#else
                output.edgeTessFactors[0] = 1;
                output.edgeTessFactors[1] = 1;
                output.edgeTessFactors[2] = 1;
                output.insideTessFactor = 1;
#endif // _CRACK_ON

                return output;
            }

            /**
             * ドメインシェーダー
             */
            [domain("tri")]
            d2g Domain(patchConstParam param, const OutputPatch<h2d, 3> inputs, float3 baryCentricCoords:SV_DomainLocation) {
                d2g output;

                // まずはフラットなポリゴン上に算出された座標を求める
                // 算出された座標を重心座標系からローカル座標等に換算
                float3 srcPositions[3];
                float3 srcLocalNormals[3];
                float3 srcUVs[3];
                float3 srcWorldTangents[3];
                [unroll]
                for (uint i = 0; i < 3; i++) {
                    srcPositions[i] = inputs[i].localPos.xyz;
                    srcLocalNormals[i] = inputs[i].localNormal;
                    srcUVs[i] = float3(inputs[i].uv, 0.0);
#ifdef _NORMALMAP
                    srcWorldTangents[i] = inputs[i].worldTangent.xyz;
#endif
                }
                float3 flatLocalPos = CalcSubdividedPos(srcPositions, baryCentricCoords);
                output.uv = CalcSubdividedPos(srcUVs, baryCentricCoords).xy;

                // 法線についてはPN-Trianglesで計算するとクラック用の頂点移動時に亀裂が発生しやすくなるので、フラットなポリゴンの法線を採用
                float3 localNormal = CalcSubdividedPos(srcLocalNormals, baryCentricCoords);
                output.worldNormal = TransformObjectToWorldNormal(localNormal);
#ifdef _CRACK_ON
                output.localNormal = localNormal;
#endif
#ifdef _NORMALMAP
                output.worldTangent =  half4(CalcSubdividedPos(srcWorldTangents, baryCentricCoords), inputs[0].worldTangent.w);
#endif

#if defined(_CRACK_ON) && defined(_PN_TRIANGLES_ON)
                // PN-Trianglesを適用すると亀裂が発生する場合はポリゴンの辺上の頂点は変位させない
                // （重心座標系では頂点から向かいの辺に向かって座標が1→0と変化することを利用）
                bool isOnSides = min(min(baryCentricCoords.x, baryCentricCoords.y), baryCentricCoords.z) == 0;
                if (_PnTriFactor == 0.0 || (!_AdaptsPolygonEdgeToPnTri && isOnSides)) {
                    output.localPos = float4(flatLocalPos, inputs[0].localPos.w);
                } else {
                    // PN-Trianglesを用いてカーブ上になるように座標変位
                    float3 b111 = CalcPnTriB111Pos(param.localPositions);
                    float3 pnTriLocalPos = CalcPnTriPosition(param.localPositions, b111, baryCentricCoords);

                    output.localPos = float4(lerp(flatLocalPos, pnTriLocalPos, _PnTriFactor), inputs[0].localPos.w);
                }
#else
                output.localPos = float4(flatLocalPos, inputs[0].localPos.w);
#endif

                return output;
            }

            /**
             * ジオメトリシェーダー
             */
            [maxvertexcount(3)]
            void Geom(triangle d2g input[3], inout TriangleStream<g2f> outStream) {
                geoParam param[3];

                [unroll]
                for (int i = 0; i < 3; i++) {
                    param[i].input = input[i];

#ifdef _CRACK_ON
                    // 頂点が模様に重なる場合は凹ませる
                    param[i].output.initLocalPos = input[i].localPos.xyz;
                    param[i].output.initLocalNormal = input[i].localNormal;
                    param[i].output.localPos = CalcCrackedPos(input[i].localPos.xyz, input[i].localNormal, input[i].worldNormal, param[i].output.crackLevel);
#else
                    param[i].output.localPos = input[i].localPos.xyz;
#endif

                    param[i].output.initWorldNormal = input[i].worldNormal;
                    param[i].output.vertex = TransformObjectToHClip(param[i].output.localPos);
                    param[i].output.uv = input[i].uv;
#ifdef _NORMALMAP
                    param[i].output.worldTangent = input[i].worldTangent;
#endif
                }

                SetTriVerticesToStream(param, outStream);
            }

            /**
             * フラグメントシェーダー
             */
            half4 Frag(g2f input) : SV_Target {

#ifdef _CRACK_ON
                float crackLevel = input.crackLevel;
                float3 positionOS = _DrawsCrackWithPixelUnit ? CalcCrackedPos(input.initLocalPos, input.initLocalNormal, input.initWorldNormal, crackLevel) : input.localPos;
                positionOS -= input.initLocalNormal * _AdditionalCrackDepthForLighting * crackLevel;
#else
                float crackLevel = 0.0;
                float3 positionOS = input.localPos;
#endif

                float3 positionWS = TransformObjectToWorld(positionOS);

                // 隣接のピクセルとのワールド座標の差分を取得後に外積を求めて法線算出
#ifdef _CRACK_ON
                float3 normalWS = crackLevel > 0.0 ? normalize(cross(ddy(positionWS), ddx(positionWS))) : input.initWorldNormal;
#else
                float3 normalWS = input.initWorldNormal;
#endif

                Varyings varyings = (Varyings)0;
                varyings.uv = input.uv;
                varyings.positionWS = positionWS;
                varyings.normalWS = normalWS;
#ifdef _NORMALMAP
                varyings.tangentWS = input.worldTangent;
#endif
                varyings.viewDirWS = GetWorldSpaceViewDir(positionWS);
                varyings.positionCS = input.vertex;

                SurfaceData surfaceData;
                InitializeStandardLitSurfaceData(input.uv, surfaceData);

                OUTPUT_SH(normalWS, varyings.vertexSH);

                InputData inputData;
                InitializeInputData(varyings, surfaceData.normalTS, inputData);
                inputData.normalWS = crackLevel > 0.0 ? normalWS : inputData.normalWS;
                inputData.vertexLighting = VertexLighting(positionWS, inputData.normalWS);


#ifdef _CRACK_ON
                /* ひび模様 */
                // ひび対象の場合はクラックカラーを追加
                surfaceData.albedo = lerp(surfaceData.albedo, _CrackColor.rgb, crackLevel);

                // ひび部分はAO設定
                surfaceData.occlusion = min(surfaceData.occlusion, CalcOcclusion(crackLevel));
#endif

#ifdef _WIREFRAME_ON
                /* ワイヤーフレーム */
                // fwidthで1ピクセルの変化量を求め、ワイヤーを描く値を算出
                float3 thOfJudgingSides = fwidth(input.baryCentricCoords) * _WireframeWidth;
                // 処理中のピクセルの値が_WireframeWidthピクセル分での変化量より小さい場合は辺からの距離が短い
                // （重心座標系では頂点から向かいの辺に向かって座標が1→0と変化することを利用）
                float3 isOnSides = 1 - smoothstep(0.0, thOfJudgingSides, input.baryCentricCoords);
                surfaceData.emission = lerp(surfaceData.emission, _WireframeColor, max(max(isOnSides.x, isOnSides.y), isOnSides.z) * 0.9);
#endif

#ifdef _CRACK_ON
                half4 color = UniversalFragmentRenderingWithCrack(inputData, surfaceData, crackLevel);
#else
                half4 color = UniversalFragmentPBR(inputData, surfaceData);
#endif

                clip(color.a <= 0 ? -1 : 1);

                return color;
            }
            ENDHLSL
        }
    }

    FallBack "Universal Render Pipeline/Lit"
}
