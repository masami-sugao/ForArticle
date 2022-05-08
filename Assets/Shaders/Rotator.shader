Shader "Custom/Rotator" {
    Properties {
        [Header(Albedo)]
        [MainColor] _BaseColor("Base Color", Color) = (1.0, 1.0, 1.0, 1.0)
        [MainTexture] _BaseMap("Base Map", 2D) = "white" {}
    }

    SubShader {
        Tags {
            "RenderType" = "Transparent"
            "Queue" = "Transparent"
            "RenderPipeline" = "UniversalPipeline"
        }
        LOD 300

        HLSLINCLUDE
        #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
        ENDHLSL

        Pass {
            Name "Rotator"
            Tags { "LightMode" = "UniversalForward" }
            Blend SrcAlpha OneMinusSrcAlpha
            ZWrite Off

            HLSLPROGRAM

            // -------------------------------------
            // Material Keywords
            #pragma shader_feature_local_fragment _ALPHATEST_ON
            #pragma shader_feature_local_fragment _ALPHAPREMULTIPLY_ON
            #pragma shader_feature_local_fragment _SPECULARHIGHLIGHTS_OFF
            #pragma shader_feature_local_fragment _ENVIRONMENTREFLECTIONS_OFF
            #pragma shader_feature_local _RECEIVE_SHADOWS_OFF

            // --------------------------------------
            // GPU Instancing
            #pragma multi_compile_instancing

            #include "Packages/com.unity.render-pipelines.universal/Shaders/UnlitInput.hlsl"

            #pragma vertex Vert
            #pragma fragment Frag


            // ---------------------------------------------------------------------------------------
            // 変数宣言
            // ---------------------------------------------------------------------------------------
            static float DoubledPI = 2.0 * PI;


            // ---------------------------------------------------------------------------------------
            // 構造体
            // ---------------------------------------------------------------------------------------
            struct Attributes {
                float4 positionOS : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct Varyings {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
                float3 positionWS : TEXCOORD1;
            };


            // ---------------------------------------------------------------------------------------
            // メソッド
            // ---------------------------------------------------------------------------------------
            /**
             * 1/fノイズを加える
             */
            float CreateOneOverFNoise(float value) {
                return value +
                    (value <= 0.05
                        ? 0.06
                        : value < 0.5
                            ? 2.0 * value * value
                            : value >= 0.95
                                ? -0.04
                                : -2 * (1 - value) * (1 - value));
            }

            /**
             * 五次曲線による補完を行う
             */
            float CreateQuinticInterpolation(float value) {
                return value * value * value * (value * (value * 6.0 - 15.0) + 10.0);
            }


            // ---------------------------------------------------------------------------------------
            // シェーダー関数
            // ---------------------------------------------------------------------------------------
            /**
             * 頂点シェーダー
             */
            Varyings Vert(Attributes input) {
                Varyings output;

                VertexPositionInputs vertexInput = GetVertexPositionInputs(input.positionOS.xyz);
                output.vertex = vertexInput.positionCS;
                output.uv = input.uv;
                output.positionWS = vertexInput.positionWS;

                return output;
            }

            /**
             * フラグメントシェーダー
             */
            half4 Frag(Varyings input) : SV_Target {
                float2 uv = input.uv;

                // オブジェクトの拡縮分UV拡縮させる
                float scaleX = length(float3(unity_ObjectToWorld[0].x, unity_ObjectToWorld[1].x, unity_ObjectToWorld[2].x));
                float scaleY = length(float3(unity_ObjectToWorld[0].y, unity_ObjectToWorld[1].y, unity_ObjectToWorld[2].y));
                //float scaleZ = length(float3(unity_ObjectToWorld[0].z, unity_ObjectToWorld[1].z, unity_ObjectToWorld[2].z));
                //uv *= float2(scaleX, scaleY);

                // ScrollingDirの方向にUVを回転
                //float cosValue = dot(float2(0, -1), scrollingDir);
                //float sinValue = dot(float2(-1, 0), scrollingDir);
                //uv = uv * cosValue + float2(-uv.y, uv.x) * sinValue;

                float radian = frac(_Time.y) * DoubledPI;
                //float radian = frac(0.12) * DoubledPI;
                float2 uvForRolling = uv - 0.5;
                uv = (uvForRolling * cos(radian) + float2(-uvForRolling.y, uvForRolling.x) * sin(radian)) + 0.5;

                half4 color = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, uv);
                color *= _BaseColor;

                clip(color.a <= 0 ? -1 : 1);

                return color;
            }

            ENDHLSL
        }
    }
}
