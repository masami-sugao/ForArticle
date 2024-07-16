Shader "Custom/Scrolling" {
    Properties {
        [Header(Albedo)]
        [MainColor] _BaseColor("Base Color", Color) = (1.0, 1.0, 1.0, 1.0)
        [MainTexture] _BaseMap("Base Map", 2D) = "white" {}

        [Toggle] _IsAutoScaling("IsAutoScaling", Int) = 0

        [Header(Scrolling)]
        [Toggle] _IsHighSpeed("IsHighSpeed", Int) = 0
        _ScrollingSpeed("スクロール速度", Range(1, 3)) = 1
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
            Name "Scrolling"
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
            bool _IsHighSpeed;
            int _ScrollingSpeed;
            bool _IsAutoScaling;

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
            // シェーダー関数
            // ---------------------------------------------------------------------------------------
            /**
             * 頂点シェーダー
             */
            Varyings Vert(Attributes input) {
                Varyings output;

                VertexPositionInputs vertexInput = GetVertexPositionInputs(input.positionOS.xyz);
                output.vertex = vertexInput.positionCS;
                output.uv = TRANSFORM_TEX(input.uv, _BaseMap);
                output.positionWS = vertexInput.positionWS;

                return output;
            }

            /**
             * フラグメントシェーダー
             */
            half4 Frag(Varyings input) : SV_Target {
                float2 uv = input.uv;

                if (_IsAutoScaling) {
                    // オブジェクトの拡縮分UV拡縮させる
                    float scaleX = length(float3(unity_ObjectToWorld[0].x, unity_ObjectToWorld[1].x, unity_ObjectToWorld[2].x));
                    //float scaleY = length(float3(unity_ObjectToWorld[0].y, unity_ObjectToWorld[1].y, unity_ObjectToWorld[2].y));
                    float scaleZ = length(float3(unity_ObjectToWorld[0].z, unity_ObjectToWorld[1].z, unity_ObjectToWorld[2].z));
                    //uv *= float2(scaleX / 4, scaleY / 4);
                    uv *= float2(scaleX / 4, scaleZ / 4);
                }

                [branch] switch (_ScrollingSpeed)
                {
                    case 1:
                        uv += _Time.x;
                        break;
                    case 2:
                        uv += frac(_Time.y) * 1.5;
                        break;
                    case 3:
                        uv += frac(_Time.w) / 2;
                        break;
                }
                half4 color = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, uv);
                color *= _BaseColor;

                clip(color.a <= 0 ? -1 : 1);

                return color;
            }

            ENDHLSL
        }
    }
}
