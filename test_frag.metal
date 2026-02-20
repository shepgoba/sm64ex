#include <metal_stdlib>
using namespace metal;
struct VertexIn {
float4 pos [[attribute(0)]];
float3 input1 [[attribute(1)]];
};

struct VertexOut {
float4 pos [[position]];
float3 output1;
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]]) {
VertexOut out;
out.pos = in.pos;
out.output1 = in.input1;
return out;
}

#include <metal_stdlib>
using namespace metal;

struct FSIn {
    float4 position [[position]];
    float3 input1;
};

struct FragmentUniforms {
};

fragment float4 fragment_main(
    FSIn in [[stage_in]],
    constant FragmentUniforms& uniforms [[buffer(0)]]
) {
float3 texel = in.input1;
    return float4(texel, 1.0);
}