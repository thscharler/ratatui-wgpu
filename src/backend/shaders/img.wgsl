struct VertexOutput {
    @location(0) UV: vec2<f32>,
    @builtin(position) gl_Position: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> ScreenSize: vec4<f32>;
@group(1) @binding(0)
var<uniform> ImageSize: vec2<f32>;
@group(1) @binding(1)
var<uniform> ViewSize: vec2<f32>;
@group(1) @binding(2)
var<uniform> UVTransform: mat2x3<f32>;

@vertex
fn vs_main(
    @location(0) VertexCoord: vec2<f32>,
    @location(1) UV: vec2<f32>,
) -> VertexOutput {
    let gl_Position = vec4<f32>((2.0 * VertexCoord / ScreenSize.xy - 1.0) * vec2(1.0, -1.0), 0.0, 1.0);

    let correctedUV = vec3<f32>(UV, 1.0) * UVTransform;

    return VertexOutput(correctedUV.xy, gl_Position);
}

struct FragmentOutput {
    @location(0) FragColor: vec4<f32>,
}

@group(2) @binding(0)
var Sampler: sampler;
@group(2) @binding(1)
var Image: texture_2d<f32>;

@fragment
fn fs_main(
    @location(0) UV: vec2<f32>,
) -> FragmentOutput {
    if (UV.x < 0.0 || UV.x > 1.0 || UV.y < 0.0 || UV.y > 1.0) {
        return FragmentOutput(vec4<f32>(0.0, 0.0, 0.0, 0.0));
    }

    let imageSize = textureDimensions(Image);
    let size = vec2<f32>(f32(imageSize.x), f32(imageSize.y));

    var textureColor = textureSample(Image, Sampler, UV);

    return FragmentOutput(textureColor);
}