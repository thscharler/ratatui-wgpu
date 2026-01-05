struct VertexOutput {
    @location(0) UV: vec2<f32>,
    @location(1) @interpolate(flat) BgIndex: u32,
    @location(2) @interpolate(flat) BgColor: u32,
    @builtin(position) gl_Position: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> ScreenSize: vec4<f32>;
@group(0) @binding(1)
var<uniform> BgSize: vec2<u32>;
@group(0) @binding(2)
var<uniform> AtlasSize: vec4<f32>;

@group(1) @binding(0)
var Mask: texture_2d<f32>;
@group(1) @binding(1)
var Sampler: sampler;

@group(2) @binding(0)
var<storage, read_write> BgBuffer: array<u32, 65636>;

@vertex
fn vs_main(
    @location(0) VertexCoord: vec2<f32>,
    @location(1) UV: vec2<f32>,
    @location(2) BgIndex: u32,
    @location(3) BgColor: u32,
) -> VertexOutput {
    let gl_Position = vec4<f32>((2.0 * VertexCoord / ScreenSize.xy - 1.0) * vec2(1.0, -1.0), 0.0, 1.0);


    return VertexOutput(UV, BgIndex, BgColor, gl_Position);
}

struct FragmentOutput {
    @location(0) FragColor: vec4<f32>,
}

fn unpack_color(color: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(color >> 24u) / 255.0,
        f32((color >> 16u) & 0xFFu) / 255.0,
        f32((color >> 8u) & 0xFFu) / 255.0,
        f32(color & 0xFFu) / 255.0,
    );
}


@fragment
fn fs_main(
    @location(0) UV: vec2<f32>,
    @location(1) @interpolate(flat) BgIndex: u32,
    @location(2) @interpolate(flat) BgColor: u32
) -> FragmentOutput {

    let mask = textureSample(Mask, Sampler, UV / AtlasSize.xy);

    var fragmentColor = BgBuffer[BgIndex];
    if mask.r >= 0.8 {
        // left
        if BgIndex > 0u {
            let idx = BgIndex - 1u;
            fragmentColor = BgBuffer[idx];
        }
    } else if mask.r >= 0.6 {
        // below
        var row = BgIndex / BgSize[0];
        let col = BgIndex % BgSize[0];
        if row + 1u < BgSize[1] {
            row = row + 1u;
        }
        let idx = row * BgSize[0] + col;
        fragmentColor = BgBuffer[idx];
    } else if mask.r >= 0.4 {
        // right
        if BgIndex < BgSize[0] {
            let idx = BgIndex + 1u;
            fragmentColor = BgBuffer[idx];
        }
    } else if mask.r >= 0.2 {
        // above
        var row = BgIndex / BgSize[0];
        let col = BgIndex % BgSize[0];
        if row > 0u {
            row = row - 1u;
        }
        let idx = row * BgSize[0] + col;
        fragmentColor = BgBuffer[idx];
    } // else exact

    let fragmentColorUnpacked = unpack_color(fragmentColor);

    return FragmentOutput(fragmentColorUnpacked);
}