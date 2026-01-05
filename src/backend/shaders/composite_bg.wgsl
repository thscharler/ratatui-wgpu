struct VertexOutput {
    @location(0) UV: vec2<f32>,
    @location(1) @interpolate(flat) BgIndex: u32,
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
) -> VertexOutput {
    let gl_Position = vec4<f32>((2.0 * VertexCoord / ScreenSize.xy - 1.0) * vec2(1.0, -1.0), 0.0, 1.0);

    return VertexOutput(UV, BgIndex, gl_Position);
}

struct FragmentOutput {
    @location(0) FragColor: vec4<f32>,
}

@fragment
fn fs_main(
    @location(0) UV: vec2<f32>,
    @location(1) @interpolate(flat) BgIndex: u32,
) -> FragmentOutput {

    let mask = textureSample(Mask, Sampler, UV / AtlasSize.xy);

    var fragmentColor = BgBuffer[BgIndex];
    if mask.r >= 0.5 {
        // left
        var row = BgIndex / BgSize[0];
        var col = BgIndex % BgSize[0];
        if col > 0u {
            col = col - 1;
        }
        let idx = row * BgSize[0] + col;
        fragmentColor = BgBuffer[idx];

    } else if mask.r >= 0.43 {
        // bottom left
        var row = BgIndex / BgSize[0];
        var col = BgIndex % BgSize[0];
        if row < BgSize[1] {
            row = row + 1;
        }
        if col > 0 {
            col = col - 1;
        }
        let idx = row * BgSize[0] + col;
        fragmentColor = BgBuffer[idx];

    } else if mask.r >= 0.37 {
        // below
        var row = BgIndex / BgSize[0];
        let col = BgIndex % BgSize[0];
        if row + 1 < BgSize[1] {
            row = row + 1;
        }
        let idx = row * BgSize[0] + col;
        fragmentColor = BgBuffer[idx];

    } else if mask.r >= 0.31 {
        // bottom right
        var row = BgIndex / BgSize[0];
        var col = BgIndex % BgSize[0];
        if row < BgSize[1] {
            row = row + 1;
        }
        if col < BgSize[0] {
            col = col + 1;
        }
        let idx = row * BgSize[0] + col;
        fragmentColor = BgBuffer[idx];

    } else if mask.r >= 0.24 {
        // right
        var row = BgIndex / BgSize[0];
        var col = BgIndex % BgSize[0];
        if col < BgSize[0] {
            col = col + 1;
        }
        let idx = row * BgSize[0] + col;
        fragmentColor = BgBuffer[idx];

    } else if mask.r >= 0.18 {
        // top right
        var row = BgIndex / BgSize[0];
        var col = BgIndex % BgSize[0];
        if row > 0u {
            row = row - 1u;
        }
        if col < BgSize[0] {
            col = col + 1u;
        }
        let idx = row * BgSize[0] + col;
        fragmentColor = BgBuffer[idx];

    } else if mask.r >= 0.12 {
        // above
        var row = BgIndex / BgSize[0];
        let col = BgIndex % BgSize[0];
        if row > 0u {
            row = row - 1u;
        }
        let idx = row * BgSize[0] + col;
        fragmentColor = BgBuffer[idx];

    } else if mask.r >= 0.06 {
        // top left
        var row = BgIndex / BgSize[0];
        var col = BgIndex % BgSize[0];
        if row > 0u {
            row = row - 1u;
        }
        if col > 0u {
            col = col - 1u;
        }
        let idx = row * BgSize[0] + col;
        fragmentColor = BgBuffer[idx];

    } // else exact

    let fragmentColorUnpacked = unpack4x8unorm(fragmentColor);

    return FragmentOutput(fragmentColorUnpacked);
}