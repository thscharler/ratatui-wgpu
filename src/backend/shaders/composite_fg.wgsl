struct VertexOutput {
    @location(0) UV: vec2<f32>,
    @location(1) @interpolate(flat) UVx0: f32,
    @location(2) @interpolate(flat) FgColor: u32,
    @location(3) @interpolate(flat) ColorGlyph: u32,
    @location(4) @interpolate(flat) UnderlinePos: u32,
    @location(5) @interpolate(flat) StrikeoutPos: u32,
    @location(6) @interpolate(flat) CursorPos: u32,
    @location(7) @interpolate(flat) CursorColor: u32,
    @builtin(position) gl_Position: vec4<f32>,
}

@group(0) @binding(0)
var<uniform> ScreenSize: vec4<f32>;

@vertex
fn vs_main(
    @location(0) VertexCoord: vec2<f32>,
    @location(1) UV: vec2<f32>,
    @location(2) UVx0: f32,
    @location(3) FgColor: u32,
    @location(4) ColorGlyph: u32,
    @location(5) UnderlinePos: u32,
    @location(6) StrikeoutPos: u32,
    @location(7) CursorPos: u32,
    @location(8) CursorColor: u32,
) -> VertexOutput {
    let gl_Position = vec4<f32>((2.0 * VertexCoord / ScreenSize.xy - 1.0) * vec2(1.0, -1.0), 0.0, 1.0);

    return VertexOutput(UV,
        UVx0,
        FgColor,
        ColorGlyph,
        UnderlinePos,
        StrikeoutPos,
        CursorPos,
        CursorColor,
        gl_Position);
}

struct FragmentOutput {
    @location(0) FragColor: vec4<f32>,
}

@group(1) @binding(0) 
var Atlas: texture_2d<f32>;
@group(1) @binding(1)
var Sampler: sampler;
@group(1) @binding(2)
var<uniform> AtlasSize: vec4<f32>;

@fragment
fn fs_main(
    @location(0) UV: vec2<f32>,
    @location(1) @interpolate(flat) UVx0: f32,
    @location(2) @interpolate(flat) FgColor: u32,
    @location(3) @interpolate(flat) ColorGlyph: u32,
    @location(4) @interpolate(flat) UnderlinePos: u32,
    @location(5) @interpolate(flat) StrikeoutPos: u32,
    @location(6) @interpolate(flat) CursorPos: u32,
    @location(7) @interpolate(flat) CursorColor: u32,
) -> FragmentOutput {
    var cursorColorUnpacked = unpack4x8unorm(CursorColor);
    var fgColorUnpacked = unpack4x8unorm(FgColor);
    var textureColor = textureSample(Atlas, Sampler, UV / AtlasSize.xy);

    var fgcolorAlpha = fgColorUnpacked;
    let alpha = textureColor.a * fgcolorAlpha.a;
    textureColor.a = alpha;
    fgcolorAlpha.a = alpha;
    var fragmentColor = select(fgcolorAlpha, textureColor, ColorGlyph == 1);

    let yMax = UnderlinePos & 0xFFFFu;
    let yMin = UnderlinePos >> 16u;
    fragmentColor = select(fragmentColor, fgColorUnpacked, u32(UV.y) >= yMin && u32(UV.y) < yMax);

    let y2Max = StrikeoutPos & 0xFFFFu;
    let y2Min = StrikeoutPos >> 16u;
    fragmentColor = select(fragmentColor, fgColorUnpacked, u32(UV.y) >= y2Min && u32(UV.y) < y2Max);

    let cur_vis = CursorPos & 0x00020000u;
    let cur_hor = CursorPos & 0x00010000u;
    let cur_min = CursorPos & 0xFFu;
    let cur_max = (CursorPos >> 8u) & 0xFFu;
    if cur_vis != 0 {
        var is_cur = true;
        if cur_hor != 0 {
            is_cur = u32(UV.y) >= cur_min && u32(UV.y) < cur_max;
        } else {
            // uv points to the atlas offset, cur_min/cur_max are relative to the texture.
            is_cur = u32(UV.x-UVx0) >= cur_min && u32(UV.x-UVx0) < cur_max;
        }
        if is_cur {
            if fragmentColor.a > 0.0 {
                let fg_a = fragmentColor.a * (1.0 - cursorColorUnpacked.a);
                let cur_a = (1.0 - fg_a);

                if any(fragmentColor.rgb != cursorColorUnpacked.rgb) {
                    fragmentColor.a = 1.0;
                } else {
                    fragmentColor.a = 1.0 - fragmentColor.a;
                }
                fragmentColor.r =  fragmentColor.r * fg_a + cursorColorUnpacked.r * cur_a;
                fragmentColor.g =  fragmentColor.g * fg_a + cursorColorUnpacked.g * cur_a;
                fragmentColor.b =  fragmentColor.b * fg_a + cursorColorUnpacked.b * cur_a;
            } else {
                fragmentColor = cursorColorUnpacked;
                fragmentColor.a = 1.0;
            }
        }
    }

    return FragmentOutput(fragmentColor);
}