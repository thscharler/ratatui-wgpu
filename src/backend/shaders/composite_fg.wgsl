struct VertexOutput {
    @location(0) UV: vec2<f32>,
    @location(1) @interpolate(flat) UVx0: f32,
    @location(2) @interpolate(flat) FgColor: u32,
    @location(3) @interpolate(flat) UnderlinePos: u32,
    @location(4) @interpolate(flat) UnderlineColor: u32,
    @location(5) @interpolate(flat) StrikeoutPos: u32,
    @location(6) @interpolate(flat) StrikeoutColor: u32,
    @location(7) @interpolate(flat) CursorPos: u32,
    @location(8) @interpolate(flat) CursorColor: u32,
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
    @location(4) UnderlinePos: u32,
    @location(5) UnderlineColor: u32,
    @location(6) StrikeoutPos: u32,
    @location(7) StrikeoutColor: u32,
    @location(8) CursorPos: u32,
    @location(9) CursorColor: u32,
) -> VertexOutput {
    let gl_Position = vec4<f32>((2.0 * VertexCoord / ScreenSize.xy - 1.0) * vec2(1.0, -1.0), 0.0, 1.0);
    return VertexOutput(UV,
        UVx0,
        FgColor,
        UnderlinePos,
        UnderlineColor,
        StrikeoutPos,
        StrikeoutColor,
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
var Mask: texture_2d<f32>;
@group(1) @binding(2) 
var Sampler: sampler;

@group(1) @binding(3) 
var<uniform> AtlasSize: vec4<f32>;

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
    @location(1) @interpolate(flat) UVx0: f32,
    @location(2) @interpolate(flat) FgColor: u32,
    @location(3) @interpolate(flat) UnderlinePos: u32,
    @location(4) @interpolate(flat) UnderlineColor: u32,
    @location(5) @interpolate(flat) StrikeoutPos: u32,
    @location(6) @interpolate(flat) StrikeoutColor: u32,
    @location(7) @interpolate(flat) CursorPos: u32,
    @location(8) @interpolate(flat) CursorColor: u32,
) -> FragmentOutput {
    let underLineColorUnpacked = unpack_color(UnderlineColor);
    let strikeOutColorUnpacked = unpack_color(StrikeoutColor);
    var cursorColorUnpacked = unpack_color(CursorColor);

    var fgColorUnpacked = unpack_color(FgColor);
    var textureColor = textureSample(Atlas, Sampler, UV / AtlasSize.xy);

    let alpha = textureColor.a * fgColorUnpacked.a;
    textureColor.a = alpha;
    fgColorUnpacked.a = alpha;

    let mask = textureSample(Mask, Sampler, UV / AtlasSize.xy);

    var fgColor = select(fgColorUnpacked, textureColor, mask.r == 1.0);

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
             if fgColor.a > 0.0 {
                 fgColor.a = 1.0;
                 fgColor.r =  fgColor.r * fgColor.a + cursorColorUnpacked.r;
                 fgColor.g =  fgColor.g * fgColor.a + cursorColorUnpacked.g;
                 fgColor.b =  fgColor.b * fgColor.a + cursorColorUnpacked.b;
             } else {
                fgColor = cursorColorUnpacked;
             }
        }
    }

    let yMax = UnderlinePos & 0xFFFFu;
    let yMin = UnderlinePos >> 16u;
    fgColor = select(fgColor, underLineColorUnpacked, u32(UV.y) >= yMin && u32(UV.y) < yMax);

    let y2Max = StrikeoutPos & 0xFFFFu;
    let y2Min = StrikeoutPos >> 16u;
    fgColor = select(fgColor, strikeOutColorUnpacked, u32(UV.y) >= y2Min && u32(UV.y) < y2Max);

    return FragmentOutput(fgColor);
}