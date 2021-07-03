struct VertexOutput {
    [[location(0)]] uv: vec2<f32>;
    [[builtin(position)]] pos: vec4<f32>;
};

[[stage(vertex)]]
fn vs_main([[builtin(vertex_index)]] in_vertex_index: u32) -> VertexOutput {
    let x = f32((in_vertex_index & 1u) << 2u);
    let y = f32((in_vertex_index & 2u) << 1u);
    var out: VertexOutput;
    out.uv = vec2<f32>(x * 0.5, 1.0 - (y * 0.5));
    out.pos = vec4<f32>(x - 1.0, y - 1.0, 0.0, 1.0);
    return out;
}

[[group(0), binding(0)]]
var texture: [[access(read)]] texture_storage_2d<rgba8unorm>;

fn get_avgd_col(uv: vec2<f32>) -> vec3<f32> {
    let dimensions = textureDimensions(texture);
    let pos = vec2<f32>(dimensions) * uv;
    let weights = pos - floor(pos);
    let ll = textureLoad(texture, clamp(vec2<i32>(pos) + vec2<i32>(0, 0), vec2<i32>(0, 0), dimensions)).rgb;
    let lh = textureLoad(texture, clamp(vec2<i32>(pos) + vec2<i32>(0, 1), vec2<i32>(0, 0), dimensions)).rgb;
    let hl = textureLoad(texture, clamp(vec2<i32>(pos) + vec2<i32>(1, 0), vec2<i32>(0, 0), dimensions)).rgb;
    let hh = textureLoad(texture, clamp(vec2<i32>(pos) + vec2<i32>(1, 1), vec2<i32>(0, 0), dimensions)).rgb;
    let lc = ll * (1.0 - weights.y) + lh * weights.y;
    let hc = hl * (1.0 - weights.y) + hh * weights.y;
    let cc = lc * (1.0 - weights.x) + hc * weights.x;
    return cc;
}

fn get_direct_col(uv: vec2<f32>) -> vec3<f32> {
    let dimensions = textureDimensions(texture);
    let pos = vec2<f32>(dimensions) * uv;
    return textureLoad(texture, vec2<i32>(pos)).rgb;
}

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let r = get_avgd_col(in.uv);
    // let r = get_direct_col(in.uv);
    return vec4<f32>(r, 1.0);
}