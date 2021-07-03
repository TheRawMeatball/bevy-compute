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

struct Settings {
    color: vec3<f32>;
    weight: f32;
};

[[block]]
struct SettingsBuffer {
    settings: array<Settings>;
};

[[group(0), binding(0)]]
var texture: [[access(read)]] texture_storage_2d_array<rgba16float>;
[[group(0), binding(1)]]
var<storage> m_disp_settings: [[access(read)]] SettingsBuffer;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    let dimensions = textureDimensions(texture);
    let pos = vec2<f32>(dimensions) * in.uv;
    let species_count = i32(arrayLength(&m_disp_settings.settings));
    var col: vec3<f32> = vec3<f32>(0.0);
    for(var i: i32 = 0; i < species_count / 4; i = i + 1) {
        let vals = textureLoad(texture, vec2<i32>(pos), i);
        col = col + m_disp_settings.settings[i * species_count + 0].color * vals.x;
        col = col + m_disp_settings.settings[i * species_count + 1].color * vals.y;
        col = col + m_disp_settings.settings[i * species_count + 2].color * vals.z;
        col = col + m_disp_settings.settings[i * species_count + 3].color * vals.w;
    }
    let completed = (species_count / 4) * 4;
    if (species_count % 4 != 0) {
        let vals = textureLoad(texture, vec2<i32>(pos), species_count / 4);
        if (completed + 2 < species_count) {
            col = col + m_disp_settings.settings[completed + 0].color * vals.x;
            col = col + m_disp_settings.settings[completed + 1].color * vals.y;
            col = col + m_disp_settings.settings[completed + 2].color * vals.z;
        }
        elseif (completed + 1 < species_count) {
            col = col + m_disp_settings.settings[completed + 0].color * vals.x;
            col = col + m_disp_settings.settings[completed + 1].color * vals.y;
        }
        elseif (completed < species_count) {
            col = col + m_disp_settings.settings[completed + 0].color * vals.x;
        }
    }
    return vec4<f32>(vec3<f32>(col), 1.0);
}