fn hash(state: u32) -> u32 {
    var s: u32 = state;
    s = s ^ 2747636419u;
    s = s * 2654435769u;
    s = s ^ (s >> 16u);
    s = s * 2654435769u;
    s = s ^ (s >> 16u);
    s = s * 2654435769u;
    return s;
}

fn scaleToRange01(state: u32) -> f32 {
    return f32(state) / 4294967295.0;
}

struct Agent {
    position: vec2<f32>;
    angle: f32;
    species: i32;
};

struct Settings {
    trail_weight: f32;
    move_speed: f32;
    turn_speed: f32;
    sensor_angle_degrees: f32;
    sensor_offset: f32;
    sensor_size: i32;
};

[[block]]
struct GlobalSettings {
    decay_rate: f32;
    diffuse_rate: f32;
};

[[block]]
struct AgentBuffer {
    agents: array<Agent>;
};

[[block]]
struct AgentSettingsBuffer {
    settings: array<Settings>;
};

[[block]]
struct Time {
    total: f32;
    delta: f32;
};

[[group(1), binding(0)]]
var<uniform> time: Time;


[[group(0), binding(0)]]
var<storage> m_agents: [[access(read_write)]] AgentBuffer;
[[group(0), binding(1)]]
var<storage> m_agent_settings: [[access(read)]] AgentSettingsBuffer;
[[group(0), binding(2)]]
var m_texture_r: [[access(read)]] texture_storage_2d_array<rgba16float>;
[[group(0), binding(3)]]
var m_texture_w: [[access(write)]] texture_storage_2d_array<r32float>;

fn sense(agent: Agent, sensor_angle_offset: f32) -> f32 {
    let settings = m_agent_settings.settings[agent.species];

    let sensor_angle = agent.angle + sensor_angle_offset;
    let sensor_dir = vec2<f32>(cos(sensor_angle), sin(sensor_angle));

    // let sensor_pos = agent.position + sensor_dir * settings.sensor_offset;
    let sensor_pos = agent.position + sensor_dir * 25.0;
    let sensor_center = vec2<i32>(sensor_pos);

    var sum: vec4<f32> = vec4<f32>(0.0);

    let dim = vec2<i32>(textureDimensions(m_texture_r));
    let sensor_size = settings.sensor_size;

    let species_count = i32(arrayLength(&m_agent_settings.settings));

    for (var species: i32 = 0; species < species_count; species = species + 4) {
        let bool_mask = vec4<bool>(
            species + 0 == agent.species,
            species + 1 == agent.species,
            species + 2 == agent.species,
            species + 3 == agent.species
        );
        let int_mask = vec4<i32>(bool_mask);
        let mask = vec4<f32>(int_mask) * 2.0 - vec4<f32>(1.0);
        for (var offset_x: i32 = -sensor_size; offset_x <= sensor_size; offset_x = offset_x + 1) {
            for (var offset_y: i32 = -sensor_size; offset_y <= sensor_size; offset_y = offset_y + 1) {
                let offset = vec2<i32>(offset_x, offset_y);
                let sample = clamp(sensor_center + offset, vec2<i32>(0), dim - vec2<i32>(1));
                sum = sum + mask * textureLoad(m_texture_r, sample, species / 4);
            }
        }
    }

    return sum.x + sum.y + sum.z + sum.w;
}

[[stage(compute), workgroup_size(32)]]
fn update(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
    let id = global_id.x;
    let agent_count = arrayLength(&m_agents.agents);

    if (id >= agent_count) {
        return;
    }

    let dim = vec2<u32>(textureDimensions(m_texture_r));

    let agent = m_agents.agents[id];
    let settings = m_agent_settings.settings[agent.species];
    let pos = agent.position;

    var random: u32 = hash(u32(pos.y) * dim.x + u32(pos.x) + hash(id + u32(time.total * 100000.0)));

    let sensor_angle_rad = settings.sensor_angle_degrees * (3.1415 / 180.0);
    let weight_forward = sense(agent, 0.0);
    let weight_left = sense(agent, sensor_angle_rad);
    let weight_right = sense(agent, -sensor_angle_rad);

    let random_steer_strength = scaleToRange01(random);

    // let turn_amount = settings.turn_speed * time.delta;
    let turn_amount = 15.0 * time.delta;

    // Continue in same direction
    if (weight_forward > weight_left && weight_forward > weight_right) {
        m_agents.agents[id].angle = agent.angle + 0.0;
    }
    elseif (weight_forward < weight_left && weight_forward < weight_right) {
        m_agents.agents[id].angle = agent.angle + (random_steer_strength - 0.5) * 2.0 * turn_amount;
    }
    // Turn right
    elseif (weight_right > weight_left) {
        m_agents.agents[id].angle = agent.angle - random_steer_strength * turn_amount;
    }
    // Turn left
    elseif (weight_left > weight_right) {
        m_agents.agents[id].angle = agent.angle + random_steer_strength * turn_amount;
    }

    let dist = time.delta * settings.move_speed;
    let dir = vec2<f32>(cos(agent.angle), sin(agent.angle));
    var new_pos: vec2<f32> = agent.position + dist * dir;

    let dimf32 = vec2<f32>(dim);
    // Clamp position to map boundaries, and pick new random move dir if hit boundary
    if (new_pos.x < 0.0 || new_pos.x >= dimf32.x || new_pos.y < 0.0 || new_pos.y >= dimf32.y) {
        random = hash(random);
        let random_angle = scaleToRange01(random) * 2.0 * 3.1415;

        new_pos = clamp(new_pos, vec2<f32>(0.0), dimf32);
        m_agents.agents[id].angle = random_angle;
    }
    else {
        textureStore(m_texture_w, vec2<i32>(new_pos), agent.species, vec4<f32>(settings.trail_weight * time.delta));
    }

    m_agents.agents[id].position = new_pos;
}

[[group(0), binding(0)]]
var<uniform> b_settings: GlobalSettings;

[[group(0), binding(1)]]
var b_texture_r: [[access(read)]] texture_storage_2d_array<rgba16float>;
[[group(0), binding(2)]]
var b_texture_painted: [[access(read)]] texture_storage_2d_array<r32float>;
[[group(0), binding(3)]]
var b_texture_w: [[access(write)]] texture_storage_2d_array<rgba16float>;

fn fetch_color(coords: vec2<i32>, index: i32) -> vec4<f32> {
    let species_count = i32(textureNumLayers(b_texture_painted));
    var sum: vec4<f32> = textureLoad(b_texture_r, coords, index);
    let species = index * 4;
    sum = sum + vec4<f32>(textureLoad(b_texture_painted, coords, species + 0).r, 0.0, 0.0, 0.0);
    if (species + 1 >= species_count) { return min(vec4<f32>(1.0), sum); }
    sum = sum + vec4<f32>(0.0, textureLoad(b_texture_painted, coords, species + 1).r, 0.0, 0.0);
    if (species + 2 >= species_count) { return min(vec4<f32>(1.0), sum); }
    sum = sum + vec4<f32>(0.0, 0.0, textureLoad(b_texture_painted, coords, species + 2).r, 0.0);
    if (species + 3 >= species_count) { return min(vec4<f32>(1.0), sum); }
    sum = sum + vec4<f32>(0.0, 0.0, 0.0, textureLoad(b_texture_painted, coords, species + 3).r);
    return min(vec4<f32>(1.0), sum);
}

[[stage(compute), workgroup_size(32, 32)]]
fn blur(
    [[builtin(global_invocation_id)]] id: vec3<u32>,
) {
    let decay_rate = b_settings.decay_rate;
    let diffuse_rate = b_settings.diffuse_rate;
    let species_group_id = i32(id.z);

    let dimensions = vec2<u32>(textureDimensions(b_texture_w));
    if (id.x < 0u || id.x >= dimensions.x || id.y < 0u || id.y >= dimensions.y) {
        return;
    }
    let coords = vec2<i32>(id.xy);
    let dim = vec2<i32>(dimensions);

    var sum: vec4<f32> = vec4<f32>(0.0);
    for (var offset_x: i32 = -1; offset_x <= 1; offset_x = offset_x + 1) {
        for (var offset_y: i32 = -1; offset_y <= 1; offset_y = offset_y + 1) {
            let offset = vec2<i32>(offset_x, offset_y);
            let sample = clamp(coords + offset, vec2<i32>(0), dim - vec2<i32>(1));
            sum = sum + fetch_color(sample, species_group_id);
        }
    }

    let mean = sum / 9.0;
    let diffuse_weight = clamp(diffuse_rate * time.delta, 0.0, 1.0);

    let original_color = fetch_color(coords, species_group_id);
    let blurred_color = original_color * (1.0 - diffuse_weight) + mean * diffuse_weight;

    let out = max(vec4<f32>(0.0), blurred_color - decay_rate * time.delta);
    textureStore(b_texture_w, coords, species_group_id, out);
}
