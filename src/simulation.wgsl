fn hash(state: u32) -> u32 {
    let state = state ^ 2747636419u;
    let state = state * 2654435769u;
    let state = state ^ (state >> 16u);
    let state = state * 2654435769u;
    let state = state ^ (state >> 16u);
    let state = state * 2654435769u;
    return state;
}

fn scaleToRange01(state: u32) -> f32 {
    return f32(state) / 4294967295.0;
}

struct Agent {
    position: vec2<f32>;
    angle: f32;
	_pad: f32;
};

[[block]]
struct AgentBuffer {
    agents: array<Agent>;
};

[[block]]
struct Metadata {
    agent_count: u32;
};

[[group(0), binding(0)]]
var<storage> m_agents: [[access(read_write)]] AgentBuffer;
[[group(0), binding(1)]]
var m_texture: [[access(read)]] texture_storage_2d<r32float>;

[[group(1), binding(0)]]
var<uniform> metadata: Metadata;

[[block]]
struct Time {
    total: f32;
    delta: f32;
};

[[group(2), binding(0)]]
var<uniform> m_time: Time;

let sensor_offset_dir: f32 = 35.0;
let sensor_size: i32 = 1;
let sensor_angle_degrees: f32 = 30.0;
let turn_speed: f32 = 2.0;
let move_speed: f32 = 20.0;
let trail_weight: f32 = 5.0;

fn sense(agent: Agent, sensor_angle_offset: f32) -> f32 {
	let sensorAngle = agent.angle + sensor_angle_offset;
	let sensorDir = vec2<f32>(cos(sensorAngle), sin(sensorAngle));

	let sensorPos = agent.position + sensorDir * sensor_offset_dir;
	let sensorCentreX = i32(sensorPos.x);
	let sensorCentreY = i32(sensorPos.y);

	var sum: f32 = 0.0;

    let dim = vec2<i32>(textureDimensions(m_texture));
	for (var offsetX: i32 = -sensor_size; offsetX <= sensor_size; offsetX = offsetX + 1) {
		for (var offsetY: i32 = -sensor_size; offsetY <= sensor_size; offsetY = offsetY + 1) {
			let sampleX = min(dim.x - 1, max(0, sensorCentreX + offsetX));
			let sampleY = min(dim.y - 1, max(0, sensorCentreY + offsetY));
			sum = sum + textureLoad(m_texture, vec2<i32>(sampleX, sampleY)).x;
		}
	}

	return sum;
}

[[stage(compute), workgroup_size(32)]]
fn move_agents(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
    let id = global_id.x;

    if (id >= metadata.agent_count) {
        return;
    }

    let dim = vec2<u32>(textureDimensions(m_texture));

    let this = m_agents.agents[id];
    let pos = this.position;

    let random = hash(u32(pos.y) * dim.x + u32(pos.x) + hash(id + u32(m_time.total * 100000.0)));

    let sensorAngleRad = sensor_angle_degrees * (3.1415 / 180.0);
	let weightForward = sense(this, 0.0);
	let weightLeft = sense(this, sensorAngleRad);
	let weightRight = sense(this, -sensorAngleRad);

    let randomSteerStrength = scaleToRange01(random);
	let turnSpeed = turn_speed * 2.0 * 3.1415;

    // Continue in same direction
	if (weightForward > weightLeft && weightForward > weightRight) {
		m_agents.agents[id].angle = this.angle + 0.0;
	}
	elseif (weightForward < weightLeft && weightForward < weightRight) {
		m_agents.agents[id].angle = this.angle + (randomSteerStrength - 0.5) * 2.0 * turn_speed * m_time.delta;
	}
	// Turn right
	elseif (weightRight > weightLeft) {
		m_agents.agents[id].angle = this.angle - randomSteerStrength * turn_speed * m_time.delta;
	}
	// Turn left
	elseif (weightLeft > weightRight) {
	    m_agents.agents[id].angle = this.angle + randomSteerStrength * turn_speed * m_time.delta;
	}

    let delta = vec2<f32>(sin(this.angle), cos(this.angle)) * m_time.delta * move_speed;
    var new_pos: vec2<f32> = this.position + delta;

    let dim = vec2<f32>(dim);
    // Clamp position to map boundaries, and pick new random move dir if hit boundary
	if (new_pos.x < 0.0 || new_pos.x >= dim.x || new_pos.y < 0.0 || new_pos.y >= dim.y) {
		let random = hash(random);
		let randomAngle = scaleToRange01(random) * 2.0 * 3.1415;

		new_pos.x = min(dim.x - 1.0, max(0.0, new_pos.x));
		new_pos.y = min(dim.y - 1.0, max(0.0, new_pos.y));
		m_agents.agents[id].angle = randomAngle;
	}

    m_agents.agents[id].position = new_pos;
}

[[group(0), binding(0)]]
var<storage> p_agents: [[access(read)]] AgentBuffer;
[[group(0), binding(1)]]
var p_texture: [[access(write)]] texture_storage_2d<r32float>;
[[group(2), binding(0)]]
var<uniform> p_time: Time;

[[stage(compute), workgroup_size(32)]]
fn paint_agents(
    [[builtin(global_invocation_id)]] global_id: vec3<u32>,
) {
    let id = global_id.x;
    if (id >= metadata.agent_count) {
        return;
    }

    let dimensions = vec2<f32>(textureDimensions(p_texture));

    let this = m_agents.agents[id];
    let pos = this.position;
    if (pos.x < 0.0 || pos.x > dimensions.x || pos.y < 0.0 || pos.y > dimensions.y) {
        return;
    }
	let pos = vec2<i32>(this.position);
    textureStore(p_texture, pos, vec4<f32>(trail_weight * p_time.delta));
}

let diffuseRate: f32 = 3.0;
let decayRate: f32 = 0.2;

[[group(0), binding(0)]]
var b_texture_r: [[access(read)]] texture_storage_2d<r32float>;
[[group(0), binding(1)]]
var b_texture_painted: [[access(read)]] texture_storage_2d<r32float>;
[[group(0), binding(2)]]
var b_texture_w: [[access(write)]] texture_storage_2d<r32float>;
[[group(1), binding(0)]]
var<uniform> b_time: Time;
[[stage(compute), workgroup_size(32, 32)]]
fn blur(
    [[builtin(global_invocation_id)]] id: vec3<u32>,
) {
    let dimensions = vec2<u32>(textureDimensions(b_texture_w));
    if (id.x < 0u || id.x >= dimensions.x || id.y < 0u || id.y >= dimensions.y) {
		return;
	}
    let coords = vec2<i32>(id.xy);
    let dim = vec2<i32>(dimensions);

	var sum: vec4<f32> = vec4<f32>(0.0);
	for (var offsetX: i32 = -1; offsetX <= 1; offsetX = offsetX + 1) {
		for (var offsetY: i32 = -1; offsetY <= 1; offsetY = offsetY + 1) {
			let sampleX: i32 = min(dim.x - 1, max(0, coords.x + offsetX));
			let sampleY: i32 = min(dim.y - 1, max(0, coords.y + offsetY));
			let sample = vec2<i32>(sampleX, sampleY);
			sum = sum + textureLoad(b_texture_r, sample) + textureLoad(b_texture_painted, sample);
		}
	}

	let blurredCol = sum / 9.0;
	let diffuseWeight = clamp(diffuseRate * b_time.delta, 0.0, 1.0);

	let originalCol = textureLoad(b_texture_r, coords).r + textureLoad(b_texture_painted, coords).r;
	let blurredCol = originalCol * (1.0 - diffuseWeight) + blurredCol * (diffuseWeight);

    let out = max(vec4<f32>(0.0), blurredCol - decayRate * b_time.delta);
    textureStore(b_texture_w, coords, vec4<f32>(out));
    // textureStore(b_texture_w, coords, vec4<f32>(scaleToRange01(hash(id.x + id.y * dimensions.x + u32(b_time.total) * 100000000u))));
}
