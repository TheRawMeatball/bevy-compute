use std::{
    num::{NonZeroU32, NonZeroU64},
    sync::Mutex,
};

use bevy::{
    math::{const_vec3, f32},
    prelude::*,
    render2::{
        core_pipeline,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext},
        render_resource::{
            BindGroup, BindGroupLayoutEntry, BindingType, BlendComponent, BlendFactor,
            BlendOperation, BlendState, Buffer, BufferBindingType, BufferSize, ColorTargetState,
            ColorWrite, ComputePipeline, RenderPipeline, ShaderStage, Texture, TextureFormat,
            TextureViewDimension, VertexState,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        shader::Shader,
        texture::BevyDefault,
        view::ExtractedWindows,
        RenderStage,
    },
    window::{WindowId, WindowMode},
    PipelinedDefaultPlugins,
};
use rand::Rng;
use wgpu::{
    util::BufferInitDescriptor, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindingResource, BufferBinding, BufferDescriptor, BufferUsage, ComputePassDescriptor, Extent3d,
    Face, FragmentState, FrontFace, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout, LoadOp,
    MapMode, MultisampleState, Operations, Origin3d, PolygonMode, PrimitiveState,
    PrimitiveTopology, RenderPassDescriptor, TextureDescriptor, TextureUsage,
    TextureViewDescriptor,
};

#[derive(Default)]
struct Fullscreen(bool);

#[bevy_main]
pub fn main() {
    let mut app = App::new();
    app.insert_resource(WindowDescriptor {
        width: 1920.,
        height: 1080.,
        ..Default::default()
    })
    .add_plugins(PipelinedDefaultPlugins)
    .init_resource::<Fullscreen>();

    let render_app = app.sub_app_mut(0);
    render_app.add_system_to_stage(RenderStage::Extract, time_extract_system.system());
    render_app.init_resource::<MoldShaders>();
    let mut graph = render_app.world.get_resource_mut::<RenderGraph>().unwrap();
    graph.add_node(
        "mold",
        MoldNode {
            inner: Mutex::new(MoldNodeInner {
                time: 0.,
                state: ReadState::A,
            }),
        },
    );
    graph
        .add_node_edge("mold", core_pipeline::node::MAIN_PASS_DEPENDENCIES)
        .unwrap();

    app.add_startup_system(setup_system.system())
        .add_system(fullscreen_system.system());

    app.run();
}

fn fullscreen_system(
    mut fs: ResMut<Fullscreen>,
    inp: Res<Input<KeyCode>>,
    mut windows: ResMut<Windows>,
) {
    if inp.just_pressed(KeyCode::F11) {
        let primary = windows.get_primary_mut().unwrap();
        fs.0 = !fs.0;
        primary.set_mode(if fs.0 {
            WindowMode::Fullscreen { use_size: false }
        } else {
            WindowMode::Windowed
        });
    }
}

fn setup_system(mut commands: Commands) {
    commands.spawn_bundle(bevy::render2::camera::PerspectiveCameraBundle::default());
    // commands.spawn_bundle(bevy::render2::camera::OrthographicCameraBundle::new_2d());
}
#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy)]
struct PlainTime {
    total: f32,
    delta: f32,
}

fn time_extract_system(time: Res<Time>, mut commands: Commands) {
    commands.insert_resource(PlainTime {
        total: time.time_since_startup().as_secs_f32(),
        delta: time.delta_seconds(),
    });
}

const AGENT_COUNT: u32 = 200_000;
const TEX_WIDTH: u32 = 1280;
const TEX_HEIGHT: u32 = 720;
const SPECIES: &[Settings] = &[Settings {
    trail_weight: 5.0,
    move_speed: 15.,
    turn_speed: 15.,
    sensor_angle_degrees: 30.,
    sensor_offset: 25.,
    sensor_size: 1,
}; 5];
const DISPLAY_SETTINGS: &[DisplaySettings] = &[
    DisplaySettings {
        color: const_vec3!([0.2, 0.8, 0.8]),
        weight: 1.0,
    },
    DisplaySettings {
        color: const_vec3!([0.2, 0.2, 0.9]),
        weight: 1.0,
    },
    DisplaySettings {
        color: const_vec3!([0.7, 0.2, 0.9]),
        weight: 1.0,
    },
    DisplaySettings {
        color: const_vec3!([0.1, 0.9, 0.2]),
        weight: 1.0,
    },
    DisplaySettings {
        color: const_vec3!([0.9, 0.5, 0.1]),
        weight: 1.0,
    },
];
const _ASSERT_SPECIES_ARRAY_MATCH: () = {
    let _ = [0][DISPLAY_SETTINGS.len() - SPECIES.len()];
};
const SPECIES_COUNT: u32 = SPECIES.len() as u32;
const GLOBAL_SETTINGS: &GlobalSettings = &GlobalSettings {
    decay_rate: 0.5,
    diffuse_rate: 4.0,
};

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy)]
struct Agent {
    position: Vec2,
    direction: f32,
    species: i32,
}

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy)]
struct Settings {
    trail_weight: f32,
    move_speed: f32,
    turn_speed: f32,
    sensor_angle_degrees: f32,
    sensor_offset: f32,
    sensor_size: i32,
}

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy)]
struct GlobalSettings {
    decay_rate: f32,
    diffuse_rate: f32,
}

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy)]
struct DisplaySettings {
    color: Vec3,
    weight: f32,
}

pub struct MoldShaders {
    update_pipeline: ComputePipeline,
    update_bg_a: BindGroup,
    update_bg_b: BindGroup,
    blur_pipeline: ComputePipeline,
    blur_bg_a: BindGroup,
    blur_bg_b: BindGroup,

    combine_pipeline: ComputePipeline,
    combine_bg_a: BindGroup,
    combine_bg_b: BindGroup,

    display_pipeline: RenderPipeline,
    display_bg: BindGroup,

    combine_texture: Texture,
    update_texture: Texture,

    time_buffer: Buffer,
    time_bg: BindGroup,

    read_buffer: Buffer,
}

#[allow(unused)]
impl Agent {
    fn gen_circle(rng: &mut impl Rng, radius: f32) -> Self {
        let radius = radius * f32::sqrt(rng.gen_range(0.0..1.0));
        let theta = rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI);
        let pos = Vec2::new(f32::cos(theta), f32::sin(theta)) * radius;
        let offset = Vec2::new(TEX_WIDTH as f32, TEX_HEIGHT as f32) / 2.;
        Agent {
            position: pos + offset,
            direction: f32::atan2(-pos.y, -pos.x),
            species: 0,
        }
    }

    fn gen_point(rng: &mut impl Rng) -> Self {
        let offset = Vec2::new(TEX_WIDTH as f32, TEX_HEIGHT as f32) / 2.;
        Agent {
            position: offset,
            direction: rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI),
            species: 0,
        }
    }
}

// TODO: this pattern for initializing the shaders / pipeline isn't ideal. this should be handled by the asset system
impl FromWorld for MoldShaders {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.get_resource::<RenderDevice>().unwrap();
        let shader = Shader::from_wgsl(include_str!("simulation.wgsl"));
        let shader_module = render_device.create_shader_module(&shader);

        let display_shader = Shader::from_wgsl(include_str!("display.wgsl"));
        let display_shader_module = render_device.create_shader_module(&display_shader);

        let mut rng = rand::thread_rng();
        let agents = (0..AGENT_COUNT)
            .map(|i| Agent {
                species: (i % SPECIES_COUNT) as i32,
                ..Agent::gen_circle(&mut rng, (u32::min(TEX_WIDTH, TEX_HEIGHT) / 2 - 20) as f32)
            })
            .collect::<Vec<_>>();
        let agent_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("mold_agents"),
            usage: BufferUsage::STORAGE,
            contents: bytemuck::cast_slice(&agents),
        });

        let settings_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("species_settings"),
            contents: bytemuck::cast_slice(SPECIES),
            usage: BufferUsage::STORAGE,
        });
        let combine_settings_buffer =
            render_device.create_buffer_with_data(&BufferInitDescriptor {
                label: Some("combine_species_settings"),
                contents: bytemuck::cast_slice(DISPLAY_SETTINGS),
                usage: BufferUsage::STORAGE,
            });
        let global_settings_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("global_settings"),
            contents: bytemuck::bytes_of(GLOBAL_SETTINGS),
            usage: BufferUsage::UNIFORM,
        });

        let texture_descriptor = TextureDescriptor {
            label: None,
            size: Extent3d {
                width: TEX_WIDTH,
                height: TEX_HEIGHT,
                depth_or_array_layers: div_ceil(SPECIES_COUNT, 4),
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba16Float,
            usage: TextureUsage::STORAGE,
        };
        let primary_texture_a = render_device.create_texture(&TextureDescriptor {
            label: Some("trail_map_a"),
            ..texture_descriptor
        });
        let primary_texture_b = render_device.create_texture(&TextureDescriptor {
            label: Some("trail_map_b"),
            ..texture_descriptor
        });
        let update_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("update_write_trail_map"),
            usage: TextureUsage::STORAGE | TextureUsage::COPY_DST,
            format: TextureFormat::R32Float,
            size: Extent3d {
                width: TEX_WIDTH,
                height: TEX_HEIGHT,
                depth_or_array_layers: SPECIES_COUNT,
            },
            ..texture_descriptor
        });
        let combine_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("combine_texture"),
            usage: TextureUsage::STORAGE | TextureUsage::COPY_SRC,
            format: TextureFormat::Rgba8Unorm,
            size: Extent3d {
                width: TEX_WIDTH,
                height: TEX_HEIGHT,
                depth_or_array_layers: 1,
            },
            ..texture_descriptor
        });

        let texture_view_descriptor = TextureViewDescriptor {
            label: None,
            format: Some(TextureFormat::Rgba16Float),
            dimension: Some(TextureViewDimension::D2Array),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: NonZeroU32::new(div_ceil(SPECIES_COUNT, 4)),
        };
        let primary_view_a = primary_texture_a.create_view(&TextureViewDescriptor {
            label: Some("primary_view_a"),
            ..texture_view_descriptor
        });
        let primary_view_b = primary_texture_b.create_view(&TextureViewDescriptor {
            label: Some("primary_view_b"),
            ..texture_view_descriptor
        });
        let update_write_view = update_texture.create_view(&TextureViewDescriptor {
            label: Some("update_write_view"),
            format: Some(TextureFormat::R32Float),
            array_layer_count: NonZeroU32::new(SPECIES_COUNT),
            ..texture_view_descriptor
        });
        let combine_view = combine_texture.create_view(&TextureViewDescriptor {
            label: Some("combine_view"),
            format: Some(TextureFormat::Rgba8Unorm),
            dimension: Some(TextureViewDimension::D2),
            array_layer_count: NonZeroU32::new(1),
            ..texture_view_descriptor
        });

        let time_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("time_buffer"),
            size: 8,
            usage: BufferUsage::COPY_DST | BufferUsage::UNIFORM,
            mapped_at_creation: false,
        });
        let time_bgl = render_device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("time_bgl"),
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStage::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: BufferSize::new(8),
                },
                count: None,
            }],
        });
        let time_bg = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("time_bg"),
            layout: &time_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &time_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        let update_bgl = render_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mold_update_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(16),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(24),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });
        let update_bg_a = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("mold_update_bg_a"),
            layout: &update_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &agent_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &settings_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&primary_view_a),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&update_write_view),
                },
            ],
        });
        let update_bg_b = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("mold_update_bg_b"),
            layout: &update_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &agent_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &settings_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&primary_view_b),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&update_write_view),
                },
            ],
        });
        let update_l = render_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mold_update_l"),
            bind_group_layouts: &[&update_bgl, &time_bgl],
            push_constant_ranges: &[],
        });
        let update_pipeline =
            render_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mold_update"),
                layout: Some(&update_l),
                module: &shader_module,
                entry_point: "update",
            });

        let blur_bgl = render_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("mold_blur_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(8),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: TextureFormat::Rgba16Float,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                    count: None,
                },
            ],
        });
        let blur_bg_a = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("mold_blur_bg_a"),
            layout: &blur_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &global_settings_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&primary_view_a),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&update_write_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&primary_view_b),
                },
            ],
        });
        let blur_bg_b = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("mold_blur_bg_b"),
            layout: &blur_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &global_settings_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&primary_view_b),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&update_write_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::TextureView(&primary_view_a),
                },
            ],
        });
        let blur_l = render_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mold_blur_l"),
            bind_group_layouts: &[&blur_bgl, &time_bgl],
            push_constant_ranges: &[],
        });

        let blur_pipeline =
            render_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("mold_blur"),
                layout: Some(&blur_l),
                module: &shader_module,
                entry_point: "blur",
            });

        let combine_bgl =
            render_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mold_combine_bgl"),
                entries: &[
                    BindGroupLayoutEntry {
                        binding: 0,
                        visibility: ShaderStage::COMPUTE,
                        ty: BindingType::Buffer {
                            ty: BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: NonZeroU64::new(16),
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 1,
                        visibility: ShaderStage::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: TextureFormat::Rgba16Float,
                            view_dimension: TextureViewDimension::D2Array,
                        },
                        count: None,
                    },
                    BindGroupLayoutEntry {
                        binding: 2,
                        visibility: ShaderStage::COMPUTE,
                        ty: BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: TextureFormat::Rgba8Unorm,
                            view_dimension: TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });

        let combine_bg_a = render_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mold_combine_bg_a"),
            layout: &combine_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &combine_settings_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&primary_view_a),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&combine_view),
                },
            ],
        });
        let combine_bg_b = render_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mold_combine_bg_b"),
            layout: &combine_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &combine_settings_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&primary_view_b),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&combine_view),
                },
            ],
        });

        let combine_l = render_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("combine_l"),
            push_constant_ranges: &[],
            bind_group_layouts: &[&combine_bgl],
        });

        let combine_pipeline =
            render_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("combine"),
                layout: Some(&combine_l),
                module: &shader_module,
                entry_point: "combine",
            });

        let display_bgl =
            render_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mold_bgl"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: TextureFormat::Rgba8Unorm,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                }],
            });

        let display_bg = render_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mold_display_bg"),
            layout: &display_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(&combine_view),
            }],
        });

        let display_l = render_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            push_constant_ranges: &[],
            bind_group_layouts: &[&display_bgl],
        });

        let display_pipeline =
            render_device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                vertex: VertexState {
                    buffers: &[],
                    module: &display_shader_module,
                    entry_point: "vs_main",
                },
                fragment: Some(FragmentState {
                    module: &display_shader_module,
                    entry_point: "fs_main",
                    targets: &[ColorTargetState {
                        format: TextureFormat::bevy_default(),
                        blend: Some(BlendState {
                            color: BlendComponent {
                                src_factor: BlendFactor::Src,
                                dst_factor: BlendFactor::OneMinusSrc,
                                operation: BlendOperation::Add,
                            },
                            alpha: BlendComponent {
                                src_factor: BlendFactor::One,
                                dst_factor: BlendFactor::One,
                                operation: BlendOperation::Add,
                            },
                        }),
                        write_mask: ColorWrite::ALL,
                    }],
                }),
                depth_stencil: None,
                layout: Some(&display_l),
                multisample: MultisampleState::default(),
                primitive: PrimitiveState {
                    topology: PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: FrontFace::Ccw,
                    cull_mode: Some(Face::Back),
                    polygon_mode: PolygonMode::Fill,
                    clamp_depth: false,
                    conservative: false,
                },
            });

        let read_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("fetch_buffer"),
            size: 4 * (TEX_WIDTH * TEX_HEIGHT) as u64,
            usage: BufferUsage::COPY_DST | BufferUsage::MAP_READ,
            mapped_at_creation: false,
        });

        MoldShaders {
            update_pipeline,
            update_bg_a,
            update_bg_b,

            blur_pipeline,
            blur_bg_a,
            blur_bg_b,

            combine_pipeline,
            combine_bg_a,
            combine_bg_b,

            display_pipeline,
            display_bg,

            combine_texture,
            update_texture,

            read_buffer,
            time_buffer,
            time_bg,
        }
    }
}

const FIXED_DELTA_TIME: f32 = 1. / 50.;

pub struct MoldNode {
    inner: Mutex<MoldNodeInner>,
}

pub struct MoldNodeInner {
    time: f32,
    state: ReadState,
}

enum ReadState {
    A,
    B,
}

const RUNS_PER_FRAME: usize = 5;
const SAVE_TO_DISK: bool = false;

impl Node for MoldNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let shaders = world.get_resource::<MoldShaders>().unwrap();
        let render_queue = world.get_resource::<RenderQueue>().unwrap();
        let this = &mut *self.inner.lock().unwrap();

        for _ in 0..RUNS_PER_FRAME {
            render_queue.write_buffer(
                &shaders.time_buffer,
                0,
                bytemuck::bytes_of(&PlainTime {
                    total: this.time,
                    delta: FIXED_DELTA_TIME,
                }),
            );

            let mut pass =
                render_context
                    .command_encoder
                    .begin_compute_pass(&ComputePassDescriptor {
                        label: Some("run-update"),
                    });

            pass.set_bind_group(1, shaders.time_bg.value(), &[]);

            let (update_bg, blur_bg) = match this.state {
                ReadState::A => (shaders.update_bg_a.value(), shaders.blur_bg_a.value()),
                ReadState::B => (shaders.update_bg_b.value(), shaders.blur_bg_b.value()),
            };

            pass.set_pipeline(&shaders.update_pipeline);
            pass.set_bind_group(0, update_bg, &[]);
            pass.dispatch(div_ceil(AGENT_COUNT, 32), 1, 1);

            pass.set_pipeline(&shaders.blur_pipeline);
            pass.set_bind_group(0, blur_bg, &[]);
            pass.dispatch(
                div_ceil(TEX_WIDTH, 32),
                div_ceil(TEX_HEIGHT, 32),
                div_ceil(SPECIES_COUNT, 4),
            );

            drop(pass);

            render_context.command_encoder.clear_texture(
                &shaders.update_texture,
                &wgpu_types::ImageSubresourceRange {
                    aspect: wgpu::TextureAspect::All,
                    base_mip_level: 0,
                    mip_level_count: None,
                    base_array_layer: 0,
                    array_layer_count: NonZeroU32::new(SPECIES_COUNT),
                },
            );

            this.time += FIXED_DELTA_TIME;
            this.state = match this.state {
                ReadState::A => ReadState::B,
                ReadState::B => ReadState::A,
            };
        }

        let mut pass = render_context
            .command_encoder
            .begin_compute_pass(&ComputePassDescriptor {
                label: Some("run-combine"),
            });

        pass.set_pipeline(&shaders.combine_pipeline);
        pass.set_bind_group(
            0,
            match this.state {
                ReadState::A => shaders.combine_bg_a.value(),
                ReadState::B => shaders.combine_bg_b.value(),
            },
            &[],
        );
        pass.dispatch(div_ceil(TEX_WIDTH, 32), div_ceil(TEX_HEIGHT, 32), 1);

        drop(pass);
        if SAVE_TO_DISK {
            render_context.command_encoder.copy_texture_to_buffer(
                ImageCopyTexture {
                    texture: &shaders.combine_texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                },
                ImageCopyBuffer {
                    buffer: &shaders.read_buffer,
                    layout: ImageDataLayout {
                        offset: 0,
                        bytes_per_row: NonZeroU32::new(4 * TEX_WIDTH),
                        rows_per_image: NonZeroU32::new(TEX_HEIGHT),
                    },
                },
                Extent3d {
                    width: TEX_WIDTH,
                    height: TEX_HEIGHT,
                    depth_or_array_layers: 1,
                },
            );

            let slice = shaders.read_buffer.slice(..);
            render_context
                .render_device
                .map_buffer(&slice, MapMode::Read);
            let view = slice.get_mapped_range();

            let filepath = format!(
                "images/frame_{}.png",
                (this.time / (FIXED_DELTA_TIME * RUNS_PER_FRAME as f32)) as u32 - 1
            );

            image::save_buffer_with_format(
                filepath,
                &view,
                TEX_WIDTH,
                TEX_HEIGHT,
                image::ColorType::Rgba8,
                image::ImageFormat::Png,
            )
            .unwrap();

            drop(view);
            shaders.read_buffer.unmap();
        }
        let ew = &world.get_resource::<ExtractedWindows>().unwrap().windows[&WindowId::primary()];
        let swapchain = ew.swap_chain_frame.as_ref().unwrap();
        let mut pass = render_context
            .command_encoder
            .begin_render_pass(&RenderPassDescriptor {
                label: Some("mold_display"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: swapchain,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

        pass.set_pipeline(&shaders.display_pipeline);
        pass.set_bind_group(0, shaders.display_bg.value(), &[]);
        pass.draw(0..3, 0..1);
        drop(pass);
        Ok(())
    }
}

fn div_ceil(val: u32, div: u32) -> u32 {
    let excess = val % div;
    if excess > 0 {
        val / div + 1
    } else {
        val / div
    }
}
