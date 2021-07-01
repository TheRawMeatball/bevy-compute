use std::num::NonZeroU32;

use bevy::{
    math::f32,
    prelude::*,
    render2::{
        core_pipeline,
        render_graph::{Node, NodeRunError, RenderGraph, RenderGraphContext},
        render_resource::{
            BindGroup, BindGroupLayoutEntry, BindingType, BlendComponent, BlendFactor,
            BlendOperation, BlendState, Buffer, BufferBindingType, BufferSize, ColorTargetState,
            ColorWrite, ComputePipeline, RenderPipeline, ShaderStage, Texture, TextureFormat,
            TextureView, TextureViewDimension, VertexState,
        },
        renderer::{RenderContext, RenderDevice, RenderQueue},
        shader::Shader,
        texture::BevyDefault,
        view::ExtractedWindows,
        RenderStage,
    },
    window::WindowId,
    PipelinedDefaultPlugins,
};
use rand::Rng;
use tiff::encoder::colortype::Gray32Float;
use wgpu::{
    util::BufferInitDescriptor, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindingResource, BufferBinding, BufferDescriptor, BufferUsage, ComputePassDescriptor, Extent3d,
    Face, FragmentState, FrontFace, ImageCopyBuffer, ImageCopyTexture, ImageDataLayout, LoadOp,
    MapMode, MultisampleState, Operations, Origin3d, PolygonMode, PrimitiveState,
    PrimitiveTopology, RenderPassDescriptor, TextureDescriptor, TextureUsage,
    TextureViewDescriptor,
};

#[bevy_main]
pub fn main() {
    let mut app = App::new();
    app.insert_resource(WindowDescriptor {
        width: 1920.,
        height: 1080.,
        ..Default::default()
    })
    .add_plugins(PipelinedDefaultPlugins);

    let render_app = app.sub_app_mut(0);
    render_app.add_system_to_stage(RenderStage::Extract, time_extract_system.system());
    render_app.init_resource::<MoldShaders>();
    let mut graph = render_app.world.get_resource_mut::<RenderGraph>().unwrap();
    graph.add_node("mold", MoldNode { time: 0. });
    graph
        .add_node_edge("mold", core_pipeline::node::MAIN_PASS_DEPENDENCIES)
        .unwrap();

    app.add_startup_system(setup_system.system());

    app.run();
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
const TEX_WIDTH: u32 = 1920;
const TEX_HEIGHT: u32 = 1080;

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy)]
struct Agent {
    position: Vec2,
    direction: f32,
    _pad: f32,
}

#[repr(C)]
#[derive(bytemuck::Zeroable, bytemuck::Pod, Clone, Copy)]
struct Metadata {
    agent_count: u32,
}

pub struct MoldShaders {
    move_pipeline: ComputePipeline,
    move_bg: BindGroup,
    blur_pipeline: ComputePipeline,
    blur_bg: BindGroup,

    metadata_bg: BindGroup,

    display_pipeline: RenderPipeline,
    display_bg: BindGroup,

    primary_texture: Texture,
    blur_write_texture: Texture,
    update_write_view: TextureView,

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
            _pad: 0.,
        }
    }

    fn gen_point(rng: &mut impl Rng) -> Self {
        let offset = Vec2::new(TEX_WIDTH as f32, TEX_HEIGHT as f32) / 2.;
        Agent {
            position: offset,
            direction: rng.gen_range(-std::f32::consts::PI..std::f32::consts::PI),
            _pad: 0.,
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
            .map(|_| Agent::gen_circle(&mut rng, 520.))
            .collect::<Vec<_>>();
        let agent_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("mold_agents"),
            usage: BufferUsage::STORAGE,
            contents: bytemuck::cast_slice(&agents),
        });
        let primary_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("trail_map"),
            size: Extent3d {
                width: TEX_WIDTH,
                height: TEX_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsage::STORAGE | TextureUsage::COPY_DST,
        });
        let update_write_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("update_write_trail_map"),
            size: Extent3d {
                width: TEX_WIDTH,
                height: TEX_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsage::STORAGE | TextureUsage::RENDER_ATTACHMENT,
        });
        let blur_write_texture = render_device.create_texture(&TextureDescriptor {
            label: Some("blur_write_trail_map"),
            size: Extent3d {
                width: TEX_WIDTH,
                height: TEX_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::R32Float,
            usage: TextureUsage::STORAGE | TextureUsage::COPY_SRC,
        });
        let primary_view = primary_texture.create_view(&TextureViewDescriptor {
            label: None,
            format: Some(TextureFormat::R32Float),
            dimension: Some(TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        let update_write_view = update_write_texture.create_view(&TextureViewDescriptor {
            label: None,
            format: Some(TextureFormat::R32Float),
            dimension: Some(TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });
        let blur_write_view = blur_write_texture.create_view(&TextureViewDescriptor {
            label: None,
            format: Some(TextureFormat::R32Float),
            dimension: Some(TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: None,
            base_array_layer: 0,
            array_layer_count: None,
        });

        let metadata = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("mold_meta"),
            contents: bytemuck::bytes_of(&Metadata {
                agent_count: AGENT_COUNT,
            }),
            usage: BufferUsage::UNIFORM,
        });
        let metadata_bgl =
            render_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mold_meta_bgl"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(4),
                    },
                    count: None,
                }],
            });

        let metadata_bg = render_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mold_meta_bg"),
            layout: &metadata_bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: &metadata,
                    offset: 0,
                    size: None,
                }),
            }],
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
            label: Some("mold_move_bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(64),
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let update_bg = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("mold_update_bg"),
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
                    resource: BindingResource::TextureView(&primary_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&update_write_view),
                },
            ],
        });
        let update_l = render_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mold_update_l"),
            bind_group_layouts: &[&update_bgl, &metadata_bgl, &time_bgl],
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
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStage::COMPUTE,
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: TextureFormat::R32Float,
                        view_dimension: TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let blur_bg = render_device.create_bind_group(&BindGroupDescriptor {
            label: Some("mold_blur_bg"),
            layout: &blur_bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::TextureView(&primary_view),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(&update_write_view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&blur_write_view),
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

        let display_bgl =
            render_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mold_bgl"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStage::FRAGMENT,
                    ty: BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::ReadOnly,
                        format: TextureFormat::R32Float,
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
                resource: BindingResource::TextureView(&primary_view),
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
            move_pipeline: update_pipeline,
            move_bg: update_bg,
            blur_pipeline,
            blur_bg,
            metadata_bg,
            display_pipeline,
            display_bg,
            primary_texture,
            update_write_view,
            blur_write_texture,
            time_buffer,
            time_bg,
            read_buffer,
        }
    }
}

const FIXED_DELTA_TIME: f32 = 1. / 50.;

pub struct MoldNode {
    time: f32,
}

const RUNS_PER_FRAME: usize = 5;

impl Node for MoldNode {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let shaders = world.get_resource::<MoldShaders>().unwrap();
        let render_queue = world.get_resource::<RenderQueue>().unwrap();
        render_queue.write_buffer(
            &shaders.time_buffer,
            0,
            bytemuck::bytes_of(&PlainTime {
                total: self.time,
                delta: FIXED_DELTA_TIME,
            }),
        );

        for _ in 0..RUNS_PER_FRAME {
            {
                let mut pass =
                    render_context
                        .command_encoder
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("run-move"),
                        });

                pass.set_pipeline(&shaders.move_pipeline);
                pass.set_bind_group(2, shaders.time_bg.value(), &[]);
                pass.set_bind_group(1, shaders.metadata_bg.value(), &[]);
                pass.set_bind_group(0, shaders.move_bg.value(), &[]);
                pass.dispatch(div_ceil(AGENT_COUNT, 32), 1, 1);
            }
            {
                let mut pass =
                    render_context
                        .command_encoder
                        .begin_compute_pass(&ComputePassDescriptor {
                            label: Some("run-blur"),
                        });

                pass.set_pipeline(&shaders.blur_pipeline);
                pass.set_bind_group(1, shaders.time_bg.value(), &[]);
                pass.set_bind_group(0, shaders.blur_bg.value(), &[]);
                pass.dispatch(div_ceil(TEX_WIDTH, 32), div_ceil(TEX_HEIGHT, 32), 1);
            }

            render_context
                .command_encoder
                .begin_render_pass(&RenderPassDescriptor {
                    label: Some("clear"),
                    color_attachments: &[wgpu::RenderPassColorAttachment {
                        view: &shaders.update_write_view,
                        resolve_target: None,
                        ops: Operations {
                            load: LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    }],
                    depth_stencil_attachment: None,
                });

            render_context.command_encoder.copy_texture_to_texture(
                ImageCopyTexture {
                    texture: &shaders.blur_write_texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                },
                ImageCopyTexture {
                    texture: &shaders.primary_texture,
                    mip_level: 0,
                    origin: Origin3d::ZERO,
                },
                Extent3d {
                    width: TEX_WIDTH,
                    height: TEX_HEIGHT,
                    depth_or_array_layers: 1,
                },
            );
        }

        render_context.command_encoder.copy_texture_to_buffer(
            ImageCopyTexture {
                texture: &shaders.blur_write_texture,
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
        let floats: &[f32] = bytemuck::cast_slice(&view);

        let mut options = std::fs::OpenOptions::new();
        options.write(true);
        options.create(true);
        let file = options
            .open(format!(
                "images/frame_{}.tiff",
                (self.time / FIXED_DELTA_TIME) as u32 - 1
            ))
            .unwrap();
        let mut encoder = tiff::encoder::TiffEncoder::new(&file).unwrap();
        encoder
            .write_image::<Gray32Float>(TEX_WIDTH, TEX_HEIGHT, floats)
            .unwrap();
        drop(encoder);
        drop(file);
        drop(view);
        shaders.read_buffer.unmap();

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

    fn update(&mut self, _world: &mut World) {
        self.time += FIXED_DELTA_TIME;
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
