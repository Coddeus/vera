//! This crate is where Vulkan is set up (`Vera::create()`) and core actions are handled (`show()`, `save()`, etc.)

// pub mod elements;
// pub use elements::*;

/// Buffers which will be sent to the GPU, and rely on the vulkano crate to be compiled
pub mod buffers;
pub use buffers::*;
/// Matrices, which interpret the input transformations from vera, on which transformations are applied, and which are sent in buffers.
pub mod matrix;
pub use matrix::*;
/// Colors, which interpret the input "colorizations" (color transformations) from vera, for the color of vertices, and are sent in a buffer.
pub mod color;
pub use color::*;
/// "Transformers", update buffers of matrices and colors according to vera input transformations and colorizations: start/end time, speed evolution
pub mod transformer;
pub use transformer::*;

use vera::{Input, Model, Tf, View, Projection, Transformation, Evolution, Colorization, Cl};
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::graphics::depth_stencil::{DepthStencilState, DepthState};

use std::sync::Arc;
use std::time::Instant;

use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, ColorBlendAttachmentState, ColorBlendState,
};
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer, AllocateBufferError, BufferContents};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, CopyBufferInfo,
    PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo,
};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageUsage, ImageCreateInfo, ImageType};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{
    AllocationCreateInfo, FreeListAllocator, GenericMemoryAllocator, MemoryTypeFilter,
    StandardMemoryAllocator, MemoryAllocator,
};
use vulkano::format::Format;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo, ComputePipeline,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::EntryPoint;
use vulkano::swapchain::{
    acquire_next_image, PresentFuture, Surface, Swapchain, SwapchainAcquireFuture,
    SwapchainCreateInfo, SwapchainPresentInfo,
};
use vulkano::sync::future::{FenceSignalFuture, JoinFuture};
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, Version, VulkanError};

use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Window, WindowBuilder};

/// The struct the user interacts with.
pub struct Vera {
    event_loop: EventLoop<()>,
    vk: Vk,
}

/// The underlying base with executes specific tasks.
struct Vk {
    // ----- Created on initialization
    library: Arc<vulkano::VulkanLibrary>,
    required_extensions: vulkano::instance::InstanceExtensions,
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    window: Arc<Window>,

    device_extensions: DeviceExtensions,
    physical_device: Arc<PhysicalDevice>,
    queue_family_index: u32,
    device: Arc<Device>,
    queue: Arc<Queue>,

    swapchain: Arc<Swapchain>,
    images: Vec<Arc<Image>>,
    render_pass: Arc<RenderPass>,
    framebuffers: Vec<Arc<Framebuffer>>,
    
    memory_allocator: Arc<GenericMemoryAllocator<FreeListAllocator>>,
    cb_allocator: StandardCommandBufferAllocator,
    ds_allocator: StandardDescriptorSetAllocator,


    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swapchain: bool,
    show_count: u32,
    time: f32,


    draw_pipeline: Arc<GraphicsPipeline>,
    draw_descriptor_set: Arc<PersistentDescriptorSet>,
    draw_descriptor_set_layout: Arc<DescriptorSetLayout>,
    draw_descriptor_set_layout_index: usize,
    compute_pipeline: Arc<ComputePipeline>,
    compute_descriptor_set: Arc<PersistentDescriptorSet>,
    compute_descriptor_set_layout: Arc<DescriptorSetLayout>,
    compute_descriptor_set_layout_index: usize,

    drawing_vs: EntryPoint,
    drawing_fs: EntryPoint,
    frames_in_flight: usize,
    drawing_fences: Vec<
        Option<
            Arc<
                FenceSignalFuture<
                    PresentFuture<
                        CommandBufferExecFuture<
                            JoinFuture<Box<dyn GpuFuture>, SwapchainAcquireFuture>,
                        >,
                    >,
                >,
            >,
        >,
    >,
    previous_drawing_fence_i: u32,
    
    // -----
    background_color: [f32; 4],
    start_time: f32,
    end_time: f32,


    general_buffer_cs: Subbuffer<[CSGeneral]>,
    general_push_vs: VSGeneral,
    general_push_transformer: (Transformer, Transformer),


    vsinput_buffer: Subbuffer<[BaseVertex]>,
    basevertex_buffer: Subbuffer<[BaseVertex]>,
    vertext_buffer: Subbuffer<[MatrixT]>,
    vertexc_buffer: Subbuffer<[VectorT]>,
    entity_buffer: Subbuffer<[Entity]>,
    modelt_buffer: Subbuffer<[MatrixT]>,

    vertex_matrixtransformation_buffer: Subbuffer<[MatrixTransformation]>,
    vertex_matrixtransformer_buffer: Subbuffer<[MatrixTransformer]>,
    vertex_colortransformation_buffer: Subbuffer<[ColorTransformation]>,
    vertex_colortransformer_buffer: Subbuffer<[ColorTransformer]>,

    model_matrixtransformation_buffer: Subbuffer<[MatrixTransformation]>,
    model_matrixtransformer_buffer: Subbuffer<[MatrixTransformer]>,
}


mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: r"
            #version 460
            #define PI 3.1415926535

            layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;


            // STRUCTS

            struct BaseVertex {
                vec4 position;
                vec4 color;
                uint entity_id; // Index in t.d
            };
            struct ModelT {
                mat4 mat;
            };
            struct MatrixTransformation {
                uint ty; // The type of matrix transformation
                vec3 val; // The values of this transformation, interpreted depending on the type.
                float start;
                float end;
                uint evolution;
            };
            struct MatrixTransformer {
                mat4 current; // The current resulting matrix, acts a pre-result cache.
                uvec2 range; // The index range of the matrix transformations of this transformer inside `t`.
            };
            struct ColorTransformation {
                uint ty; // The type of color transformation
                vec4 val; // The values of this transformation, interpreted depending on the type.
                float start;
                float end;
                uint evolution;
            };
            struct ColorTransformer {
                vec4 current; // The current resulting color, acts a pre-output cache.
                uvec2 range; // The index range of the color transformations of this transformer inside `c`.
            };


            // BUFFERS

            // Per-vertex
            layout(set = 0, binding = 0) buffer OutputVertices {
                BaseVertex d[]; // 'd' for 'Data'.
            } ov;
            layout(set = 0, binding = 1) readonly buffer InputVertices {
                BaseVertex d[];
            } iv;
            layout(set = 0, binding = 2) buffer MatrixTransformers {
                MatrixTransformer d[];
            } tf;
            layout(set = 0, binding = 3) readonly buffer MatrixTransformations {
                MatrixTransformation d[];
            } t;
            layout(set = 0, binding = 4) buffer ColorTransformers {
                ColorTransformer d[];
            } cl;
            layout(set = 0, binding = 5) readonly buffer ColorTransformations {
                ColorTransformation d[];
            } c;
            // Per-model
            layout(set = 0, binding = 6) buffer OutputModels {
                ModelT d[];
            } om;
            layout(set = 0, binding = 7) buffer Model_MatrixTransformers {
                MatrixTransformer d[];
            } m_tf;
            layout(set = 0, binding = 8) readonly buffer Model_MatrixTransformations {
                MatrixTransformation d[];
            } m_t;
            // Unique
            layout(set = 0, binding = 9) buffer GeneralInfo {
                float time;
                uint entity_count;
            } gen;


            // TRANSFORMATIONS

            vec4 interpolate(vec4 start_color, vec4 end_color, float advancement) {
                return vec4(
                    start_color[0] * (1.0-advancement) + end_color[0] * advancement,
                    start_color[1] * (1.0-advancement) + end_color[1] * advancement,
                    start_color[2] * (1.0-advancement) + end_color[2] * advancement,
                    start_color[3] * (1.0-advancement) + end_color[3] * advancement
                );
            }

            mat4 scale(float x_scale, float y_scale, float z_scale) {
                return mat4(
                    x_scale ,   0.0     ,   0.0     , 0.0 , 
                    0.0     ,   y_scale ,   0.0     , 0.0 , 
                    0.0     ,   0.0     ,   z_scale , 0.0 , 
                    0.0     ,   0.0     ,   0.0     , 1.0
                );
            }

            mat4 translate(float x_move, float y_move, float z_move) {
                return mat4(
                    1.0 ,   0.0 ,   0.0 ,   x_move ,
                    0.0 ,   1.0 ,   0.0 ,   y_move ,
                    0.0 ,   0.0 ,   1.0 ,   z_move ,
                    0.0 ,   0.0 ,   0.0 ,   1.0
                );
            }

            mat4 rotate_x(float angle) {
                return mat4(
                    1.0     ,   0.0         ,   0.0         , 0.0 , 
                    0.0     ,   cos(angle) ,   sin(angle) , 0.0 , 
                    0.0     ,   -sin(angle),   cos(angle) , 0.0 , 
                    0.0     ,   0.0         ,   0.0         , 1.0
                );
            }

            mat4 rotate_y(float angle) {
                return mat4(
                    cos(angle) ,   0.0 ,   sin(angle) , 0.0 , 
                    0.0         ,   1.0 ,   0.0         , 0.0 , 
                    -sin(angle),   0.0 ,   cos(angle) , 0.0 , 
                    0.0         ,   0.0 ,   0.0         , 1.0
                );
            }

            mat4 rotate_z(float angle) {
                return mat4(
                    cos(angle) ,   sin(angle) ,   0.0 ,   0.0 , 
                    -sin(angle),   cos(angle) ,   0.0 ,   0.0 , 
                    0.0         ,   0.0         ,   1.0 ,   0.0 , 
                    0.0         ,   0.0         ,   0.0 ,   1.0
                );
            }


            // EVOLUTIONS

            float advancement(float start, float end, uint e) {
                if (start>=end) {
                    if (gen.time<end) { return 0.0; }
                    else { return 1.0; }
                }
                if (gen.time < start) {
                    return 0.0;
                }
                if (gen.time >= end) {
                    return 1.0;
                }
                float init = (gen.time-start)/(end-start);

                if (e==0) {
                    return init;
                } else if (e==1) {
                    return sin(init * PI / 2.0);
                } else if (e==2) {
                    return 1.0 - cos(init * PI / 2.0);
                } else if (e==3) {
                    return (sin((init - 0.5) * PI) + 1.0) / 2.0;
                } else if (e==4) {
                    if (init < 0.5) { return sin(init * PI) / 2.0; }
                    else { return 0.5 + (1.0 - sin(init * PI)) / 2.0; }
                }
            }

            // TRANSFORMATIONS MATCHING

            mat4 vertex_matrix_transformation(uint i) {
                float adv = advancement(t.d[i].start, t.d[i].end, t.d[i].evolution);
                if (t.d[i].ty==0) {
                    return scale(t.d[i].val.x * adv, t.d[i].val.y * adv, t.d[i].val.z * adv);
                } else if (t.d[i].ty==1) {
                    return translate(t.d[i].val.x * adv, t.d[i].val.y * adv, t.d[i].val.z * adv);
                } else if (t.d[i].ty==2) {
                    return rotate_x(t.d[i].val.x * adv);
                } else if (t.d[i].ty==3) {
                    return rotate_y(t.d[i].val.x * adv);
                } else {
                    return rotate_z(t.d[i].val.x * adv);
                }
            }

            vec4 vertex_color_transformation(uint i, vec4 out_color) {
                float adv = advancement(c.d[i].start, c.d[i].end, c.d[i].evolution);
                if (c.d[i].ty==0) {
                    return interpolate(out_color, c.d[i].val, adv);
                }
            }

            mat4 model_matrix_transformation(uint i) {
                float adv = advancement(m_t.d[i].start, m_t.d[i].end, m_t.d[i].evolution);
                if (m_t.d[i].ty==0) {
                    return scale(m_t.d[i].val.x * adv, m_t.d[i].val.y * adv, m_t.d[i].val.z * adv);
                } else if (m_t.d[i].ty==1) {
                    return translate(m_t.d[i].val.x * adv, m_t.d[i].val.y * adv, m_t.d[i].val.z * adv);
                } else if (m_t.d[i].ty==2) {
                    return rotate_x(m_t.d[i].val.x * adv);
                } else if (m_t.d[i].ty==3) {
                    return rotate_y(m_t.d[i].val.x * adv);
                } else {
                    return rotate_z(m_t.d[i].val.x * adv);
                }
            }


            // TRANSFORMING

            void transform_vertex() {
                // position
                bool first = true;
                vec4 out_position = iv.d[gl_GlobalInvocationID.x].position * tf.d[gl_GlobalInvocationID.x].current;
                for (uint i=tf.d[gl_GlobalInvocationID.x].range.x; i<tf.d[gl_GlobalInvocationID.x].range.y; i++) {
                    if (first && t.d[i].end<gen.time) {
                        tf.d[gl_GlobalInvocationID.x].current *= vertex_matrix_transformation(i);
                        tf.d[gl_GlobalInvocationID.x].range.x += 1;
                    } else {
                        first = false;
                    }
                    out_position *= vertex_matrix_transformation(i);
                }
                ov.d[gl_GlobalInvocationID.x].position = out_position;
                
                // color
                first = true;
                vec4 out_color = cl.d[gl_GlobalInvocationID.x].current;
                for (uint i=cl.d[gl_GlobalInvocationID.x].range.x; i<cl.d[gl_GlobalInvocationID.x].range.y; i++) {
                    if (first && t.d[i].end<gen.time) {
                        cl.d[gl_GlobalInvocationID.x].current *= vertex_color_transformation(i, out_color);
                        cl.d[gl_GlobalInvocationID.x].range.x += 1;
                    } else {
                        first = false;
                    }
                    out_color *= vertex_color_transformation(i, out_color);
                }
                ov.d[gl_GlobalInvocationID.x].color = out_color;
            }

            void transform_model() {
                bool first = true;
                uint entity_id = atomicAdd(gen.entity_count, 1) + 1;
                mat4 out_model = m_tf.d[entity_id].current;
                for (uint i=m_tf.d[entity_id].range.x; i<m_tf.d[entity_id].range.y; i++) {
                    if (first && m_t.d[i].end<gen.time) {
                        m_tf.d[entity_id].current *= model_matrix_transformation(i);
                        m_tf.d[entity_id].range.x += 1;
                    } else {
                        first = false;
                    }
                    out_model *= model_matrix_transformation(i);
                }
                om.d[entity_id].mat = out_model;
            }


            // MAIN

            void main() {
                // If *model* transformation not already calculated, calculate it and atomically increase gen.entity_count.
                if (iv.d[gl_GlobalInvocationID.x].entity_id != gen.entity_count) {
                    transform_model();
                }

                // Calculate & fill vertex position and color.
                transform_vertex();

                // Copy entity_id.
                ov.d[gl_GlobalInvocationID.x].entity_id = iv.d[gl_GlobalInvocationID.x].entity_id;
            }
        ",
    }
}

mod vs { // Position/Color already calcutated
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            struct Entity {
                uint parent_id; // Index in t.d. 0 if no parent
            };
            struct ModelT {
                mat4 mat;
            };

            layout(location = 0) in vec4 position;
            layout(location = 1) in vec4 color;
            layout(location = 2) in uint entity_id;

            layout(location = 0) out vec4 out_color;

            layout(set = 0, binding = 0) readonly buffer Entities {
                Entity data[];
            } ent;
            layout(set = 0, binding = 1) readonly buffer ModelTransformations {
                ModelT data[];
            } mod;
            layout(push_constant) uniform GeneralInfo {
                mat4 mat;
            } gen;

            void main() {
                gl_Position = position;
                uint model_id = entity_id;
                while (model_id!=0) {
                    gl_Position *= mod.data[model_id].mat;
                    model_id = ent.data[model_id].parent_id;
                }
                gl_Position *= gen.mat;
                out_color = color;
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            layout(location = 0) in vec4 in_color;

            void main() {
                f_color = in_color;
            }
        ",
    }
}

impl Vera {
    /// Sets up Vera with Vulkan
    /// - `input` defines what's to be drawn and how.
    pub fn init(input: Input) -> Self {
        // Event_loop/extensions/instance/surface/window/physical_device/queue_family/device/queue/swapchain/images/render_pass/framebuffers
        // ---------------------------------------------------------------------------------------------------------------------------------
        let event_loop = EventLoop::new();
        let library = vulkano::VulkanLibrary::new().expect("no local Vulkan library/DLL");
        let required_extensions = Surface::required_extensions(&event_loop);
        let instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                application_name: Some("Vera".to_owned()),
                application_version: Version::major_minor(0, 2),
                enabled_extensions: required_extensions,
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .expect("failed to create instance");

        let window = Arc::new(
            WindowBuilder::new()
                .with_inner_size(LogicalSize {
                    width: 800,
                    height: 600,
                })
                .with_resizable(true)
                .with_title("Vera")
                .with_transparent(false)
                .build(&event_loop)
                .unwrap(),
        );
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .expect("failed to enumerate physical devices")
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.contains(QueueFlags::GRAPHICS)
                            && q.queue_flags.contains(QueueFlags::COMPUTE)
                            && q.queue_flags.contains(QueueFlags::TRANSFER)
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|q| (p, q as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                _ => 4,
            })
            .expect("no device available");

        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        let (device, mut queues) = Device::new(
            physical_device.clone(),
            DeviceCreateInfo {
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                enabled_extensions: device_extensions,
                ..Default::default()
            },
        )
        .expect("failed to create device");
        let queue = queues.next().unwrap();

        // Allocators
        // ----------
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let cb_allocator = StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let ds_allocator = StandardDescriptorSetAllocator::new(device.clone(), Default::default());

        // ----------

        let (swapchain, images) = {
            let caps = physical_device
                .surface_capabilities(&surface, Default::default())
                .expect("failed to get surface capabilities");

            let dimensions = window.inner_size();
            let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
            let image_format = physical_device
                .clone()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0;

            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: caps.min_image_count.max(2),
                    image_format,
                    image_extent: dimensions.into(),
                    image_usage: ImageUsage::COLOR_ATTACHMENT,
                    composite_alpha,
                    ..Default::default()
                },
            )
            .unwrap()
        };

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: swapchain.image_format(), // set the format the same as the swapchain
                    samples: 1, // TODOSAMPLES
                    load_op: Clear,
                    store_op: Store,
                },
                depth_stencil: {
                    format: Format::D16_UNORM,
                    samples: 1,
                    load_op: Clear,
                    store_op: DontCare,
                },
            },
            pass: {
                color: [color],
                depth_stencil: {depth_stencil},
            },
        )
        .unwrap();
    

        let depth_attachment = ImageView::new_default(
            Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: Format::D16_UNORM,
                    extent: images[0].extent(),
                    usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                    ..Default::default()
                },
                AllocationCreateInfo::default(),
            )
            .unwrap(),
        )
        .unwrap();

        let framebuffers: Vec<Arc<Framebuffer>> = images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view, depth_attachment.clone()],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<Arc<Framebuffer>>>();

        // ---------------------------------------------------------------------------------------------------------------------------------

        // Buffers
        // -------
        let (
            (background_color, start_time, end_time),
            (general_buffer_cs, general_push_vs, general_push_transformer),
            (
                vsinput_buffer,
                basevertex_buffer,
                vertext_buffer,
                vertexc_buffer,
                entity_buffer,
                modelt_buffer,
    
                vertex_matrixtransformation_buffer,
                vertex_matrixtransformer_buffer,
                vertex_colortransformation_buffer,
                vertex_colortransformer_buffer,
    
                model_matrixtransformation_buffer,
                model_matrixtransformer_buffer,
            ),
        ) = from_input(queue.clone(), memory_allocator.clone(), &cb_allocator, input);

        // -------

        // Graphics pipeline & Drawing command buffer
        // ------------------------------------------
        let drawing_vs = vs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .expect("failed to create vertex shader module");
        let drawing_fs = fs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .expect("failed to create fragment shader module");
        let compute_cs = cs::load(device.clone())
            .unwrap()
            .entry_point("main")
            .expect("failed to create fragment shader module");

        let draw_pipeline: Arc<GraphicsPipeline> = {
            let vertex_input_state = BaseVertex::per_vertex()
                .definition(&drawing_vs.info().input_interface)
                .unwrap();

            let stages = [
                PipelineShaderStageCreateInfo::new(drawing_vs.clone()),
                PipelineShaderStageCreateInfo::new(drawing_fs.clone()),
            ];

            let pipeline_layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    // How vertex data is read from the vertex buffers into the vertex shader.
                    vertex_input_state: Some(vertex_input_state),
                    // How vertices are arranged into primitive shapes.
                    // The default primitive shape is a triangle.
                    input_assembly_state: Some(InputAssemblyState::default()),
                    // How primitives are transformed and clipped to fit the framebuffer.
                    // We use a resizable viewport, set to draw over the entire window.
                    viewport_state: Some(ViewportState::default()),
                    // How polygons are culled and converted into a raster of pixels.
                    // The default value does not perform any culling.
                    rasterization_state: Some(RasterizationState::default()),
                    // How multiple fragment shader samples are converted to a single pixel value.
                    // The default value does not perform any multisampling.
                    multisample_state: Some(MultisampleState::default()),
                    // How pixel values are combined with the values already present in the framebuffer.
                    // The default value overwrites the old value with the new one, without any blending.
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState {
                            blend: Some(AttachmentBlend::alpha()),
                            ..Default::default()
                        },
                    )),
                    subpass: Some(subpass.into()),
                    depth_stencil_state: Some(DepthStencilState {
                        depth: Some(DepthState::simple()),
                        ..Default::default()
                    }),
                    ..GraphicsPipelineCreateInfo::layout(pipeline_layout)
                },
            )
            .unwrap()
        };

        let draw_pipeline_layout: &Arc<vulkano::pipeline::PipelineLayout> = draw_pipeline.layout();
        let descriptor_set_layouts: &[Arc<vulkano::descriptor_set::layout::DescriptorSetLayout>] =
            draw_pipeline_layout.set_layouts();
        
        let draw_descriptor_set_layout_index: usize = 0;
        let draw_descriptor_set_layout: Arc<vulkano::descriptor_set::layout::DescriptorSetLayout> =
            descriptor_set_layouts
                .get(draw_descriptor_set_layout_index)
                .unwrap()
                .clone();
        let draw_descriptor_set: Arc<PersistentDescriptorSet> = PersistentDescriptorSet::new(
            &ds_allocator,
            draw_descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, entity_buffer.clone()),
                WriteDescriptorSet::buffer(1, modelt_buffer.clone()),
            ],
            [],
        )
        .unwrap();



        let compute_pipeline: Arc<ComputePipeline> = {
            let stage = PipelineShaderStageCreateInfo::new(compute_cs);
            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();
            ComputePipeline::new(
                device.clone(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let compute_pipeline_layout: &Arc<vulkano::pipeline::PipelineLayout> = compute_pipeline.layout();
        let descriptor_set_layouts: &[Arc<vulkano::descriptor_set::layout::DescriptorSetLayout>] =
            compute_pipeline_layout.set_layouts();

        let compute_descriptor_set_layout_index: usize = 0;
        let compute_descriptor_set_layout: Arc<vulkano::descriptor_set::layout::DescriptorSetLayout> =
            descriptor_set_layouts
                .get(compute_descriptor_set_layout_index)
                .unwrap()
                .clone();
        let compute_descriptor_set: Arc<PersistentDescriptorSet> = PersistentDescriptorSet::new(
            &ds_allocator,
            compute_descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, vsinput_buffer.clone()),
                WriteDescriptorSet::buffer(1, basevertex_buffer.clone()),
                WriteDescriptorSet::buffer(2, vertex_matrixtransformer_buffer.clone()),
                WriteDescriptorSet::buffer(3, vertex_matrixtransformation_buffer.clone()),
                WriteDescriptorSet::buffer(4, vertex_colortransformer_buffer.clone()),
                WriteDescriptorSet::buffer(5, vertex_colortransformation_buffer.clone()),
                WriteDescriptorSet::buffer(6, model_matrixtransformer_buffer.clone()),
                WriteDescriptorSet::buffer(7, model_matrixtransformation_buffer.clone()),
                WriteDescriptorSet::buffer(8, general_buffer_cs.clone()),
            ],
            [],
        )
        .unwrap();

        // Command buffers:
        // 1. Compute update staging_uniform_buffer,                                     //
        // 1. Draw graphics pipeline using vertex_buffer and final_uniform_buffer,       //
        // 2. Copy data from staging_uniform_buffer to final_uniform_buffer,             //
        // 2. Swap swapchain images,                                                     // Done

        // let drawing_command_buffers: Vec<Arc<PrimaryAutoCommandBuffer>> = framebuffers
        //     .iter()
        //     .map(|framebuffer| {
        //         let mut builder = AutoCommandBufferBuilder::primary(
        //             &command_buffer_allocator,
        //             queue.queue_family_index(),
        //             CommandBufferUsage::MultipleSubmit,
        //         )
        //         .unwrap();
        //
        //         builder
        //             .begin_render_pass(
        //                 RenderPassBeginInfo {
        //                     clear_values: vec![Some([0.0, 0.0, 0.0, 0.0].into())],
        //                     ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
        //                 },
        //                 vulkano::command_buffer::SubpassBeginInfo { contents: SubpassContents::Inline, ..Default::default() },
        //             )
        //             .unwrap()
        //             .set_viewport(0, [drawing_viewport.clone()].into_iter().collect())
        //             .unwrap()
        //             .bind_pipeline_graphics(drawing_pipeline.clone())
        //             .unwrap()
        //             .bind_descriptor_sets(
        //                 PipelineBindPoint::Graphics,
        //                 pipeline_layout.clone(),
        //                 descriptor_set_layout_index as u32,
        //                 descriptor_set.clone(),
        //             )
        //             .unwrap()
        //             .bind_vertex_buffers(0, vertex_buffer.clone())
        //             .unwrap()
        //             .draw(vertex_buffer.len() as u32, 1, 0, 0)
        //             .unwrap()
        //             .end_render_pass(Default::default())
        //             .unwrap();
        //
        //         builder.build().unwrap()
        //     })
        //     .collect();

        let frames_in_flight: usize = images.len();
        let drawing_fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
        let previous_drawing_fence_i: u32 = 0;

        // ------------------------------------------

        // Window-related updates
        // ----------------------
        let previous_frame_end: Option<Box<dyn GpuFuture>> =
            Some(sync::now(device.clone()).boxed());
        let recreate_swapchain: bool = false;
        let show_count: u32 = 0;
        let time: f32 = 0.0;

        // ----------------------

        Vera {
            event_loop,
            vk: Vk {
                //
                library,
                required_extensions,
                instance,
                surface,
                window,

                device_extensions,
                physical_device,
                queue_family_index,
                device,
                queue,

                swapchain,
                images,
                render_pass,
                framebuffers,


                memory_allocator,
                cb_allocator,
                ds_allocator,


                previous_frame_end,
                recreate_swapchain,
                show_count,
                time,


                draw_descriptor_set,
                draw_descriptor_set_layout,
                draw_descriptor_set_layout_index,
                draw_pipeline,
                compute_descriptor_set,
                compute_descriptor_set_layout,
                compute_descriptor_set_layout_index,
                compute_pipeline,


                drawing_vs,
                drawing_fs,
                frames_in_flight,
                drawing_fences,
                previous_drawing_fence_i,


                background_color,
                start_time,
                end_time,


                general_buffer_cs,
                general_push_vs,
                general_push_transformer,


                vsinput_buffer,
                basevertex_buffer,
                vertext_buffer,
                vertexc_buffer,
                entity_buffer,
                modelt_buffer,

                vertex_matrixtransformation_buffer,
                vertex_matrixtransformer_buffer,
                vertex_colortransformation_buffer,
                vertex_colortransformer_buffer,

                model_matrixtransformation_buffer,
                model_matrixtransformer_buffer,

            },
        }
    }

    /// Resets vertex and uniform data from input, which is consumed.
    /// 
    /// In heavy animations, resetting is useful for reducing the CPU/GPU usage if you split you animations in several parts.
    /// Also useful if called from a loop for hot-reloading.
    pub fn reset(&mut self, input: Input) {
        (
            (self.vk.background_color, self.vk.start_time, self.vk.end_time),
            (self.vk.general_buffer_cs, self.vk.general_push_vs, self.vk.general_push_transformer),
            (
                self.vk.vsinput_buffer,
                self.vk.basevertex_buffer,
                self.vk.vertext_buffer,
                self.vk.vertexc_buffer,
                self.vk.entity_buffer,
                self.vk.modelt_buffer,
    
                self.vk.vertex_matrixtransformation_buffer,
                self.vk.vertex_matrixtransformer_buffer,
                self.vk.vertex_colortransformation_buffer,
                self.vk.vertex_colortransformer_buffer,
    
                self.vk.model_matrixtransformation_buffer,
                self.vk.model_matrixtransformer_buffer,
            ),
        ) = from_input(self.vk.queue.clone(), self.vk.memory_allocator.clone(), &self.vk.cb_allocator, input);
    }

    /// Shows the animation form start to end once.
    /// - Panics if there was an unexpected exit code from the event loop
    /// - Returns `false` if the window was closed while showing.
    /// - Returns true otherwise.
    pub fn show(&mut self) -> bool {
        match self.vk.show(&mut self.event_loop) {
            0 => {
                // Successfully finished.
                true
            }
            1 => {
                // Window closed
                println!("\nℹ Window closed.");
                false
            }
            n => {
                println!("🛑 Unexpected return code \"{}\" when running the main loop", n);
                false
            }
        }
    }

    pub fn save(&mut self, width: u32, height: u32) {
        match self.vk.show(&mut self.event_loop) { // , (width, height)
            0 => {
                // Successfully finished
                println!("✨ Saved video!");
            }
            1 => {
                // Window closed
                println!("⁉ Window closed. Stopping encoding now.");
            }
            _ => {
                panic!("🛑 Unexpected return code when running the main loop");
            }
        }
    }
}

/// Treats `input`, and returns created buffers and metadata.
fn from_input(
    queue: Arc<Queue>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    cb_allocator: &StandardCommandBufferAllocator,
    input: Input
) -> (
    ([f32; 4], f32, f32),
    (Subbuffer<[CSGeneral]>, VSGeneral, (Transformer, Transformer)),
    (
        Subbuffer<[BaseVertex]>,
        Subbuffer<[BaseVertex]>,
        Subbuffer<[MatrixT]>,
        Subbuffer<[VectorT]>,
        Subbuffer<[Entity]>,
        Subbuffer<[MatrixT]>,

        Subbuffer<[MatrixTransformation]>,
        Subbuffer<[MatrixTransformer]>,
        Subbuffer<[ColorTransformation]>,
        Subbuffer<[ColorTransformer]>,

        Subbuffer<[MatrixTransformation]>,
        Subbuffer<[MatrixTransformer]>,
    ),
) {
    // Meta and general (CPU and push constants)
    let Input {
        meta,
        m,
        v: View {
            t: view_t,
        },
        p: Projection {
            t: projection_t,
        },
    } = input;

    let general_push_vs: VSGeneral = VSGeneral {
        mat: [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
    };
    let general_buffer_cs: Subbuffer<[CSGeneral]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        [CSGeneral {
            time: 0.0,
            entity_count: 0,
        }].into_iter(),
        1,
        true,
    );
    let general_push_transformer: (Transformer, Transformer) = (Transformer::from_t(view_t), Transformer::from_t(projection_t)); // (View, Projection)

    // SHADER DATA (for the below filled buffers)

    // vb_cs
    let mut basevertex_data: Vec<BaseVertex> = vec![];
    let mut entity_data: Vec<Entity> = vec![];

    // vertext_cs
    // Outputs are vertext_data and vertexc_data (transformations and colorizations)
    let mut vertex_matrixtransformation_data: Vec<MatrixTransformation> = vec![];
    let mut vertex_matrixtransformer_data: Vec<MatrixTransformer> = vec![];
    let mut vertex_colortransformation_data: Vec<ColorTransformation> = vec![];
    let mut vertex_colortransformer_data: Vec<ColorTransformer> = vec![];

    // modelt_cs
    // Output is modelt_data
    let mut model_matrixtransformation_data: Vec<MatrixTransformation> = vec![];
    let mut model_matrixtransformer_data: Vec<MatrixTransformer> = vec![];


    static mut ENTITY_INDEX: u32 = 0;
    static mut VERTEX_MATRIXTRANSFORMATION_OFFSET: u32 = 0;
    static mut VERTEX_COLORTRANSFORMATION_OFFSET: u32 = 0;
    static mut MODEL_MATRIXTRANSFORMATION_OFFSET: u32 = 0;

    for model in m.into_iter() {
        let (
            m_basevertex,
            m_entity,
            m_vertex_matrixtransformation,
            m_vertex_matrixtransformer,
            m_vertex_colortransformation,
            m_vertex_colortransformer,
            m_model_matrixtransformation,
            m_model_matrixtransformer
        ) = from_model(model, 0);

        basevertex_data.extend(m_basevertex);
        entity_data.extend(m_entity);
        vertex_matrixtransformation_data.extend(m_vertex_matrixtransformation);
        vertex_matrixtransformer_data.extend(m_vertex_matrixtransformer);
        vertex_colortransformation_data.extend(m_vertex_colortransformation);
        vertex_colortransformer_data.extend(m_vertex_colortransformer);
        model_matrixtransformation_data.extend(m_model_matrixtransformation);
        model_matrixtransformer_data.extend(m_model_matrixtransformer);
    }

    fn from_model(model: Model, parent_id: u32) -> (Vec<BaseVertex>, Vec<Entity>, Vec<MatrixTransformation>, Vec<MatrixTransformer>, Vec<ColorTransformation>, Vec<ColorTransformer>, Vec<MatrixTransformation>, Vec<MatrixTransformer>,) {
        unsafe { ENTITY_INDEX+=1; }

        let mut basevertex: Vec<BaseVertex> = vec![]; // done
        let mut entity: Vec<Entity> = vec![]; // done
    
        let mut vertex_matrixtransformation: Vec<MatrixTransformation> = vec![]; // 
        let mut vertex_matrixtransformer: Vec<MatrixTransformer> = vec![]; // 
        let mut vertex_colortransformation: Vec<ColorTransformation> = vec![]; // 
        let mut vertex_colortransformer: Vec<ColorTransformer> = vec![]; // 
    
        let mut model_matrixtransformation: Vec<MatrixTransformation> = to_gpu_tf(model.t); // done
        let mmt_len = model_matrixtransformation.len() as u32;
        let mut model_matrixtransformer: Vec<MatrixTransformer> = vec![MatrixTransformer::from_lo(mmt_len, unsafe { MODEL_MATRIXTRANSFORMATION_OFFSET })]; // 
        unsafe { MODEL_MATRIXTRANSFORMATION_OFFSET+=mmt_len; }

        for v in model.vertices.into_iter() {
            basevertex.push(BaseVertex {
                position: v.position,
                color: v.color,
                entity_id: unsafe { ENTITY_INDEX },
            });

            vertex_matrixtransformation.extend(to_gpu_tf(v.t));
            let vmt_len = vertex_matrixtransformation.len() as u32;
            vertex_matrixtransformer.push(MatrixTransformer::from_lo(vmt_len, unsafe { VERTEX_MATRIXTRANSFORMATION_OFFSET }));
            unsafe { VERTEX_MATRIXTRANSFORMATION_OFFSET+=vmt_len; }

            vertex_colortransformation.extend(to_gpu_cl(v.c));
            let vmt_len = vertex_colortransformation.len() as u32;
            vertex_colortransformer.push(ColorTransformer::from_lo(vmt_len, unsafe { VERTEX_COLORTRANSFORMATION_OFFSET }));
            unsafe { VERTEX_COLORTRANSFORMATION_OFFSET+=vmt_len; }
        }

        entity.push(Entity {
            parent_id,
        });

        

        for m in model.models.into_iter() {
            let (
                m_basevertex,
                m_entity,
                m_vertex_matrixtransformation,
                m_vertex_matrixtransformer,
                m_vertex_colortransformation,
                m_vertex_colortransformer,
                m_model_matrixtransformation,
                m_model_matrixtransformer
            ) = from_model(m, parent_id);

            basevertex.extend(m_basevertex);
            entity.extend(m_entity);
            vertex_matrixtransformation.extend(m_vertex_matrixtransformation);
            vertex_matrixtransformer.extend(m_vertex_matrixtransformer);
            vertex_colortransformation.extend(m_vertex_colortransformation);
            vertex_colortransformer.extend(m_vertex_colortransformer);
            model_matrixtransformation.extend(m_model_matrixtransformation);
            model_matrixtransformer.extend(m_model_matrixtransformer);
        }

        (
            basevertex,
            entity,

            vertex_matrixtransformation,
            vertex_matrixtransformer,
            vertex_colortransformation,
            vertex_colortransformer,

            model_matrixtransformation,
            model_matrixtransformer,
        )
    }


    let vertices_len: u64 = basevertex_data.len() as u64;
    let entities_len: u64 = entity_data.len() as u64;
    
    // BUFFERS

    // make non_empty
    vertex_matrixtransformation_data.push(MatrixTransformation::default());
    vertex_colortransformation_data.push(ColorTransformation::default());
    model_matrixtransformation_data.push(MatrixTransformation::default());

    let vsinput_buffer: Subbuffer<[BaseVertex]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::STORAGE_BUFFER | BufferUsage::VERTEX_BUFFER,
        [].into_iter(),
        vertices_len,
        false,
    );
    
    let basevertex_buffer: Subbuffer<[BaseVertex]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        basevertex_data,
        vertices_len,
        true,
    );
    
    let vertext_buffer: Subbuffer<[MatrixT]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::STORAGE_BUFFER | BufferUsage::VERTEX_BUFFER,
        [].into_iter(),
        vertices_len,
        false,
    );

    let vertexc_buffer: Subbuffer<[VectorT]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::STORAGE_BUFFER | BufferUsage::VERTEX_BUFFER,
        [].into_iter(),
        vertices_len,
        false,
    );
    
    let entity_buffer: Subbuffer<[Entity]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        entity_data,
        entities_len,
        true,
    );
    
    let modelt_buffer: Subbuffer<[MatrixT]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::STORAGE_BUFFER,
        [].into_iter(),
        vertices_len,
        false,
    );
    
    let vertex_matrixtransformation_buffer: Subbuffer<[MatrixTransformation]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        vertex_matrixtransformation_data,
        vertices_len,
        true,
    );
    
    let vertex_matrixtransformer_buffer: Subbuffer<[MatrixTransformer]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        vertex_matrixtransformer_data,
        vertices_len,
        true,
    );
    
    let vertex_colortransformation_buffer: Subbuffer<[ColorTransformation]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        vertex_colortransformation_data,
        vertices_len,
        true,
    );
    
    let vertex_colortransformer_buffer: Subbuffer<[ColorTransformer]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        vertex_colortransformer_data,
        vertices_len,
        true,
    );
    
    let model_matrixtransformation_buffer: Subbuffer<[MatrixTransformation]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        model_matrixtransformation_data,
        entities_len,
        true,
    );
    
    let model_matrixtransformer_buffer: Subbuffer<[MatrixTransformer]> = create_buffer(
        queue.clone(),
        memory_allocator.clone(),
        &cb_allocator,
        BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER,
        model_matrixtransformer_data,
        entities_len,
        true,
    );

    (
        (meta.bg, meta.start, meta.end),
        (general_buffer_cs, general_push_vs, general_push_transformer),
        (
            vsinput_buffer,
            basevertex_buffer,
            vertext_buffer,
            vertexc_buffer,
            entity_buffer,
            modelt_buffer,

            vertex_matrixtransformation_buffer,
            vertex_matrixtransformer_buffer,
            vertex_colortransformation_buffer,
            vertex_colortransformer_buffer,

            model_matrixtransformation_buffer,
            model_matrixtransformer_buffer,
        ),
    )
}

fn to_gpu_tf(t: Vec<Tf>) -> Vec<MatrixTransformation> {
    let mut gpu_tf: Vec<MatrixTransformation> = vec![];
    for tf in t.into_iter() {
        let (ty, val) = match tf.t {
            Transformation::Scale(x, y, z) => (0, [x, y, z]),
            Transformation::Translate(x, y, z) => (1, [x, y, z]),
            Transformation::RotateX(angle) => (2, [angle, 0.0, 0.0]),
            Transformation::RotateY(angle) => (3, [angle, 0.0, 0.0]),
            Transformation::RotateZ(angle) => (4, [angle, 0.0, 0.0]),
            _ => { println!("Vertex/Model transformation not implemented, ignoring."); continue; },
        };
        let evolution = match tf.e {
            Evolution::Linear => 0,
            Evolution::FastIn | Evolution::SlowOut => 1,
            Evolution::FastOut | Evolution::SlowIn => 2,
            Evolution::FastMiddle | Evolution::SlowInOut => 3,
            Evolution::FastInOut | Evolution::SlowMiddle =>  4,
        };
        gpu_tf.push(MatrixTransformation {
            ty,
            val,
            start: tf.start,
            end: tf.end,
            evolution,
        })
    }

    gpu_tf
}

fn to_gpu_cl(t: Vec<Cl>) -> Vec<ColorTransformation> {
    let mut gpu_cl: Vec<ColorTransformation> = vec![];
    for tf in t.into_iter() {
        let (ty, val) = match tf.c {
            Colorization::ToColor(r, g, b, a) => (0, [r, g, b, a]),
            _ => { println!("Vertex/Model transformation not implemented, ignoring."); continue; },
        };
        let evolution = match tf.e {
            Evolution::Linear => 0,
            Evolution::FastIn | Evolution::SlowOut => 1,
            Evolution::FastOut | Evolution::SlowIn => 2,
            Evolution::FastMiddle | Evolution::SlowInOut => 3,
            Evolution::FastInOut | Evolution::SlowMiddle =>  4,
        };
        gpu_cl.push(ColorTransformation {
            ty,
            val,
            start: tf.start,
            end: tf.end,
            evolution,
        })
    }

    gpu_cl
}

fn create_buffer<T, I>(
    queue: Arc<Queue>,
    memory_allocator: Arc<dyn MemoryAllocator>,
    cb_allocator: &StandardCommandBufferAllocator,
    usage: BufferUsage,
    iter: I,
    len: u64,
    filled: bool,
) -> Subbuffer<[T]>
where
    T: BufferContents,
    I: IntoIterator<Item = T>,
    I::IntoIter: ExactSizeIterator,
{
    let buffer: Subbuffer<[T]> = Buffer::new_slice::<T>(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
            ..Default::default()
        },
        len,
    )
    .expect("failed to create buffer");

    if filled {

        let staging_buffer: Subbuffer<[T]> = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            iter,
        )
        .expect("failed to create staging_buffer");

        let mut cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
            AutoCommandBufferBuilder::primary(
                cb_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .expect("failed to create cbb");

        cbb
            .copy_buffer(CopyBufferInfo::buffers(
                staging_buffer.clone(),
                buffer.clone(),
            ))
            .unwrap();

        let copy_command_buffer: Arc<PrimaryAutoCommandBuffer> = cbb.build().unwrap();

        copy_command_buffer
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

    }

    buffer
}

impl Vk {

    /// Updates the buffers used for drawing using the compute shaders
    fn update(&mut self) {

    }

    /// Draws a frame
    fn draw(&mut self) {
        let image_extent: [u32; 2] = self.window.inner_size().into();

        if image_extent.contains(&0) {
            return;
        }

        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swapchain {
            self.recreate_swapchain = false;

            (self.swapchain, self.images) = self
                .swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent,
                    ..self.swapchain.create_info()
                })
                .expect("failed to recreate swapchain");

            let extent = self.images[0].extent();
            

            let depth_attachment = ImageView::new_default(
                Image::new(
                    self.memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format: Format::D16_UNORM,
                        extent: self.images[0].extent(),
                        usage: ImageUsage::DEPTH_STENCIL_ATTACHMENT | ImageUsage::TRANSIENT_ATTACHMENT,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap();

            self.framebuffers = self
                .images
                .iter()
                .map(|image| {
                    let view = ImageView::new_default(image.clone()).unwrap();
                    Framebuffer::new(
                        self.render_pass.clone(),
                        FramebufferCreateInfo {
                            attachments: vec![view, depth_attachment.clone()],
                            ..Default::default()
                        },
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();

            // In the triangle example we use a dynamic viewport, as its a simple example. However in the
            // teapot example, we recreate the pipelines with a hardcoded viewport instead. This allows the
            // driver to optimize things, at the cost of slower window resizes.
            // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
            self.draw_pipeline = {
                let vertex_input_state = BaseVertex::per_vertex()
                    .definition(&self.drawing_vs.info().input_interface)
                    .unwrap();
                let stages = [
                    PipelineShaderStageCreateInfo::new(self.drawing_vs.clone()),
                    PipelineShaderStageCreateInfo::new(self.drawing_fs.clone()),
                ];
                let layout = PipelineLayout::new(
                    self.device.clone(),
                    PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                        .into_pipeline_layout_create_info(self.device.clone())
                        .unwrap(),
                )
                .unwrap();
                let subpass = Subpass::from(self.render_pass.clone(), 0).unwrap();

                GraphicsPipeline::new(
                    self.device.clone(),
                    None,
                    GraphicsPipelineCreateInfo {
                        stages: stages.into_iter().collect(),
                        vertex_input_state: Some(vertex_input_state),
                        input_assembly_state: Some(InputAssemblyState::default()),
                        viewport_state: Some(ViewportState {
                            viewports: [Viewport {
                                offset: [0.0, 0.0],
                                extent: [extent[0] as f32, extent[1] as f32],
                                depth_range: 0.0..=1.0,
                            }]
                            .into_iter()
                            .collect(),
                            ..Default::default()
                        }),
                        rasterization_state: Some(RasterizationState::default()),
                        multisample_state: Some(MultisampleState::default()),
                        color_blend_state: Some(ColorBlendState::with_attachment_states(
                            subpass.num_color_attachments(),
                            ColorBlendAttachmentState {
                                blend: Some(AttachmentBlend::alpha()),
                                ..Default::default()
                            },
                        )),
                        subpass: Some(subpass.into()),
                        depth_stencil_state: Some(DepthStencilState {
                            depth: Some(DepthState::simple()),
                            ..Default::default()
                        }),
                        ..GraphicsPipelineCreateInfo::layout(layout)
                    },
                )
                .unwrap()
            };
        }

        let (image_i, suboptimal, acquire_future) =
            match acquire_next_image(self.swapchain.clone(), None).map_err(Validated::unwrap) {
                Ok(r) => r,
                Err(VulkanError::OutOfDate) => {
                    self.recreate_swapchain = true;
                    return;
                }
                Err(e) => panic!("failed to acquire next image: {e}"),
            };

        if suboptimal {
            self.recreate_swapchain = true;
        }

        let mut mat: Mat4 = self.general_push_transformer.0.update_vp(self.time);
        mat.mult(self.general_push_transformer.1.update_vp(self.time));
        let push_const = VSGeneral {
            mat: mat.0,
        };

        let mut draw_builder = AutoCommandBufferBuilder::primary(
            &self.cb_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        draw_builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some(self.background_color.into()),
                        Some(1.0.into())
                    ],
                    ..RenderPassBeginInfo::framebuffer(self.framebuffers[image_i as usize].clone())
                },
                Default::default(), //vulkano::command_buffer::SubpassBeginInfo { contents: SubpassContents::Inline, ..Default::default() },
            )
            .unwrap()
            .bind_pipeline_graphics(self.draw_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.draw_pipeline.layout().clone(),
                self.draw_descriptor_set_layout_index as u32,
                self.draw_descriptor_set.clone(),
            )
            .unwrap()
            .bind_vertex_buffers(0, self.vsinput_buffer.clone())
            .unwrap()
            .push_constants(self.draw_pipeline.layout().clone(), 0, push_const)
            .unwrap()
            .draw(self.vsinput_buffer.len() as u32, 1, 0, 0)
            .unwrap()
            .end_render_pass(Default::default())
            .unwrap();

        let draw_command_buffer = draw_builder.build().unwrap();
        

        let mut compute_builder = AutoCommandBufferBuilder::primary(
            &self.cb_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        compute_builder
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0,
                self.compute_descriptor_set.clone(),
            )
            .unwrap();

        let compute_command_buffer = compute_builder.build().unwrap();

        // self.uniform_copy_command_buffer.clone().execute(
        //     self.queue.clone(),
        // )
        // .unwrap()
        // .then_signal_fence_and_flush().unwrap();

        let future = self
            .previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
            .then_execute(self.queue.clone(), compute_command_buffer)
            .unwrap()
            .then_execute(self.queue.clone(), draw_command_buffer)
            .unwrap()
            .then_swapchain_present(
                self.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(self.swapchain.clone(), image_i),
            )
            .then_signal_fence_and_flush();

        match future.map_err(Validated::unwrap) {
            Ok(future) => {
                self.previous_frame_end = Some(future.boxed());
            }
            Err(VulkanError::OutOfDate) => {
                self.recreate_swapchain = true;
                self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
            Err(e) => {
                panic!("failed to flush future: {e}");
                // self.previous_frame_end = Some(sync::now(self.device.clone()).boxed());
            }
        }
    }

    /// Runs the animation in the window in real-time.
    fn show(
        &mut self,
        event_loop: &mut EventLoop<()>,
        // save: (u32, u32),
    ) -> i32 {
        println!("♻ --- {}: Showing with updated data.", self.show_count);
        let start = Instant::now();
        let mut max_elapsed: bool = true;
        event_loop.run_return(move |event, _, control_flow| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::ExitWithCode(1);
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                self.recreate_swapchain = true;
            }
            Event::MainEventsCleared => {
                self.time = start.elapsed().as_secs_f32() + self.start_time;
                // if elements.ended() {
                //     *control_flow = ControlFlow::ExitWithCode(0);
                // }
                if self.time > self.end_time {
                    if max_elapsed {
                        max_elapsed = false;
                        self.show_count += 1;
                    }
                    *control_flow = ControlFlow::ExitWithCode(0);
                }
                self.update();
                self.draw();
                // if save.0 > 0 && save.0 > 0 { self.encode(); }
            }
            _ => (),
        })
    }

    // /// Encodes a frame to the output video, for saving.
    // fn encode(&mut self) {
    //     unimplemented!("Cannot encode yet!");
    // }
}
