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

use vera::{Input, Model, Tf, Vertex as ModelVertex, View, Projection};
use vulkano::descriptor_set::layout::DescriptorSetLayout;
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

use vulkano::buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer};
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
    StandardMemoryAllocator,
};
use vulkano::format::Format;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
    PipelineShaderStageCreateInfo,
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
    command_buffer_allocator: StandardCommandBufferAllocator,
    descriptor_set_allocator: StandardDescriptorSetAllocator,

    
    max_uniform_buffer_size: u32,
    max_storage_buffer_size: u32,


    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swapchain: bool,
    show_count: u32,
    time: f32,

    
    // -----
    position_vertex_data: Vec<VertexData>,
    vertex_len: u64,

    vertex_buffer: Subbuffer<[VertexData]>,
    staging_vertex_buffer: Subbuffer<[VertexData]>,
    vertex_copy_command_buffer: Arc<PrimaryAutoCommandBuffer>,
    vertex_copy_fence:
        Option<Arc<FenceSignalFuture<CommandBufferExecFuture<sync::future::NowFuture>>>>,


    color_vertex_data: Vec<ColorVertexData>,
    color_vertex_colorizer: Vec<Colorizer>,
    
    color_vertex_buffer: Subbuffer<[ColorVertexData]>,
    staging_color_vertex_buffer: Subbuffer<[ColorVertexData]>,
    color_vertex_copy_command_buffer: Arc<PrimaryAutoCommandBuffer>,
    color_vertex_copy_fence:
        Option<Arc<FenceSignalFuture<CommandBufferExecFuture<sync::future::NowFuture>>>>,


    transform_vertex_data: Vec<TransformVertexData>,
    transform_vertex_transformer: Vec<Transformer>,

    transform_vertex_buffer: Subbuffer<[TransformVertexData]>,
    staging_transform_vertex_buffer: Subbuffer<[TransformVertexData]>,
    transform_vertex_copy_command_buffer: Arc<PrimaryAutoCommandBuffer>,
    transform_vertex_copy_fence:
        Option<Arc<FenceSignalFuture<CommandBufferExecFuture<sync::future::NowFuture>>>>,

    // -----
    general_uniform_data: GeneralData,
    general_uniform_transformer: (Transformer, Transformer),
    
    general_uniform_buffer: Subbuffer<GeneralData>,
    general_staging_uniform_buffer: Subbuffer<GeneralData>,
    general_uniform_copy_command_buffer: Arc<PrimaryAutoCommandBuffer>,
    general_uniform_copy_fence:
        Option<Arc<FenceSignalFuture<CommandBufferExecFuture<sync::future::NowFuture>>>>,

    // -----
    entities_uniform_data: Vec<EntityData>,
    entities_uniform_len: u64,
    entities_uniform_transformer: Vec<Transformer>,

    entities_uniform_buffer: Subbuffer<[EntityData]>,
    entities_staging_uniform_buffer: Subbuffer<[EntityData]>,
    entities_uniform_copy_command_buffer: Arc<PrimaryAutoCommandBuffer>,
    entities_uniform_copy_fence:
        Option<Arc<FenceSignalFuture<CommandBufferExecFuture<sync::future::NowFuture>>>>,

    // -----
    // uniform_update_cs: Arc<ShaderModule>,
    // uniform_update_pipeline: Arc<ComputePipeline>,
    // uniform_update_command_buffer: Arc<PrimaryAutoCommandBuffer>,

    // -----
    drawing_vs: EntryPoint,
    drawing_fs: EntryPoint,
    drawing_pipeline: Arc<GraphicsPipeline>,
    descriptor_set: Arc<PersistentDescriptorSet>,
    descriptor_set_layout: Arc<DescriptorSetLayout>,
    descriptor_set_layout_index: usize,

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


    background_color: [f32; 4],
    start_time: f32,
    end_time: f32,
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: r"
            #version 460

            struct uGeneralData {
                mat4 projection_matrix;
                mat4 view_matrix;
                vec2 resolution;
                float time;
            };

            struct uEntityData {
                mat4 model_matrix;
            };
            
            layout(set = 0, binding = 0) buffer GeneralData {
                uGeneralData data;
            } general_buf;
            
            layout(set = 0, binding = 1) buffer EntityData {
                uEntityData data[];
            } entities_buf;

            layout(location = 0) in uint entity_id;
            layout(location = 1) in vec3 position;
            layout(location = 2) in vec4 color;
            layout(location = 3) in vec4 vertex_matrix0;
            layout(location = 4) in vec4 vertex_matrix1;
            layout(location = 5) in vec4 vertex_matrix2;
            layout(location = 6) in vec4 vertex_matrix3;

            layout(location = 0) out vec4 out_color;

            void main() {
                out_color = color;

                mat4 transform = mat4(vertex_matrix0, vertex_matrix1, vertex_matrix2, vertex_matrix3);
                vec4 uv = vec4(position, 1.0) * transform * entities_buf.data[entity_id].model_matrix * general_buf.data.view_matrix * general_buf.data.projection_matrix;

                                if (general_buf.data.resolution.x > general_buf.data.resolution.y) {
                    uv.x *= general_buf.data.resolution.y/general_buf.data.resolution.x;
                                    } else {
                    uv.y *= general_buf.data.resolution.x/general_buf.data.resolution.y;
                }

                gl_Position = uv;
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

// TODOCOMPUTEUPDATE
// mod cs {
//     vulkano_shaders::shader! {
//         ty: "compute",
//         src: r"
//             #version 460
//
//             layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
//
//             struct uData {
//                 mat3 model_matrix;
//             };
//
//             layout(set = 0, binding = 0) buffer EntityData {
//                 uData data[];
//             } buf;
//
//             void main() {
//                 uint idx = gl_GlobalInvocationID.x;
//                 buf.model_matrix[idx] *= 12;
//             }
//         ",
//     }
// }

impl Vera {
    /// Sets up Vera with Vulkan
    /// - `input` define all Shapes
    pub fn create(input: Input) -> Self {
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
        let command_buffer_allocator =
            StandardCommandBufferAllocator::new(device.clone(), Default::default());
        let descriptor_set_allocator =
            StandardDescriptorSetAllocator::new(device.clone(), Default::default());

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

        // Max buffer sizes
        // ----------------
        // // If the entities fit (inlcuding the other uniforms), use UBO, otherwise use SSBO
        let max_uniform_buffer_size: u32 = physical_device.properties().max_uniform_buffer_range;
        let max_storage_buffer_size: u32 = physical_device.properties().max_storage_buffer_range;
        println!(
            "max_uniform_buffer_size: {}\nmax_storage_buffer_size: {}\n",
            max_uniform_buffer_size, max_storage_buffer_size
        );

        // ----------------

        // Initial buffer data, len and transform.
        // ---------------------------------------

        let (
            (general_uniform_data, general_uniform_transformer), 
            (entities_uniform_len, entities_uniform_data, entities_uniform_transformer), 
            (transform_vertex_data, transform_vertex_transformer), 
            (color_vertex_data, color_vertex_colorizer), 
            (vertex_len, position_vertex_data),
            (background_color, start_time, end_time)
        ) = from_input(input);
        // ---------------------------------------

        // Staging & Device-local vertex buffers, and their copy & update command buffers  // TODOCOMPUTEUPDATE
        // ------------------------------------------------------------------------------
        let staging_vertex_buffer: Subbuffer<[VertexData]> = Buffer::from_iter(
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
            position_vertex_data,
        )
        .expect("failed to create staging_vertex_buffer");

        let vertex_buffer: Subbuffer<[VertexData]> = Buffer::new_slice::<VertexData>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            vertex_len,
        )
        .expect("failed to create vertex_buffer");

        let mut vertex_cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
            AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .expect("failed to create vertex_cbb");

        vertex_cbb
            .copy_buffer(CopyBufferInfo::buffers(
                staging_vertex_buffer.clone(),
                vertex_buffer.clone(),
            ))
            .unwrap();

        let vertex_copy_command_buffer: Arc<PrimaryAutoCommandBuffer> = vertex_cbb.build().unwrap();

        vertex_copy_command_buffer
            .clone()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

        let vertex_copy_fence: Option<
            Arc<FenceSignalFuture<CommandBufferExecFuture<sync::future::NowFuture>>>,
        > = None;

        
// -----------------------------------------

        let staging_color_vertex_buffer: Subbuffer<[ColorVertexData]> = Buffer::from_iter(
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
            color_vertex_data,
        )
        .expect("failed to create staging_color_vertex_buffer");

        let color_vertex_buffer: Subbuffer<[ColorVertexData]> = Buffer::new_slice::<ColorVertexData>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            vertex_len,
        )
        .expect("failed to create color_vertex_buffer");

        let mut color_vertex_cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
            AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .expect("failed to create color_vertex_cbb");
        
        color_vertex_cbb
            .copy_buffer(CopyBufferInfo::buffers(
                staging_color_vertex_buffer.clone(),
                color_vertex_buffer.clone(),
            ))
            .unwrap();
        
        let color_vertex_copy_command_buffer: Arc<PrimaryAutoCommandBuffer> = color_vertex_cbb.build().unwrap();
        
        color_vertex_copy_command_buffer
            .clone()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();
        
        let color_vertex_copy_fence: Option<
            Arc<FenceSignalFuture<CommandBufferExecFuture<sync::future::NowFuture>>>,
        > = None;

// -----------------------------------------


        let staging_transform_vertex_buffer: Subbuffer<[TransformVertexData]> = Buffer::from_iter(
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
            transform_vertex_data,
        )
        .expect("failed to create staging_transform_vertex_buffer");

        let transform_vertex_buffer: Subbuffer<[TransformVertexData]> = Buffer::new_slice::<TransformVertexData>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            vertex_len,
        )
        .expect("failed to create transform_vertex_buffer");

        let mut transform_vertex_cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
            AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .expect("failed to create transform_vertex_cbb");

        transform_vertex_cbb
            .copy_buffer(CopyBufferInfo::buffers(
                staging_transform_vertex_buffer.clone(),
                transform_vertex_buffer.clone(),
            ))
            .unwrap();

        let transform_vertex_copy_command_buffer: Arc<PrimaryAutoCommandBuffer> = transform_vertex_cbb.build().unwrap();

        transform_vertex_copy_command_buffer
            .clone()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

        let transform_vertex_copy_fence: Option<
            Arc<FenceSignalFuture<CommandBufferExecFuture<sync::future::NowFuture>>>,
        > = None;

        // ------------------------------------------------------------------------------

        // Staging & Device-local uniform buffers for general data, and their copy command buffers
        // ---------------------------------------------------------------------------------------
        let general_staging_uniform_buffer = Buffer::from_data(
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
            general_uniform_data.clone(),
        )
        .expect("failed to create staging_uniform_buffer");

        let general_uniform_buffer = Buffer::new_sized::<GeneralData>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::UNIFORM_BUFFER
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .expect("failed to create uniform_buffer");

        let mut general_uniform_copy_cbb = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .expect("failed to create uniform_copy_cbb");

        general_uniform_copy_cbb
            .copy_buffer(CopyBufferInfo::buffers(
                general_staging_uniform_buffer.clone(),
                general_uniform_buffer.clone(),
            ))
            .unwrap();

        let general_uniform_copy_command_buffer = general_uniform_copy_cbb.build().unwrap();

        general_uniform_copy_command_buffer
            .clone()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

        let general_uniform_copy_fence: Option<
            Arc<FenceSignalFuture<CommandBufferExecFuture<sync::future::NowFuture>>>,
        > = None;

        // ---------------------------------------------------------------------------------------

        // Staging & Device-local uniform buffers for entities, and their copy & update command buffers  // TODOCOMPUTEUPDATE
        // --------------------------------------------------------------------------------------------

        let entities_staging_uniform_buffer = Buffer::from_iter(
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
            entities_uniform_data,
        )
        .expect("failed to create staging_uniform_buffer");

        let entities_uniform_buffer = Buffer::new_slice::<EntityData>(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::UNIFORM_BUFFER
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            entities_uniform_len,
        )
        .expect("failed to create uniform_buffer");

        let mut entities_uniform_copy_cbb = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .expect("failed to create uniform_copy_cbb");

        entities_uniform_copy_cbb
            .copy_buffer(CopyBufferInfo::buffers(
                entities_staging_uniform_buffer.clone(),
                entities_uniform_buffer.clone(),
            ))
            .unwrap();

        let entities_uniform_copy_command_buffer = entities_uniform_copy_cbb.build().unwrap();

        entities_uniform_copy_command_buffer
            .clone()
            .execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

        let entities_uniform_copy_fence: Option<
            Arc<FenceSignalFuture<CommandBufferExecFuture<sync::future::NowFuture>>>,
        > = None;

        /*
                let uniform_update_cs =
                    cs::load(device.clone()).expect("failed to create compute shader module");

                let uniform_update_pipeline = ComputePipeline::new(
                    device.clone(),
                    uniform_update_cs.entry_point("main").unwrap(),
                    &(),
                    None,
                    |_| {},
                )
                .expect("failed to create compute pipeline");

                let mut uniform_update_cbb = AutoCommandBufferBuilder::primary(
                    &command_buffer_allocator,
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .expect("failed to create uniform_update_cbb");

                // let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
                // let pipeline_layout = uniform_update_pipeline.layout();
                // let descriptor_set_layouts = pipeline_layout.set_layouts();
                //
                // let descriptor_set_layout_index = 0;
                // let descriptor_set_layout = descriptor_set_layouts
                //     .get(descriptor_set_layout_index)
                //     .unwrap();
                // let descriptor_set = PersistentDescriptorSet::new(
                //     &descriptor_set_allocator,
                //     descriptor_set_layout.clone(),
                //     [WriteDescriptorSet::buffer(
                //         0,
                //         staging_uniform_buffer.clone(),
                //     )], // 0 is the binding
                // )
                // .unwrap();

                uniform_update_cbb
                    .copy_buffer(CopyBufferInfo::buffers(
                        staging_uniform_buffer,
                        uniform_buffer.clone(),
                    ))
                    .unwrap();

                let uniform_update_command_buffer = Arc::new(uniform_update_cbb.build().unwrap());
                uniform_update_command_buffer
                    .execute(queue.clone())
                    .unwrap()
                    .then_signal_fence_and_flush()
                    .unwrap()
                    .wait(None /* timeout */)
                    .unwrap();

        */

        // --------------------------------------------------------------------------------------------

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

        let drawing_pipeline: Arc<GraphicsPipeline> = {
            let vertex_input_state = [VertexData::per_vertex(), ColorVertexData::per_vertex(), TransformVertexData::per_vertex()]
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

        let pipeline_layout: &Arc<vulkano::pipeline::PipelineLayout> = drawing_pipeline.layout();
        let descriptor_set_layouts: &[Arc<vulkano::descriptor_set::layout::DescriptorSetLayout>] =
            pipeline_layout.set_layouts();

        let descriptor_set_layout_index: usize = 0;
        let descriptor_set_layout: Arc<vulkano::descriptor_set::layout::DescriptorSetLayout> =
            descriptor_set_layouts
                .get(descriptor_set_layout_index)
                .unwrap()
                .clone();
        let descriptor_set: Arc<PersistentDescriptorSet> = PersistentDescriptorSet::new(
            &descriptor_set_allocator,
            descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, general_uniform_buffer.clone()),
                WriteDescriptorSet::buffer(1, entities_uniform_buffer.clone()),
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

                // -----
                memory_allocator,
                command_buffer_allocator,
                descriptor_set_allocator,

                // -----
                max_uniform_buffer_size,
                max_storage_buffer_size,

                // -----
                vertex_buffer,
                staging_vertex_buffer,
                vertex_copy_command_buffer,
                vertex_copy_fence,

                // -----
                general_uniform_buffer,
                general_staging_uniform_buffer,
                general_uniform_copy_command_buffer,
                general_uniform_copy_fence,

                // -----
                entities_uniform_buffer,
                entities_staging_uniform_buffer,
                entities_uniform_copy_command_buffer,
                entities_uniform_copy_fence,

                // -----

                // -----
                // uniform_update_cs,
                // uniform_update_pipeline,
                // uniform_update_command_buffer,

                // -----
                drawing_vs,
                drawing_fs,
                descriptor_set,
                descriptor_set_layout,
                descriptor_set_layout_index,
                drawing_pipeline,
                frames_in_flight,
                drawing_fences,
                previous_drawing_fence_i,

                // -----
                previous_frame_end,
                recreate_swapchain,
                show_count,
                time,

                // -----
                position_vertex_data: vec![],
                vertex_len,
                general_uniform_data,
                general_uniform_transformer,
                entities_uniform_data: vec![],
                entities_uniform_transformer,
                entities_uniform_len,
                transform_vertex_data: vec![],
                transform_vertex_transformer,
                transform_vertex_buffer,
                staging_transform_vertex_buffer,
                transform_vertex_copy_command_buffer,
                transform_vertex_copy_fence,
                color_vertex_data: vec![],
                color_vertex_colorizer,
                color_vertex_buffer,
                staging_color_vertex_buffer,
                color_vertex_copy_command_buffer,
                color_vertex_copy_fence,

                background_color,
                start_time,
                end_time,
            },
        }
    }

    /// Resets vertex and uniform data from input, which is consumed.
    /// 
    /// In heavy animations, resetting is useful for reducing the CPU/GPU usage if you split you animations in several parts.
    /// Also useful if called from a loop for hot-reloading.
    pub fn reset(&mut self, input: Input) {
        self.vk.source(input);
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

/// Treats `input`, and returns related data, transformers and lengths.
fn from_input(input: Input) -> ((GeneralData, (Transformer, Transformer)), (u64, Vec<EntityData>, Vec<Transformer>), (Vec<TransformVertexData>, Vec<Transformer>), (Vec<ColorVertexData>, Vec<Colorizer>), (u64, Vec<VertexData>), ([f32; 4], f32, f32))  {
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

    let general_uniform_transformer: (Transformer, Transformer) = (Transformer::from_t(view_t), Transformer::from_t(projection_t)); // (View, Projection)
    let general_uniform_data: GeneralData = GeneralData::from_resolution([1.0, 1.0]);

    let mut position_vertex_data: Vec<VertexData> = vec![];
    let mut color_vertex_data: Vec<ColorVertexData> = vec![];
    let mut color_vertex_colorizer: Vec<Colorizer> = vec![];
    let mut transform_vertex_transformer: Vec<Transformer> = vec![];
    let mut entities_uniform_transformer: Vec<Transformer> = vec![];

    for (entity_id, model) in m.into_iter().enumerate() {
        for v in model.vertices.into_iter() {
            position_vertex_data.push(VertexData { entity_id: entity_id as u32, position: v.position });
            color_vertex_data.push(ColorVertexData { color: v.color });
            color_vertex_colorizer.push(Colorizer::from_c(v.color, v.c));
            transform_vertex_transformer.push(Transformer::from_t(v.t));
        }
        entities_uniform_transformer.push(Transformer::from_t(model.t));
    }

    let entities_uniform_len = entities_uniform_transformer.len() as u64;
    let entities_uniform_data: Vec<EntityData> = vec![EntityData::new(); entities_uniform_len as usize];
    let vertex_len = position_vertex_data.len() as u64;

    let transform_vertex_data: Vec<TransformVertexData> = vec![TransformVertexData::new(); vertex_len as usize];

    (
        (general_uniform_data, general_uniform_transformer), 
        (entities_uniform_len, entities_uniform_data, entities_uniform_transformer), 
        (transform_vertex_data, transform_vertex_transformer), 
        (color_vertex_data, color_vertex_colorizer), 
        (vertex_len, position_vertex_data),
        (meta.bg, meta.start, meta.end)
    )
}

impl Vk {
    /// Resets all content to `input`.
    fn source(&mut self, input: Input) {
        (
            (self.general_uniform_data, self.general_uniform_transformer), 
            (self.entities_uniform_len, self.entities_uniform_data, self.entities_uniform_transformer), 
            (self.transform_vertex_data, self.transform_vertex_transformer), 
            (self.color_vertex_data, self.color_vertex_colorizer), 
            (self.vertex_len, self.position_vertex_data),
            (self.background_color, self.start_time, self.end_time)
        ) = from_input(input);

        self.recreate_vertex_buffer();
        self.recreate_color_vertex_buffer();
        self.recreate_transform_vertex_buffer();
        self.recreate_entities_uniform_buffer();
        self.recreate_general_uniform_buffer();

        self.descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            self.descriptor_set_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, self.general_uniform_buffer.clone()),
                WriteDescriptorSet::buffer(1, self.entities_uniform_buffer.clone()),
            ],
            [],
        )
        .unwrap();

        self.copy_vertex_buffer();
        self.copy_color_vertex_buffer();
        self.copy_transform_vertex_buffer();
        self.copy_entities_uniform_buffer();
        self.copy_general_uniform_buffer();
    }

    // Buffer recreation

    fn recreate_vertex_buffer(&mut self) {
        self.staging_vertex_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            std::mem::take(&mut self.position_vertex_data),
        )
        .expect("failed to create staging_vertex_buffer");

        self.vertex_buffer = Buffer::new_slice::<VertexData>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            self.vertex_len,
        )
        .expect("failed to create vertex_buffer");

        let mut vertex_cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
            AutoCommandBufferBuilder::primary(
                &self.command_buffer_allocator,
                self.queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .expect("failed to create vertex_cbb");

        vertex_cbb
            .copy_buffer(CopyBufferInfo::buffers(
                self.staging_vertex_buffer.clone(),
                self.vertex_buffer.clone(),
            ))
            .unwrap();

        self.vertex_copy_command_buffer = vertex_cbb.build().unwrap();
    }

    fn recreate_color_vertex_buffer(&mut self) {
        self.staging_color_vertex_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            std::mem::take(&mut self.color_vertex_data),
        )
        .expect("failed to create staging_color_vertex_buffer");

        self.color_vertex_buffer = Buffer::new_slice::<ColorVertexData>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            self.vertex_len,
        )
        .expect("failed to create color_vertex_buffer");

        let mut color_vertex_cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
            AutoCommandBufferBuilder::primary(
                &self.command_buffer_allocator,
                self.queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .expect("failed to create color_vertex_cbb");
        
        color_vertex_cbb
            .copy_buffer(CopyBufferInfo::buffers(
                self.staging_color_vertex_buffer.clone(),
                self.color_vertex_buffer.clone(),
            ))
            .unwrap();
        
        self.color_vertex_copy_command_buffer = color_vertex_cbb.build().unwrap();
    }

    fn recreate_transform_vertex_buffer(&mut self) {
        self.staging_transform_vertex_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            std::mem::take(&mut self.transform_vertex_data),
        )
        .expect("failed to create staging_vertex_buffer");

        self.transform_vertex_buffer = Buffer::new_slice::<TransformVertexData>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::TRANSFER_DST
                    | BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            self.vertex_len,
        )
        .expect("failed to create vertex_buffer");

        let mut transform_vertex_cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
            AutoCommandBufferBuilder::primary(
                &self.command_buffer_allocator,
                self.queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .expect("failed to create transform_vertex_cbb");

        transform_vertex_cbb
            .copy_buffer(CopyBufferInfo::buffers(
                self.staging_transform_vertex_buffer.clone(),
                self.transform_vertex_buffer.clone(),
            ))
            .unwrap();

        self.transform_vertex_copy_command_buffer = transform_vertex_cbb.build().unwrap();
    }

    fn recreate_general_uniform_buffer(&mut self) {
        self.general_staging_uniform_buffer = Buffer::from_data(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.general_uniform_data.clone(),
        )
        .expect("failed to create staging_uniform_buffer");

        self.general_uniform_buffer = Buffer::new_sized::<GeneralData>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::UNIFORM_BUFFER
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .expect("failed to create uniform_buffer");

        let mut general_uniform_cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
            AutoCommandBufferBuilder::primary(
                &self.command_buffer_allocator,
                self.queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .expect("failed to create general_uniform_cbb");

        general_uniform_cbb
            .copy_buffer(CopyBufferInfo::buffers(
                self.general_staging_uniform_buffer.clone(),
                self.general_uniform_buffer.clone(),
            ))
            .unwrap();

        self.general_uniform_copy_command_buffer = general_uniform_cbb.build().unwrap();
    }

    fn recreate_entities_uniform_buffer(&mut self) {
        self.entities_staging_uniform_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            std::mem::take(&mut self.entities_uniform_data),
        )
        .expect("failed to create staging_uniform_buffer");

        self.entities_uniform_buffer = Buffer::new_slice::<EntityData>(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER
                    | BufferUsage::UNIFORM_BUFFER
                    | BufferUsage::TRANSFER_DST,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            self.entities_uniform_len,
        )
        .expect("failed to create uniform_buffer");

        let mut entities_uniform_cbb: AutoCommandBufferBuilder<PrimaryAutoCommandBuffer> =
            AutoCommandBufferBuilder::primary(
                &self.command_buffer_allocator,
                self.queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .expect("failed to create entities_uniform_cbb");

        entities_uniform_cbb
            .copy_buffer(CopyBufferInfo::buffers(
                self.entities_staging_uniform_buffer.clone(),
                self.entities_uniform_buffer.clone(),
            ))
            .unwrap();

        self.entities_uniform_copy_command_buffer = entities_uniform_cbb.build().unwrap();
    }

    // Staging buffer update & copy to device-local memory

    // fn update_vertex_buffer(&mut self) {} // Set once

    fn update_color_vertex_buffer(&mut self) {
        for (o, i) in self.staging_color_vertex_buffer.write().unwrap().iter_mut()
            .zip(self.color_vertex_colorizer.iter_mut()) {
                (*o).color = i.update(self.time).0;
        }

        self.copy_color_vertex_buffer();
    }

    fn update_transform_vertex_buffer(&mut self) {
        for (o, i) in self.staging_transform_vertex_buffer.write().unwrap().iter_mut()
            .zip(self.transform_vertex_transformer.iter_mut()) {
                let mat = i.update_vm(self.time).0;
                ((*o).vertex_matrix0, (*o).vertex_matrix1, (*o).vertex_matrix2, (*o).vertex_matrix3) = (
                    [mat[0], mat[1], mat[2], mat[3]],
                    [mat[4], mat[5], mat[6], mat[7]],
                    [mat[8], mat[9], mat[10], mat[11]],
                    [mat[12], mat[13], mat[14], mat[15]],
                );
        }

        self.copy_transform_vertex_buffer();
    }

    fn update_general_uniform_buffer(&mut self) {
        let image_extent: [u32; 2] = self.window.inner_size().into();
        {
            let mut buf = self.general_staging_uniform_buffer.write().unwrap();
            buf.time = self.time;
            buf.resolution = [image_extent[0] as f32, image_extent[1] as f32];
            buf.view_matrix = self.general_uniform_transformer.0.update_vp(self.time).0;
            buf.projection_matrix = self.general_uniform_transformer.1.update_vp(self.time).0;
        }
        
        self.copy_general_uniform_buffer();
    }

    fn update_entities_uniform_buffer(&mut self) {
        for (o, i) in self.entities_staging_uniform_buffer.write().unwrap().iter_mut()
            .zip(self.entities_uniform_transformer.iter_mut()) {
            (*o).model_matrix = i.update_vm(self.time).0;
        }

        self.copy_entities_uniform_buffer();
    }

    // Buffer copy

    fn copy_vertex_buffer(&mut self) {
        self.vertex_copy_command_buffer
            .clone()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();
    }

    fn copy_color_vertex_buffer(&mut self) {
        self.color_vertex_copy_command_buffer
            .clone()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();
    }

    fn copy_transform_vertex_buffer(&mut self) { // Add fences to these methods
        self.transform_vertex_copy_command_buffer
            .clone()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();
    }

    fn copy_general_uniform_buffer(&mut self) {
        self.general_uniform_copy_command_buffer
            .clone()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();
    }

    fn copy_entities_uniform_buffer(&mut self) {
        self.entities_uniform_copy_command_buffer
            .clone()
            .execute(self.queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();
    }

    /// Updates the staging buffers and copies their data to the device-local buffers
    fn update(&mut self) {
        self.update_color_vertex_buffer();
        self.update_transform_vertex_buffer();
        self.update_general_uniform_buffer();
        self.update_entities_uniform_buffer();
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
            self.drawing_pipeline = {
                let vertex_input_state = [VertexData::per_vertex(), ColorVertexData::per_vertex(), TransformVertexData::per_vertex()]
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

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            self.queue.queue_family_index(),
            CommandBufferUsage::MultipleSubmit,
        )
        .unwrap();

        builder
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
            .bind_pipeline_graphics(self.drawing_pipeline.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.drawing_pipeline.layout().clone(),
                self.descriptor_set_layout_index as u32,
                self.descriptor_set.clone(),
            )
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap()
            .bind_vertex_buffers(1, self.color_vertex_buffer.clone())
            .unwrap()
            .bind_vertex_buffers(2, self.transform_vertex_buffer.clone())
            .unwrap()
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap()
            .end_render_pass(Default::default())
            .unwrap();

        let drawing_command_buffer = builder.build().unwrap();

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
            .then_execute(self.queue.clone(), drawing_command_buffer)
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
